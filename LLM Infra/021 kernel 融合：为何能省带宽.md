[[021 kernel 融合：为何能省带宽]] 是把多个连续算子合并进**一个 kernel**,让中间结果留在寄存器 / [[003 GPU 内存层级：HBM、L2、SRAM、寄存器|SRAM]] 里、不写回 HBM,从而省掉中间张量的全部 HBM 读写。对 memory-bound 算子,这是最直接的提速手段——[[024 FlashAttention 1、2：IO 感知精确注意力|FlashAttention]] 本质就是把整个注意力融成一个 kernel。

## 直觉类比
做菜:朴素方式是切完菜放回冰箱、炒之前再取出、炒完又放回、装盘前再取——每一步都开关冰箱(HBM)。融合就是切→炒→装盘一气呵成,食材全程在案板(寄存器/SRAM)上,冰箱只开两次:取原料、放成品。开关冰箱(HBM 往返)正是 memory-bound 算子的主要耗时。

## 小数字例子
算 `y = relu(a*x + b)`,x 有 $N$ 个元素(每个 4B):
- **朴素 3 个 kernel**:`t1=a*x`(读 x、写 t1)、`t2=t1+b`(读 t1、写 t2)、`y=relu(t2)`(读 t2、写 y)。忽略标量 $a,b$ 与缓存命中，HBM 流量为读 $3N$ + 写 $3N$ = **6N** 个元素。
- **融合 1 个 kernel**:读 x、写 y,中间 t1/t2 只在寄存器。HBM 流量 = 读 N + 写 N = **2N**。
- 省下 $(6N-2N)/(6N)=\mathbf{66.7\%}$ 的理想 HBM 流量；对确实被 HBM 主导的链，耗时才可能近似按这个比例下降。

## 原理:算术强度与 IO 主导
逐元素算子的算术强度极低:

$$I = \frac{\text{FLOP}}{\text{访存字节}} \ll \text{机器平衡点}$$

故落在 roofline 的 memory-bound 区,耗时 ≈ HBM 流量 / 带宽。融合把 $k$ 个算子的中间张量读写($\approx 2(k{-}1)$ 趟)压成 0:

$$\text{流量}_{\text{融合}} = \underbrace{N_\text{in} + N_\text{out}}_{\text{只首尾各一次}} \;\ll\; \text{流量}_{\text{未融合}} = N_\text{in}+N_\text{out}+2(k{-}1)N$$

**融合不是越多越好——spilling 反例**。融合要靠寄存器/SRAM 把中间值兜住,而每个 SM 的寄存器是死的(如 64K 个 32-bit 寄存器/SM)。设把 10 个算子硬融成一个 kernel,链上同时活着的中间量多,单线程需 128 个寄存器才放得下;一个 block 256 线程就要 $256\times128=32768$ 个,逼近上限后编译器只能把放不下的寄存器**溢出(spilling)到 local memory**——而 local memory 物理上就在 HBM 里。本想省的 HBM 往返,被 spill 的读写又**补了回去**:省了 $2(k{-}1)N$ 的中间张量流量,却多出 spill 流量,还因寄存器吃紧压低 [[019 CUDA 执行模型：grid、block、warp|occupancy]](能并发的 warp 变少)、藏不住延迟,净收益可能为负甚至更慢。所以融合有**最优粒度**:融到寄存器快溢出就该停,FlashAttention 也是靠分块(tiling)把工作集卡在 SRAM 容量内,而非无脑全融。

## 图
![[kern-融合前后HBM往返对比.png]]

把上面 `y=relu(a·x+b)` 的流量逐算子记成账,6N→2N 一目了然(末尾也标了融合的代价):

![[kern-021融合流量手算表.png]]

## 代码:PyTorch 视角的融合
```python
# ❌ 未融合:每个算子各自一趟 HBM 往返(eager 模式逐个 launch)
t1 = a * x          # 读 x, 写 t1
t2 = t1 + b         # 读 t1, 写 t2
y  = torch.relu(t2) # 读 t2, 写 y      → 中间张量全进出 HBM

# ✅ 候选融合:让编译器尝试把链合成 kernel；用 profiler 验证是否真的融合
@torch.compile
def f(x, a, b):
    return torch.relu(a * x + b)   # 理想情况下:读 x、寄存器内计算、写 y
```
```python
# Triton 里手写融合 kernel 的核心:load 一次、算完链、store 一次
xi = tl.load(x_ptr + offs, mask=mask)
yi = tl.maximum(a * xi + b, 0.0)   # mul→add→relu 全在寄存器
tl.store(y_ptr + offs, yi, mask=mask)
```

## 面试高频
- **融合为什么快?** 省掉中间张量的 HBM 读写;对 memory-bound 算子，收益上界可由减少的流量估算，但需要 profiler 确认编译器实际融合和带宽瓶颈。
- **Q:三段逐元素链为什么是 $6N$ 而非 $7N$?** A:每段都读一个长度 $N$ 的主张量、写一个长度 $N$ 的结果，合计读 $3N$ + 写 $3N$。把第一段错误地算成读两个长度 $N$ 张量会重复计入标量 $a$；若 $a,b$ 真是逐元素向量，则必须在账本中显式加上它们的读取。
- **哪些算子最值得融合?** 逐元素(bias+激活)、归一化(LayerNorm/RMSNorm)、softmax、注意力——都低算术强度、memory-bound。
- **融合的代价?** 寄存器/SRAM 压力上升,可能压低 [[019 CUDA 执行模型：grid、block、warp|occupancy]];融合粒度过大反而溢出寄存器(register spilling)。
- **compute-bound 算子(大 GEMM)融合还有用吗?** 收益小,因为它本就被算力而非带宽卡住。

## 关键事实
- 对标量 $a,b$ 的三段逐元素链，朴素 HBM 流量为 $6N$、理想融合为 $2N$，理论节省 $66.7\%$；向量广播、缓存命中、读写合并和 spilling 会改变实测值。
- torch.compile / TorchInductor、Triton、TensorRT 都支持融合路径，但是否生成一个 kernel 依赖图、版本、形状、设备与编译设置；应以 profiler 轨迹为准。来源:PyTorch `torch.compile` / TorchInductor 官方文档(核验于 2026-07-18)。
- FlashAttention(Dao 2022)是融合思想的巅峰:把 QKᵀ、softmax、×V 融成单 kernel,n×n 中间矩阵全不落 HBM。
