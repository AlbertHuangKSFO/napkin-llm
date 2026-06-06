[[029 何时写自定义 kernel：Nsight 性能分析]] 回答一个工程判断题:**什么时候值得手写/融合 GPU 内核**。答案是"**先 profile,后优化**"——用 **Nsight Systems** 看全局时间线找热点,用 **Nsight Compute** 钻进单个 kernel 看 roofline,判定它是 **compute-bound / memory-bound / latency-bound**,再对症下药。盲目去写 [[028 Triton：用 Python 写 GPU kernel|Triton]] 或 CUDA 内核而不先量瓶颈,往往优化了不是瓶颈的地方,白费功夫。

## 直觉类比
优化 GPU 程序像给堵车的城市修路。不先看交通监控就拍脑袋修路,可能把六车道修成八车道,结果堵点其实在一个红绿灯(同步点)或一座桥(HBM 带宽)。Nsight Systems 是**城市级监控**(哪条路最堵、车在哪干等),Nsight Compute 是**单个路口的高速摄像**(这个路口为什么通行慢)。Roofline 则告诉你这条路的**物理上限**——是车太多(算力),还是进出料口太窄(带宽)。

## 小数字例子
H100:算力(FP16)~989 TFLOPS,HBM 带宽 ~3.35 TB/s,**脊点(ridge point)** 算术强度 = $989\text{e}12 / 3.35\text{e}12 \approx 295$ FLOP/Byte。
- 一个 kernel 实测算术强度 = 50 FLOP/Byte < 295 → 落在 roofline 的**斜线段** → **memory-bound**:加 Tensor Core 没用,该减少 HBM 往返(融合算子、量化)。
- 另一个 GEMM 算术强度 = 400 FLOP/Byte > 295 → 落在**平顶段** → **compute-bound**:上 FP8、改 tiling、提 Tensor Core 利用率才有效。
- 若两者都没打满,实测 FLOPS 和带宽都很低、SM 占用率仅 20% → **latency-bound**:kernel 太小、启动开销/同步占主导,该增大 batch、合并 kernel、提 occupancy。

## 原理:三种 bound 与 roofline
Roofline 把"可达性能"画成算术强度 $I$(FLOP/Byte)的函数,上限是两条屋顶取小:

$$
\text{Perf}(I) = \min\big(\,\underbrace{P_{\text{peak}}}_{\text{算力屋顶}},\ \underbrace{B_{\text{peak}}\cdot I}_{\text{带宽屋顶}}\,\big)
$$

脊点 $I^\* = P_{\text{peak}} / B_{\text{peak}}$ 是分界:$I < I^\*$ 带宽受限,$I > I^\*$ 算力受限。诊断分三类:

- **memory-bound**:DRAM/带宽吞吐接近上限,算术强度低。对策——**算子融合**(把多个小 kernel 合一,省中间张量的 HBM 往返)、**量化**(搬更少字节,见 [[027 量化内核：W4A16、W8A8 GEMM|量化内核]])、提升数据复用。典型:decode、LayerNorm、激活、elementwise。
- **compute-bound**:Tensor Core/FLOPS 接近上限。对策——**低精度**(FP8,见 [[LLM/098 FP8 训练推理与 AQLM 极低比特|FP8]])、更好的 tiling、用 WGMMA 重叠(见 [[025 FlashAttention-3：Hopper TMA、WGMMA 与 FP8|FA3]])。典型:大 batch GEMM、prefill。
- **latency-bound**:两个屋顶都没碰到,SM 占用率低、warp 频繁 stall。对策——提 **occupancy**、增大 batch、减少同步与 kernel 启动次数、用 CUDA Graph 摊薄启动开销。典型:小 kernel、串行依赖、启动开销主导。

工具分工:**Nsight Systems(nsys)** 给**全局时间线**——哪些 kernel 占比最大、有没有空泡、CPU↔GPU 拷贝/同步等待;先定位。**Nsight Compute(ncu)** 给**单 kernel 深剖**——roofline、占用率、warp stall 原因、内存事务效率;再深挖。**先 Systems 后 Compute**,别上来就 ncu 全量(很慢)。

![[cuda-性能分析流程.svg]]

![[cuda-029roofline三种bound.svg]]

## 命令与判断
```bash
# ① Nsight Systems:先看全局时间线,找热点 kernel 与空泡
nsys profile -o report --trace=cuda,nvtx,osrt python infer.py
nsys stats report.nsys-rep            # 按 kernel 耗时排序,看占比

# ② Nsight Compute:钻进热点 kernel,看 roofline / 占用率 / stall
ncu --set full -k "regex:my_fused_kernel" -c 10 -o ncu_rep python infer.py
ncu --section SpeedOfLight --section Occupancy ./app   # 快速看 compute% vs memory%
```

```text
判定口诀(ncu 的 Speed Of Light 区):
  Compute(SM)% 高、Memory% 低  → compute-bound  → 上 FP8 / 改 tiling
  Memory% 高、Compute% 低       → memory-bound   → 融合 / 量化 / 减 HBM 往返
  两者都低、Occupancy 低         → latency-bound  → 增大 batch / CUDA Graph / 合并 kernel

❌ 没 profile 就动手写 Triton/CUDA:优化了非瓶颈,墙钟纹丝不动甚至变慢
✅ 先 nsys 定位热点 → ncu 看 roofline 判 bound → 只对真瓶颈写自定义/融合 kernel
```

## 面试高频
- **优化 GPU 第一步做什么?** Profile,不是写代码。先 nsys 看时间线找热点,再 ncu 看单 kernel 的 roofline 判 bound,对症优化。
- **Nsight Systems 和 Compute 区别?** Systems 是全局时间线(kernel 占比、空泡、CPU-GPU 同步),用于定位;Compute 是单 kernel 深剖(roofline、占用率、stall),用于深挖。先 Systems 后 Compute。
- **怎么判断 memory/compute/latency bound?** 看算术强度相对 roofline 脊点,或 ncu 的 Speed Of Light:Memory% 高→memory-bound,Compute% 高→compute-bound,两者都低+占用率低→latency-bound。
- **memory-bound 怎么优化?** 融合算子省 HBM 往返、量化减搬运字节、提数据复用;加算力没用。
- **什么时候才值得手写 kernel?** 当 profile 确认热点占比大、且现有库内核没打满 roofline、且融合/换算法能显著提升时。否则优先用 PyTorch/Triton/厂商库。
- **decode 为什么常 memory-bound?** 每步只算 1 个 query 但要读全量 KV,算术强度极低,瓶颈在带宽——这也是 [[026 FlashMLA：DeepSeek 的 MLA 推理内核|FlashMLA]] 压缩 KV 的动机。

## 关键事实
- 工具:**Nsight Systems(nsys)** 全局时间线定位;**Nsight Compute(ncu)** kernel 级 roofline/占用率/stall 深剖。NVIDIA 出品,Nsight 系列。
- **Roofline**:$\text{Perf}=\min(P_{\text{peak}},\,B_{\text{peak}}\cdot I)$,脊点 $I^\*=P_{\text{peak}}/B_{\text{peak}}$ 分界 memory/compute bound。
- 三类 bound 对策:memory→融合/量化;compute→低精度/tiling;latency→提 occupancy/CUDA Graph/合并 kernel。
- 方法论:**先 profile 后优化**,先 Systems 后 Compute,只对真瓶颈写自定义/融合 kernel。
- 关联:写内核用 [[028 Triton：用 Python 写 GPU kernel|Triton]] 或 CUDA;优化目标见 [[025 FlashAttention-3：Hopper TMA、WGMMA 与 FP8|FA3]]、[[027 量化内核：W4A16、W8A8 GEMM|量化内核]];硬件上限取决于 [[002 GPU 架构：SM、CUDA Core 与 Tensor Core|GPU 架构]]。
