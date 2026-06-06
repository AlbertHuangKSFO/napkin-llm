[[025 FlashAttention-3：Hopper TMA、WGMMA 与 FP8]] 是把 [[LLM/025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 重写到 **Hopper(H100)专属硬件**上的第三代内核:用 **TMA 异步搬运 + WGMMA warpgroup 矩阵乘 + FP8 低精度**,让 softmax 与矩阵乘**深度重叠**,把 [[024 FlashAttention 1、2：IO 感知精确注意力|FA1、2]] 在 H100 上仅 ~35% 的 Tensor Core 利用率拉到 85%。它解决的不是"内存墙",而是"**Tensor Core 太快、softmax 跟不上**"这个新瓶颈。

## 直觉类比
FA2 像一条流水线:工人先把零件搬到工位(载数据),再用快得离谱的机床加工(Tensor Core 做 QKᵀ 和 PV),中间还要手工打磨(softmax 的 exp)。问题是机床快到打磨时它只能干等。FA3 的做法是**配三班倒**:搬运班(TMA)、机床班(WGMMA)、打磨班(softmax)同时开工,机床加工块 2 时,搬运班已在搬块 3,打磨班在磨块 1——谁都不闲着。这就是 **warp 专门化(warp specialization)+ 异步(asynchrony)**。

## 小数字例子
H100 的 Tensor Core:FP16 算力 ~989 TFLOPS,FP8 翻倍到 ~1979 TFLOPS。但 softmax 的 `exp` 走的是 SFU/CUDA Core,吞吐低两个数量级。假设一个 attention 块,GEMM 耗 100 个时间单位,softmax 耗 50 个:
- **串行(朴素)**:100 + 50 = 150,Tensor Core 有 50 个单位在闲置(利用率 67%)。
- **FA3 重叠**:用乒乓(ping-pong)把块 A 的 softmax 塞进块 B 的 GEMM 阴影,墙钟约 max(100,…) ≈ 100,利用率逼近 85%。
论文实测(arXiv 2407.08608):BF16 达 ~840 TFLOPS,FP8 达 ~1.3 PFLOPS,相对 FA2 提速 **1.5–2.0×**。

## 原理:为什么需要异步与重叠
FlashAttention 的核心仍是在线 softmax 分块累加,对每个 KV 块更新 running max $m$ 与 running sum $\ell$:

$$
m_i = \max(m_{i-1}, \text{rowmax}(S_i)),\quad
P_i = \exp(S_i - m_i),\quad
\ell_i = e^{m_{i-1}-m_i}\,\ell_{i-1} + \text{rowsum}(P_i)
$$

$$
O_i = e^{m_{i-1}-m_i}\,O_{i-1} + P_i V_i,\qquad S_i = Q K_i^\top
$$

FA3 不改这个数学,改的是**怎么在硬件上排这些算子**。三个 Hopper 特性各司其职:

1. **TMA(Tensor Memory Accelerator)**:一个专用 DMA 单元,单线程发一条指令就异步把整块 $K,V$ 从 HBM 搬进 SMEM,搬运期间 warp 不阻塞 → 数据搬运与计算解耦。
2. **WGMMA(Warpgroup MMA)**:整个 warpgroup(128 线程)协同的异步 Tensor Core 指令,直接吃 SMEM 操作数,吞吐远高于 Ampere 的 `mma.sync`;发射后立即返回,可与 softmax 重叠。
3. **FP8 + 块量化(block quantization)+ 非相干处理(incoherent processing)**:用 FP8 把 GEMM 吞吐翻倍,再用块级缩放和随机正交变换压低离群值带来的量化误差(较朴素 FP8 误差 ↓2.6×)。

重叠靠两招:**warp 专门化**(部分 warp 当生产者只发 TMA,部分当消费者做 MMA/softmax,经 SMEM + barrier 通信)和 **块内交错 / 乒乓调度**(两个 warpgroup 交替,A 做 GEMM 时 B 做 softmax)。

![[flash-FA3异步流水线.svg]]

![[flash-025利用率对比手算.svg]]

## 内核伪码(warp 专门化骨架)
```python
# FA3 概念伪码(CUTLASS/CuTe 风格,非真实 API)
if warpgroup_is_producer():
    for j in range(num_kv_blocks):
        tma_load_async(K_smem[j%STAGES], K_gmem[j])   # TMA 异步搬运,不阻塞
        tma_load_async(V_smem[j%STAGES], V_gmem[j])
        producer_barrier_arrive(j)                     # 通知消费者:块 j 就绪
else:  # 消费者 warpgroup
    m, l, acc = -inf, 0, 0
    for j in range(num_kv_blocks):
        consumer_barrier_wait(j)                        # 等 TMA 完成
        S = wgmma(Q_smem, K_smem[j%STAGES])             # GEMM0:异步 Tensor Core
        # —— 关键:下一块的 wgmma 已发射,本块在 CUDA Core 上算 softmax,二者重叠 ——
        m_new = max(m, rowmax(S)); P = exp(S - m_new)
        l = exp(m-m_new)*l + rowsum(P); acc = exp(m-m_new)*acc
        acc += wgmma(P, V_smem[j%STAGES])               # GEMM1:与下一轮 softmax 乒乓
        m = m_new
    O = acc / l
```

```bash
# ❌ 在 H100 上仍调 FA2:Tensor Core 利用率 ~35%,白白浪费 Hopper 硬件
# ✅ 用 FA3(需 CUDA 12.3+、Hopper SM90):pip install flash-attn>=3 / 走 vLLM、PyTorch SDPA 的 FA3 后端
python -c "import flash_attn_interface"   # FA3 提供独立的 hopper 接口
```

## 面试高频
- **FA3 相对 FA2 的本质改进是什么?** 不是改算法(还是在线 softmax 分块),而是吃满 Hopper 的**异步硬件**:TMA 搬运、WGMMA、FP8;把 softmax 藏进 GEMM 阴影,解决 Tensor Core 利用率低的新瓶颈。
- **为什么 FA2 在 H100 上利用率反而低?** Hopper 的 Tensor Core 太快,softmax(走 SFU 的 exp)成了串行瓶颈,GEMM 与 softmax 不重叠时 Tensor Core 大量闲置。
- **warp 专门化是什么?** 把 warpgroup 分成生产者(只发 TMA)和消费者(做 MMA+softmax),通过 SMEM 和异步 barrier 解耦搬运与计算。
- **FP8 attention 精度怎么保?** 块量化(每块独立缩放,抗离群)+ 非相干处理(Hadamard 类正交变换打散离群值),误差较朴素 FP8 降 2.6×。
- **FA3 只能 H100 跑吗?** 是,依赖 TMA/WGMMA(SM90)。Ampere/Ada 仍用 FA2。Blackwell 需再适配(FA4 方向)。

## 关键事实
- **FlashAttention-3**,Shah、Bikshandi、Zhang、Thakkar、Ramani、Dao,**NeurIPS 2024**,arXiv **2407.08608**。
- H100 实测:**BF16 ~840 TFLOPS(85% 利用率)、FP8 ~1.3 PFLOPS**,较 FA2 提速 **1.5–2.0×**;FP8 数值误差较基线 ↓**2.6×**(2024)。
- 三大 Hopper 特性:**TMA**(异步搬运)、**WGMMA**(warpgroup 异步 MMA)、**FP8**(吞吐翻倍,989→1979 TFLOPS)。
- 由 Colfax Research 用 **CUTLASS/CuTe** 实现;已集成进 PyTorch SDPA、vLLM。
- 相关:[[026 FlashMLA：DeepSeek 的 MLA 推理内核|FlashMLA]] 即受 FA2/FA3 启发;[[LLM/098 FP8 训练推理与 AQLM 极低比特|FP8]]。
