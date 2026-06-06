这是 [[LLM Infra]] 域讲**集群通信**的起点:[[058 TP 的通信：每层 all-reduce 与 NVLink 依赖|TP 通信]]、[[060 数据并行与副本扩展|数据并行]]、[[061 专家并行 EP：大规模 MoE 服务|专家并行]] 底下统统调它,并参见 [[LLM/074 通信原语与计算通信重叠|通信原语]] 与 [[LLM/069 数据并行与 AllReduce|AllReduce]]。一句话:**NCCL 是 GPU 之间通信的「标准库」——上层只发逻辑原语(all-reduce 等),NCCL 负责拓扑感知地把它翻译成「谁先发给谁、走 NVLink 还是 IB、用 ring 还是 tree」;算得再快,通信选错路也会卡死,所以拓扑感知是它的灵魂。**

类比:NCCL 像快递公司的调度系统。你(PyTorch/Megatron/vLLM)只说"把这批货发给所有分仓并合账"(all-reduce),不关心走高速还是国道;调度系统看路况(NVLink/PCIe/IB 带宽),自动拼车、选路、决定先到先合还是分段合。你换机房(换拓扑),代码一行不改,它重新探测重新选路。

5 个核心原语,记住"谁有什么 → 谁要什么":**all-reduce**(各卡求和,人手一份全和;DP 梯度同步、TP 层内合并)、**all-gather**(各持一片→拼全再分发;FSDP 收权重)、**reduce-scatter**(求和后各取一片;FSDP 散梯度)、**broadcast**(一卡复制给所有卡)、**all-to-all**(每卡给每卡各送一片,等于矩阵转置;MoE/EP 的 dispatch 与 combine,大规模下的头号瓶颈)。关键恒等式:**all-reduce = reduce-scatter + all-gather**,这就是为什么 ring all-reduce 的通信量是 reduce-scatter 的两倍。

小数字:8 卡做 all-reduce,每张卡持有 $M$=1 GiB 梯度。朴素做法(每卡把自己的 1 GiB 发给其余 7 张)要发 $8\times7=56$ 份;ring all-reduce 只让每卡收发约 $2(N-1)/N\cdot M = 2\times7/8\times1=1.75$ GiB,与卡数几乎无关——这就是"带宽最优"。

$$
\text{ring all-reduce 每卡收发} = \frac{2(N-1)}{N}\,M
\;\xrightarrow{N\to\infty}\; 2M,\qquad
\text{all-reduce} = \underbrace{\text{reduce-scatter}}_{\frac{N-1}{N}M}+\underbrace{\text{all-gather}}_{\frac{N-1}{N}M}
$$

($N$=参与卡数,$M$=每卡数据量。两段各搬 $\tfrac{N-1}{N}M$,合起来 $\tfrac{2(N-1)}{N}M$。)

![[net-NCCL原语总览.svg]]

![[net-066五原语映射.svg]]

![[net-066环allreduce两阶段.svg]]

```bash
# 拿 nccl-tests 量真实带宽(busbw 才是有效带宽,别看 algbw)
# 8 卡机内 all-reduce,从 8B 扫到 8GB
./build/all_reduce_perf -b 8 -e 8G -f 2 -g 8
#  Size(B)    busbw(GB/s)   ← NVLink 上应接近 ~480 GB/s 的有效双向带宽

# 看它选了什么算法/协议(调试用)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH ./all_reduce_perf -b 1G -e 1G -g 8
```

```text
❌ 以为 all-reduce 是 NCCL 内置的"一种"固定实现
   → 它其实是 reduce-scatter+all-gather,且 ring/tree/NVLS 由 NCCL 按规模和消息大小动态选
❌ 看 algbw(算法带宽)评估性能
   → busbw(总线带宽)才反映物理链路利用率;ring all-reduce 的 busbw ≈ 2×algbw
✅ 调 ncclAllReduce 这一个逻辑接口,把选路交给 NCCL;用 nccl-tests 的 busbw 验真实带宽
```

## 面试高频
- **NCCL 5 个原语各干什么、谁用?** all-reduce(DP 梯度/TP 合并)、all-gather(FSDP 收权重)、reduce-scatter(FSDP 散梯度/ZeRO)、broadcast(初始化)、all-to-all(MoE/EP)。
- **all-reduce 和 reduce-scatter、all-gather 的关系?** all-reduce = reduce-scatter + all-gather,通信量正好是后两者之和,所以是 reduce-scatter 的 2 倍。
- **NCCL 的核心价值是什么?** 拓扑感知:启动时探测 NVLink/PCIe/IB,自动选算法(ring/tree/NVLS)、协议(Simple/LL/LL128)、传输层,上层零感知。
- **algbw 和 busbw 区别?** algbw=数据量/时间;busbw=按算法把链路实际搬的字节数算,跨算法可比,评估硬件利用率要看 busbw。
- **NCCL 走哪些传输?** 机内 NVLink/NVSwitch,退路 PCIe;跨节点 IB Verbs/RoCE/TCP,配 GPUDirect RDMA 让网卡直读显存绕开 CPU。

## 关键事实
- NCCL 实现 all-reduce、all-gather、reduce、broadcast、reduce-scatter 及任意 send/recv(point-to-point),优化覆盖 PCIe、NVLink、NVSwitch、IB Verbs、TCP/IP(NVIDIA NCCL 官方文档,**2025**)。
- ring all-reduce 带宽最优(每卡收发 $\tfrac{2(N-1)}{N}M$,与 $N$ 几乎无关),但延迟随 $N$ 线性增长——所以小消息/大规模改用 tree(见下一篇)。
- 竞品生态(**2025**):AMD RCCL、Intel oneCCL、微软 MSCCL(可编程集合)、Meta NCCLX(支撑 10 万+ GPU)。
- NCCL 2.27(**2025**)新增对称内存(symmetric memory),小消息延迟最高降 9×;并对 NVLink/IB SHARP 扩展到 all-gather/reduce-scatter。
