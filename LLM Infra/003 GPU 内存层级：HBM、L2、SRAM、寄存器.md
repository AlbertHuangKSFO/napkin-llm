[[003 GPU 内存层级：HBM、L2、SRAM、寄存器|GPU 内存层级]] 是寄存器 → SRAM(共享内存/L1)→ L2 → HBM 的金字塔:越往上越快越小,**HBM 带宽决定 [[001 LLM 推理的系统视角：从一次请求到一张卡|decode]] 速度**,而把数据留在 SRAM 是 FlashAttention 提速的根本原因。

类比仓储:**寄存器是工人手里的工具**(瞬取、极少);**SRAM 是车间工作台**(一伸手就够到,放当前在加工的料);**L2 是车间共用的货架**;**HBM 是厂区大仓库**(容量大但每次跑一趟有可观延迟和有限吞吐)。LLM 权重和 [[LLM/102 KV-Cache|KV-Cache]] 都堆在大仓库(HBM),decode 每出一个 token 就得把整仓库的权重搬一遍——所以**仓库到车间的运输带宽**(HBM 带宽)是瓶颈。

H100 的数量级速查(SXM5):
- **HBM3**:80 GB,带宽 3.35 TB/s,延迟数百 ns。
- **L2 Cache**:50 MB 片上 SRAM,全 SM 共享,带宽高于 HBM。
- **L1/Shared Memory**:每 SM ~228 KB,聚合带宽达数十 TB/s,延迟近 1 周期级。
- **寄存器堆**:每 SM 256 KB。
SRAM 比 HBM 快约一个数量级,但容量小约 6 个数量级——这个鸿沟是所有 GPU kernel 优化的起点。

![[mem-003内存层级延迟容量对比.png]]

原理:decode 单 token 的时间下限由"搬运全部权重"决定,与算力无关:
$$t_{\text{token}} \ge \frac{\text{权重字节数}}{\text{HBM 带宽}} = \frac{2P}{\text{BW}_{\text{HBM}}}$$
70B、FP16:$t \ge \frac{2 \times 70\text{e}9}{3.35\text{e}12} \approx 42$ ms,即 [[004 算力 vs 带宽：Roofline 与算术强度|memory-bound]]。FlashAttention 的洞见是:朴素 attention 把 $N\times N$ 的注意力矩阵写回 HBM 再读回,产生 $O(N^2)$ 的 HBM 流量;改成**在 SRAM 里分块(tiling)累加、永不落地完整矩阵**,HBM 流量降到 $O(N^2 \cdot d / M)$($M$ 为 SRAM 容量),从而把 attention 从 memory-bound 拉回来。这就是"留在 SRAM"的威力。相关[[LLM/076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]]决定 HBM 够不够装,本篇关注的是**搬得多快**。

![[mem-003FlashAttention留SRAM.png]]

![[mem-GPU内存层级金字塔.png]]

用脚本验证 decode 是 HBM 带宽下限主导,以及"读全权重"的直觉:

```python
# ✅ 估单卡 decode 的带宽下限(忽略 KV,只算权重)
def decode_floor(P_billion, bytes_per_param, bw_TBs):
    weight_bytes = P_billion*1e9 * bytes_per_param
    t = weight_bytes / (bw_TBs*1e12)
    return t*1e3, 1/t          # ms/token, token/s

for name, P, b, bw in [("70B FP16 H100", 70, 2, 3.35),
                       ("70B FP8  H100", 70, 1, 3.35),
                       ("70B FP16 H200", 70, 2, 4.8)]:
    ms, tps = decode_floor(P, b, bw)
    print(f"{name}: {ms:.1f} ms/token, 上限 {tps:.0f} tok/s")
# 70B FP16 H100: 41.8 ms, 24 tok/s
# 70B FP8  H100: 20.9 ms, 48 tok/s  ← 位宽减半,带宽流量减半,速度翻倍
# 70B FP16 H200: 29.2 ms, 34 tok/s  ← 换更快 HBM3e,直接更快
```

```bash
# ❌ 只看显存占用("装得下就行"),忽略带宽 → 解释不了为何 decode 慢
# ✅ 同时看带宽:HBM 带宽是 decode 的硬上限
nvidia-smi --query-gpu=memory.total,memory.used --format=csv   # 看容量
# 带宽不在 nvidia-smi 里,需查 datasheet 或跑 bandwidthTest(CUDA samples)
```

输出印证两件事:**位宽减半 → 速度翻倍**(FP16→FP8),**换更快 HBM → 速度直接涨**(H100→H200,这正是 H200 的卖点)。

## 面试高频
- **Q:GPU 内存层级从快到慢有哪几级?** A:寄存器 → L1/Shared Memory(SRAM)→ L2 Cache → HBM(全局显存)。越往上越快、容量越小、越靠近计算单元。陷阱:漏掉片上的 SRAM/L2,只说"显存"——FlashAttention 等优化正是利用了 SRAM 这一层。
- **Q:为什么说 decode 是 memory-bound,瓶颈在 HBM 带宽?** A:decode 每步只算 1 个 token(算力需求极小),却必须把全部模型权重从 HBM 读一遍,时间 ≈ 权重字节数 / HBM 带宽,与算力无关。陷阱:答"因为显存不够"——是带宽(搬多快)不是容量(装多少)。
- **Q:FlashAttention 为什么快?** A:它把 attention 计算在 SRAM 里分块累加,避免把 $N\times N$ 注意力矩阵写回再读 HBM,大幅降低 HBM 流量(从 $O(N^2)$ 降到与 SRAM 容量相关),从而把 memory-bound 的 attention 提速。陷阱:答"减少了计算量"——FLOPs 基本不变,减少的是 HBM 读写。

## 关键事实
- H100 SXM5:80GB HBM3、3.35 TB/s 带宽、50MB L2 Cache;每 SM 含 256KB 寄存器堆与最高 228KB 可配 L1/Shared Memory。来源:NVIDIA H100 Datasheet / Hopper Architecture In-Depth(NVIDIA, 2022)。
- A100 80GB:HBM2e、约 2.0 TB/s;H200:HBM3e 141GB、4.8 TB/s。来源:NVIDIA A100/H200 Datasheet(2020/2023)。
- FlashAttention 通过 SRAM 分块将 attention 的 HBM 访问降为线性级,显著加速且省显存。来源:FlashAttention(Dao et al., NeurIPS 2022, arXiv:2205.14135);FlashAttention-2(Dao, 2023, arXiv:2307.08691)。
