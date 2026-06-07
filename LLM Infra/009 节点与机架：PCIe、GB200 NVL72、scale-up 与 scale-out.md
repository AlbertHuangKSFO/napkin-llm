[[009 节点与机架：PCIe、GB200 NVL72、scale-up 与 scale-out|节点与机架]] 讲 GPU 集群的物理层级:机内靠 PCIe / NVLink 连卡(scale-up,纵向扩),机间靠 InfiniBand / 以太连节点(scale-out,横向扩);GB200 NVL72 的杀招是把一整个机架 72 卡拉进同一个 NVLink 域,模糊了机内与机间的界线。

## 直觉:车间、厂房、厂区
- **scale-up(纵向)**= 在一个车间里多塞机床,靠内部传送带([[008 片内互联：NVLink、NVSwitch、NVLink-C2C|NVLink]])直连,快但有物理上限(一台服务器塞不下无限张卡)。
- **scale-out(横向)**= 再盖一栋厂房,厂房之间靠卡车(InfiniBand / 以太)运货,能无限扩但慢。
- **PCIe** = 车间里通往大门和仓库(CPU、网卡)的内部公路,限速。
- **GB200 NVL72** = 把一整个厂区围墙打掉,72 台机床共用一条超级传送带,厂区内部全是「快域」。

## 小数字例子:跨节点 vs 机内,慢多少
all-reduce 同一份 1GB:
- 机内 NVLink 5(1.8 TB/s):≈ **0.56 ms**。
- 跨节点 InfiniBand NDR(400 Gbps ≈ 50 GB/s):≈ **20 ms** —— 慢约 **36 倍**。
所以把通信最频繁的张量并行([[008 片内互联：NVLink、NVSwitch、NVLink-C2C|TP]])放机内,把容忍慢的流水线/数据并行放跨节点。GB200 NVL72 把 72 卡塞进同一 NVLink 域后,原本要跨 9 个 8 卡节点(慢)的通信,变成一个机架内的快通信。

![[hw-009机内跨节点36倍.png]]

## 原理:并行策略 ↔ 互联层级映射
设 TP=$t$、PP=$p$、DP=$d$,总卡数 $N = t \cdot p \cdot d$。最优摆放法则:

$$\underbrace{t}_{\text{NVLink 域内}} \;\times\; \underbrace{p \cdot d}_{\text{跨节点网络}} = N, \quad t \le (\text{NVLink 域大小})$$

NVLink 域大小:普通 8 卡节点 = 8;GB200 NVL72 = 72。域越大,$t$ 能越大 → 单层激活切得越细、单卡显存压力越小(见 [[011 单卡能放多大模型：参数与 KV 显存预算|显存预算]]),且无需切 PP 引入气泡。这是 NVL72 对超大模型/MoE 推理的根本优势。

## 图
![[hw-互联层级金字塔.png]]

![[hw-009并行映射互联层级.png]]

## 层级与带宽(已验证,2025–2026)

| 层级 | 互联 | 带宽 | 放什么并行 |
|---|---|---|---|
| 机内 GPU↔GPU | NVLink 4 / 5 | 900 GB/s ~ 1.8 TB/s | TP(频繁 all-reduce) |
| 机内 GPU↔CPU/NIC | PCIe Gen5 | ~128 GB/s | 喂数据、KV offload、出网 |
| 单节点 | — | 典型 8× GPU + 2 CPU | 一个 NVLink 域,TP 上限 8 |
| 跨节点 scale-out | InfiniBand / RoCE | ~400–800 Gbps/卡 (50–100 GB/s) | PP / DP / EP(通信稀疏) |
| 机架级(GB200 NVL72) | NVLink Switch | 130 TB/s 总,72 卡同域 | TP 域扩到 72 卡 |

## 代码/配置:并行度按层级摆放

```python
# ❌ 普通 8 卡节点上设 TP=16 → 跨节点做 TP,all-reduce 走 IB,吞吐崩
parallel = dict(tensor_parallel=16, pipeline_parallel=1)   # 16 > NVLink 域 8

# ✅ TP 限在 NVLink 域(=8),跨节点用 PP/DP
parallel = dict(
    tensor_parallel=8,        # 单节点 NVLink 域内
    pipeline_parallel=4,      # 4 节点流水线,跨 IB,通信稀疏可容忍
)
# ✅ GB200 NVL72 上,NVLink 域 = 72 → 可直接 TP=8/16 甚至更大,
#    省掉 PP 引入的流水线气泡,超大 MoE 单机架内推理
```

```bash
# ✅ 确认两张卡是否真在同一 NVLink 域(NV# 才是)
nvidia-smi topo -m        # 看到 SYS/NODE = 跨 PCIe/跨节点,慢
```

## 面试高频
- **scale-up 和 scale-out 区别?** scale-up = 机内加卡靠 NVLink(快、有物理上限);scale-out = 跨节点加机器靠 IB/以太(可无限扩、慢一个数量级)。先 scale-up 到域满,再 scale-out。
- **为什么 TP 上限通常是 8?** 普通服务器一个 NVLink/NVSwitch 域 = 8 卡;TP 通信必须在域内,所以 TP ≤ 8。GB200 NVL72 把域扩到 72,TP 上限随之放大。
- **GB200 NVL72 的本质创新?** 把整机架 72 卡变成单一 NVLink 域(130 TB/s),原本 scale-out 的慢通信被拉回 scale-up 快域 → 万亿参数/MoE 推理在一个机架内当「一个大 GPU」用。
- **PCIe 在 LLM 推理里还重要吗?** 重要但不在 GPU 互联关键路径:它负责喂数据、KV-Cache offload 到 CPU 内存、出网卡。带宽 ~128 GB/s,比 NVLink 慢 ~7×,所以不能拿来做 TP。
- **怎么排查并行度配错?** `nvidia-smi topo -m` 看拓扑,确认 TP 涉及的卡两两是 NV# 直连;出现 SYS/PHB 说明走了 PCIe/跨节点,该重排并行度。

## 关键事实(2025–2026)
- **典型节点** = 8× GPU + 2× CPU,8 卡构成一个 NVLink/NVSwitch 全互联域;TP 上限 = 域大小。
- **PCIe Gen5** ≈ 128 GB/s,是机内通用总线,比 NVLink 4(900 GB/s)慢约 7 倍;NVLink-C2C(900 GB/s)约 PCIe Gen5 的 7 倍。
- **scale-out 网络**:InfiniBand NDR/XDR 或 RoCE 以太,~400–800 Gbps/卡,跨节点 all-reduce 比机内慢一两个数量级 → 只放 PP/DP/EP。
- **GB200 NVL72**:72× B200 + 36× Grace,单一 NVLink 域、130 TB/s 总互联、13.4 TB 统一显存、1.44 EFLOPS FP4;把机架变「一个 GPU」,TP 域 8→72。
