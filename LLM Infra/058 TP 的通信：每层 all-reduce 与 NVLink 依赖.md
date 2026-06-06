这是 [[LLM Infra]] 域对 [[057 张量并行推理：延迟换显存|张量并行推理]] 通信代价的拆解,底层依赖 [[008 片内互联：NVLink、NVSwitch、NVLink-C2C|NVLink]] 与 [[066 NCCL：集合通信库与原语|NCCL]],并参见 [[LLM/074 通信原语与计算通信重叠|通信原语]]。一句话:**TP 的每一个 Transformer 层都要做 2 次 all-reduce,通信量与隐藏维成正比、且在关键路径上高频出现,所以 TP 必须跑在 NVLink 上、必须限在一台机内——跨 PCIe 直接崩。**

类比:TP 像两个会计合算同一本账,每过一道工序就要碰头对账(all-reduce)。如果两人坐在同一间办公室(NVLink,~900 GB/s),对账几秒搞定、被本职计算掩盖;如果一个在北京一个在上海(跨 PCIe/网络),每次对账都要打长途、等回执——对账时间反而超过算账时间,整体瘫掉。

**生活类比**:4 个会计合算一道大题,每人只算一截,但每过一道工序就得碰头把数凑齐对一遍(这一"凑"就是 all-reduce)。坐**同一桌**(NVLink ~900 GB/s),伸手一碰几秒搞定,80 层 × 2 次 = 160 次对账都不痛,2 卡照样拿 ≈1.84× 吞吐;改成一个在北京一个在上海、隔着走廊喊(PCIe ~64 GB/s,慢一个数量级),每道工序都打长途等回执,对账比算账还久——通信吃掉 40–50% 时间、缩放效率塌到 0.7,TP 直接废。技术对应:坐同桌=机内 NVLink,隔走廊=跨 PCIe/跨节点;"凑数对一遍"=每层 2 次 all-reduce、通信量 ∝ b·s·h。

![[ipar-058类比同桌凑答案.svg]]

为什么是 2 次?Megatron 的切法里,一层有两个并行块:**自注意力**算完要 all-reduce 一次合 head 输出,**MLP**(列并行→行并行)算完要 all-reduce 一次合行并行的部分和。所以 $L$ 层模型一次前向 = $2L$ 次 all-reduce。每次的数据量 $\propto b\cdot s\cdot h$(batch × seq × hidden),隐藏维越大、通信越重。

小数字:Llama-70B 有 80 层 → 一次前向 **160 次 all-reduce**。在 H100 SXM(NVLink 第4代 ~900 GB/s)上这些通信被算力掩盖,2 卡 TP 仍能拿到 ≈1.84× 吞吐(≈0.92 缩放效率)。换到无 NVLink 的 L40S(走 PCIe5 ~64 GB/s,低一个数量级):**TP=4 时通信吃掉 40–50% 推理时间**,缩放效率塌到 0.70–0.78。这就是为什么这类机器宁可用 PP——PP 每个 stage 边界只点对点传一次激活,通信稀疏得多。

$$
\text{通信次数}_{\text{fwd}} = 2L,\qquad
V_{\text{allreduce}} \propto b\cdot s\cdot h,\qquad
t_{\text{comm}}\approx \frac{2(N-1)}{N}\cdot\frac{V}{\text{BW}}
$$

($N$=TP 度,ring all-reduce 每卡收发约 $\tfrac{2(N-1)}{N}V$ 字节;BW 是互联带宽——NVLink 还是 PCIe,决定生死。)

![[ipar-TP通信NVLink依赖.svg]]

```python
# 部署前先确认拓扑:有没有 NVLink 全互联
# nvidia-smi topo -m   →  期望看到 NV# / NVLink,而非 PIX/PHB(PCIe)
from vllm import LLM
llm = LLM("meta-llama/Llama-3.1-70B-Instruct",
          tensor_parallel_size=8)   # 仅当 8 卡在同一节点且 NVSwitch 全互联
```

```text
❌ 跨节点 / 无 NVLink 上 tensor_parallel_size=8
   → 每层 all-reduce 走 PCIe 或 IB,延迟暴涨,推理不可用
✅ TP 限在机内 NVLink;要跨节点放更大模型,改用 PP(点对点、通信稀疏)
   nvidia-smi topo -m 看到 NV# 才敢上 TP;看到 PIX/PHB 就别上
```


![[ipar-058TP通信暴露对比.svg]]

## 面试高频
- **TP 每层几次通信、什么原语?** 2 次 all-reduce(注意力 1 次 + MLP 1 次);$L$ 层 = $2L$ 次。
- **TP 为什么不能跨节点?** all-reduce 高频且在关键路径,跨节点链路(IB/PCIe)带宽与延迟撑不住,通信占比飙到一半以上。
- **通信量随什么变?** 与 $b\cdot s\cdot h$ 成正比,隐藏维 / batch / 序列越大越重;不随权重大小变。
- **为什么 NVLink 上 all-reduce 几乎"免费"?** 带宽高(~900 GB/s)且可与计算重叠,通信被算力掩盖,因此缩放效率接近 1。
- **没 NVLink 的机器怎么办?** 放弃 TP,改 PP(每 stage 边界只点对点传一次激活),或换 SXM 机型。

## 关键事实
- 每层 2 次 all-reduce 来自 Megatron 切法(Shoeybi et al., **2019**)。
- H100 NVLink 第4代 ~900 GB/s vs PCIe5 ×16 ~64 GB/s,相差一个数量级。
- 实测:PCIe 上 TP=4 通信占 40–50% 推理时间,缩放效率 0.70–0.78;NVLink 上 ≈0.92。
- 工程定律(**2025**):**TP 在节点内,PP 跨节点**——慢链路只承担稀疏的点对点激活传输。
