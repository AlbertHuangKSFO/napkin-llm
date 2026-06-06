[[116 长上下文服务：streaming、attention sink 与 KV 预算]] 关心**怎么把一条可能无限长的会话稳定地服务下去**:朴素做法下 KV-Cache 随序列长度 $T$ 线性增长,迟早 OOM、prefill 越来越慢、并发越压越低。**StreamingLLM** 给出一个免微调的处方——**永久保留最前几个「[[035 KV 驱逐与压缩：H2O 与注意力汇|注意力汇(attention sink)]]」+ 滑动近窗**,让驻留 GPU 的 KV **恒定**,从而支持数百万 token 的流式生成。这是一种**有损但极省**的 KV 预算管理,与真正扩窗的 [[LLM/107 长上下文推理：YaRN、位置插值与 StreamingLLM|长上下文推理]](YaRN/位置插值)互补:后者精确但 KV 仍随长度涨,前者定长但远端精确召回丢失。它直接关系到 [[015 KV-Cache 的显存账(逐层手算)|KV 显存账]] 里的 $T$ 维,并和 [[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|分层 offload]] 形成「有损压缩 vs 无损搬走」的两条路。

## 直觉类比
长会话的 KV 像写连载小说时桌上摊开的**所有草稿纸**,越堆越高,桌子(显存)迟早放不下。Streaming 服务的处方是**定期清桌但保两样**:① 最前面那几张「提纲/封面」必须留——后文每句都要瞄一眼它(这就是注意力汇,softmax 的「泄洪口」);② 手边最近几张留着(近窗)。中间没人看的整体丢。于是桌面**始终差不多大**,无论写到第几百万字。代价:中间那些被扔掉的剧情,你再也精确记不起来了。

## 小数字例子
设上下文已生成到 2,000,000 token,每 token KV 约 0.5 MB/层级折算。
- **全量保留**:KV ∝ 2,000,000 → 早就 OOM,prefill/attention 也随长度爆。
- **StreamingLLM**:留前 **4** 个 sink + 滑窗 **2044** = 恒定 **2048** 个 KV,无论生成到第几百万 token,显存**不变** → 稳定流式,论文支持 **400 万 token**,相对「滑窗重算」基线最高 **22.2×** 加速。
- 关键观察:朴素滑窗若把最前 4 个 sink 也淘汰,**困惑度爆炸**——丢掉了 softmax 的泄洪口。

## 原理:注意力汇与 KV 预算
训练好的 decoder-only 模型(Llama、Mistral、Falcon、Pythia 皆然)深层里,后面每个 query 都把**大量注意力质量倒给最前 1~4 个 token**,哪怕它们语义为空。原因:softmax 必须把权重归一到和为 1,当 query 对当前所有 token 都「不太想看」时,多余质量需要一个**默认倾倒处**,前几位天然充当:

$$
\sum_{j=1}^{t}\mathrm{softmax}(s_{tj})=1,\qquad s_{tj}=\frac{q_t k_j^\top}{\sqrt{d}}
$$

一旦把承接这部分质量的 sink 的 KV 驱逐,剩余权重被迫重新归一化 → 分布剧变 → 掉点。所以处方是**永久保留前 $K$ 个 sink(常 $K=4$)+ 滑动近窗 $W$**,KV 预算恒定:

$$
|\text{KV}| \approx K + W \quad(\text{与会话总长 } T \text{ 无关})
$$

这让有限窗训练的模型**免微调**泛化到极长序列;但中间历史被丢,长程精确检索(大海捞针)可能失败。

![[srv-长上下文流式服务.svg]]

把全量、朴素滑窗、StreamingLLM 三种 KV 预算策略并排看(含 sink 为何不能丢):

![[srv-116KV预算三方案.svg]]

## 配置 / 代码
```python
# vLLM:对长会话用 sliding window / KV 驱逐控制 KV 预算
from vllm import LLM, SamplingParams

# 1) 模型自带 sliding window(如 Mistral)→ 引擎只保近窗 KV
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.3")  # 滑窗在模型配置里

# 2) StreamingLLM 思路:KV = 固定 sink + 滑窗(伪代码)
class StreamingKVCache:
    def __init__(self, n_sink=4, window=2044):
        self.n_sink, self.window = n_sink, window
        self.keys, self.vals = [], []

    def append(self, k, v):
        self.keys.append(k); self.vals.append(v)
        if len(self.keys) > self.n_sink + self.window:
            # 只留 [前 n_sink 个 sink] + [最近 window 个],中间整体丢
            self.keys = self.keys[:self.n_sink] + self.keys[-self.window:]
            self.vals = self.vals[:self.n_sink] + self.vals[-self.window:]
        return self.keys, self.vals

sp = SamplingParams(max_tokens=1_000_000)  # 超长流式生成
```

```text
❌ 无限会话仍保全量 KV → KV ∝ T 线性涨 → OOM、prefill 变慢、并发塌方;或朴素滑窗连 sink 一起淘汰 → 困惑度爆炸
✅ 留前 4 个 attention sink + 近窗 → KV 定长、显存稳、时延可预测、免微调,代价是远端精确召回丢失(按需配 offload 补)
```

## 面试高频
- **长会话为什么会 OOM?** KV-Cache ∝ 序列长 $T$ 线性增长;不裁剪迟早撑爆显存,且 attention/prefill 也随长度变慢、可服务并发骤降。
- **attention sink 是什么,为什么不能丢?** 后面 token 把大量注意力质量倒给最前几位(即使语义空),因 softmax 要归一、需泄洪口;丢掉它们权重重归一化剧变 → 困惑度爆。
- **StreamingLLM 怎么把 KV 钉成定长?** 永久留前 $K$ 个 sink + 滑动近窗 $W$,KV 预算 $\approx K+W$ 与会话总长无关,支持百万级流式,免微调。
- **它和 YaRN/位置插值什么区别?** YaRN 真正**扩大有效窗**、精确但 KV 仍随长度涨;StreamingLLM **定长**、省显存但丢中间历史的精确召回,二者互补。
- **代价是什么?何时不能用?** 被丢 token 永久消失,长程精确检索(needle-in-haystack)会掉点;需精确长程记忆时改用真扩窗 + 全量 KV 或分层 offload。
- **为什么免微调?** 纯推理期 KV 管理,不改权重;让有限窗训练的模型直接泛化到极长序列。

## 关键事实
- **StreamingLLM**,Xiao et al.,arXiv **2309.17453**(2023,ICLR 2024,mit-han-lab):发现 attention sink 现象,保留前 4 个 sink + 滑窗,支持 **400 万 token** 稳定流式,相对滑窗重算最高 **22.2×**,免微调。
- 注意力汇在 Llama / Mistral / Falcon / Pythia 普遍存在;在预训练加一个专用 sink 占位符可进一步改善流式部署。
- 与 H2O(按累计注意力分留重击者)同属 KV 驱逐家族,见 [[035 KV 驱逐与压缩：H2O 与注意力汇|注意力汇]];StreamingLLM 主打无限长流式,H2O 主打一般生成。
- vLLM / SGLang(2025)支持 sliding-window 与 KV 驱逐策略;长上下文服务常配「近窗精确 + 远端驱逐/offload」分段。
- 真扩窗路线见 [[LLM/107 长上下文推理：YaRN、位置插值与 StreamingLLM|长上下文推理]];定长省显存与精确召回是一对取舍。
