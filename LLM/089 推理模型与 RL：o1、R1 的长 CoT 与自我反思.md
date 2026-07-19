[[089 推理模型与 RL：o1、R1 的长 CoT 与自我反思|推理模型]]:推理模型是把「先想、再答、必要时回头检查」这种行为通过大规模 RL 与测试时算力训进策略的模型；但**隐藏 reasoning、可见摘要、API 暴露的 reasoning 字段、可审计轨迹**是四件不同的事，不能混成一句“看 `<think>` 就知道模型在想什么”。

## 直觉：别把 raw CoT、摘要和审计证据混为一谈

普通 LLM 常靠提示词临时触发 “let’s think step by step”；推理模型则把“多走几步、尝试替代路径、自我纠偏、最后验证”训练成默认策略。

但产品层到底暴露什么，各家并不一样：

- **隐藏 reasoning**：模型内部可能有长推理，但用户看不到 raw chain。
- **可见摘要**：产品只展示一个经过模型整理的 reasoning summary。
- **供应商字段**：某些 API 直接返回 `reasoning_content` 之类字段。
- **可审计轨迹**：工具调用、引用、测试输出、终态验证，这才是工程上可靠的证据。

OpenAI 在 2024-09-12 发布 o1 时就明确说明：**raw chain-of-thought 不向用户展示，而是展示模型生成的摘要**。DeepSeek 当前 API 文档则明确提供 `reasoning_content` 字段。两者都说明：**“可见的推理文本”是供应商接口设计，不是跨模型协议，更不是天然可信的审计载体**。

![[post-long-cot-reflect.png]]

## 小数字手算：为什么“多想一轮再验证”能提高通过率

还是看题目「一个数加上它的一半等于 18，求这个数」。

若模型第一反应直接猜：

$$
x=9 \Rightarrow 9+4.5=13.5\neq18.
$$

这条路径失败。

若模型愿意再做一轮代数整理：

$$
x+\frac{x}{2}=18
\Rightarrow
\frac{3x}{2}=18
\Rightarrow
3x=36
\Rightarrow
x=12.
$$

最后再做一次代回验证：

$$
12+\frac{12}{2}=12+6=18.
$$

这就是“推理模型为什么愿意花更多 token”的最小例子：**多一轮候选路径和终态验证，能把 pass 概率从 0 提到 1**。训练时 RL 只奖励“最后答对”，但策略会自己学会“多想一步再检查”。

## 公式推导：RL 优化的是答对率，不是 `<think>` 的外观

把解题看成序列决策：状态是当前问题与已生成前缀，动作是下一个 token，奖励是终态 verifier 对最终答案的判定。

最简写法是

$$
\max_{\pi_\theta}\ \mathbb{E}_{x,\ y\sim\pi_\theta}\big[r_{\text{verify}}(x,y)\big].
$$

如果奖励是二值的：

$$
r_{\text{verify}}(x,y)=
\begin{cases}
1,& \text{最终答案或终态通过验证}\\
0,& \text{否则}
\end{cases}
$$

那么 RL 会偏好一切能提高通过率的策略：列中间式、尝试备选路线、回头纠错、最后自检，都只是提高期望奖励的手段，而不是预先写死的格式。

这也是为什么 **“推理 token 变长”不等于“能力一定变强”**。如果增长来自有效搜索和自检，准确率会同步上升；如果增长主要来自算法偏差或模板诱导，只会让输出更长、更慢，未必更对。Dr.GRPO 对 R1-Zero-like 训练的分析就指出，部分长度增长来自长度归一偏差，而不全是能力增益。

所以判断推理模型，不能只盯着 `<think>` 有多长，要同时看：

$$
\text{有效推理增益}\approx
\frac{\Delta \text{终态通过率}}{\Delta \text{思考 token 或思考时间}}.
$$

分子不升、分母暴涨，就是“越想越贵”而不是“越想越强”。

![[rl-长度准确率爬升.png]]

## 原理：o1、R1 真正改变的是“训练目标 × 接口暴露 × 验证方式”

**1）训练目标变了。**
推理模型的核心不是“会输出 `<think>` 标签”，而是通过 RL 把“多步搜索 + 自我纠偏 + 终态验证”变成高奖励行为。R1-Zero 的重要结论是：即使不给人工标注的思维链，只给可验证结果奖励，长推理和自我反思也能涌现。

**2）接口暴露不统一。**
OpenAI o1 的公开说法是：raw CoT 不直接暴露，向用户展示的是摘要。DeepSeek 当前 reasoning/thinking 文档则把 `reasoning_content` 作为 API 字段暴露出来。**这两者都是产品/接口选择，不构成行业统一协议**。因此不能把“某模型可返回 reasoning_content”外推成“所有推理模型都该公开 raw CoT”。

**3）审计对象也不该是 raw CoT。**
工程上更可靠的是：

- 最终答案或引用是否正确；
- 工具调用是否完整；
- verifier 是否通过；
- 环境终态是否满足规范。

raw CoT 可能泄露敏感中间推理，也可能被策略性伪装；**可审计轨迹的重点应放在外显行为与终态验证，而不是要求所有模型暴露原始心智独白**。

**4）R1 的产品化修补很关键。**
R1-Zero 证明了纯 RL 可涌现推理，但也暴露了可读性差、语言混杂、重复等问题。DeepSeek-R1 进一步加入冷启动数据和后续蒸馏/对齐流程，把“能想”变成“能稳定交付”。

**5）蒸馏成立，但蒸的是行为分布，不是审计协议。**
把强推理模型的高质量解题轨迹蒸馏给小模型，通常比让小模型自己从零 RL 更稳。这说明推理能力可迁移；但迁移的是**求解行为与输出分布**，不意味着被蒸馏模型也该公开相同的 raw reasoning 接口。

![[rl-蒸馏vs自训.png]]

![[post-r1-pipeline.png]]

## o1 与 R1：关键差异不是“谁有 `<think>`”，而是谁暴露什么

- **OpenAI o1**：2024-09-12 官方文章说明其通过大规模 RL 获得推理能力，并明确选择**隐藏 raw CoT，只给摘要**；2024-12-05 的 system card 继续沿用这一安全取向。
- **DeepSeek-R1**：2025 技术报告公开了 R1/R1-Zero 的 RL 思路；DeepSeek 当前 API 文档把 `reasoning_content` 暴露给开发者使用。
- **共同点**：都把测试时算力、长推理、自我检查和 RL 放进核心范式。
- **不同点**：**接口层是否显示 reasoning text、是否要求回传某字段、是否展示摘要**，都是厂商约定，不是跨模型标准。

因此，面试里如果被问“推理模型是不是就是会输出 `<think>` 的模型”，正确回答应该是：**不是。`<think>` 只是某些模型/模板的外显格式；推理模型的本质是利用 RL 与测试时算力提升可验证任务的通过率，而展示 raw reasoning 与否是产品决策**。

## 代码/配置：别用 raw thinking 当验收条件

```python
def evaluate_reasoning_run(question, response):
    # ❌ 错误：把是否有 <think> / reasoning_content 当成能力或审计证据
    # return 1.0 if "<think>" in response else 0.0

    # ✅ 正确：验收外显行为和终态
    final_answer = extract_final_answer(response)
    passed = verify_final_answer(question, final_answer)
    cited = verify_citations_if_needed(response)
    return {
        "passed": passed,
        "cited": cited,
    }


def api_adapter(message):
    # 供应商差异：有的接口给 reasoning summary，
    # 有的接口给 reasoning_content，
    # 有的只给 final answer。
    #
    # 工程上把这些字段视为“可选调试/产品层输出”，
    # 不把它们当成跨模型协议或唯一验收依据。
    return normalize_vendor_response(message)
```

要点：

- `<think>`、`reasoning_content`、reasoning summary 都可能出现，也都可能不存在。
- 这些字段可用于产品体验、蒸馏或调试，但**不该直接等价成“模型真实思维过程”**。
- 高风险评估更应看 verifier、工具轨迹、终态与引用正确性。

## 面试高频

- 题库路由：[[LLM 面试题库]]
- **推理模型和普通 LLM 的根本差异是什么？** 不是提示里加不加 “think step by step”，而是推理行为是否通过 RL 和测试时算力被训练进默认策略。
- **为什么说 raw CoT、摘要和审计轨迹不是一回事？** raw CoT 是模型内部或近内部文本；摘要是产品层再表达；`reasoning_content` 是供应商 API 字段；审计轨迹是工具调用、引用、测试与终态。四者用途不同，可信边界也不同。
- **OpenAI o1 和 DeepSeek-R1 在 reasoning 可见性上有什么差异？** OpenAI o1 官方明确隐藏 raw CoT、展示摘要；DeepSeek 当前 API 文档暴露 `reasoning_content`。这反映厂商接口选择，不是统一标准。
- **R1-Zero 证明了什么？** 即便没有人工标注思维链，只用可验证结果奖励做 RL，也能诱导长推理、自我反思与自检行为。
- **为什么“长 CoT 变长”不等于“能力更强”？** 因为长度可能来自有效搜索，也可能来自训练偏差或模板冗长；必须和终态通过率、AIME/HumanEval 等可验证指标一起看。
- **什么是测试时算力？** 推理阶段允许模型花更多 token 或更长时间搜索答案；它是与预训练算力不同的一条 scaling 维度。
- **为什么不能把 `<think>` 当跨模型协议？** 因为很多模型根本不暴露 raw reasoning；即使暴露，字段名、回传规则和语义也由供应商定义。
- **如果要做安全审计，应该看什么？** 看最终答案、引用、工具调用、隐藏测试与环境终态，而不是强依赖 raw CoT。
- **蒸馏为什么常常比小模型直接做 RL 更稳？** 因为小模型从零探索长推理更难，模仿强模型已经形成的高质量求解轨迹通常更样本高效。

## 关键事实

- OpenAI, *Learning to Reason with LLMs*, 发布于 **2024-09-12**：o1 通过大规模 RL 获得推理能力，并明确说明对用户展示的是 reasoning summary，而不是 raw chain-of-thought。
- OpenAI, *OpenAI o1 System Card*, 更新于 **2024-12-05**：继续把 reasoning safety 和隐藏 CoT 作为重要安全边界。
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025，arXiv **2501.12948**：R1-Zero 展示了纯 RL + 可验证奖励可诱导长推理与自我反思。
- DeepSeek API Docs（核验日 **2026-07-19**）：`deepseek-reasoner` / thinking mode 文档明确提供 `reasoning_content` 字段；这说明 reasoning 暴露是供应商接口设计，而非通用协议。
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*, 2025，arXiv **2503.20783**，COLM 2025：指出部分响应变长来自 GRPO 长度偏差，不能把“更长”简单等同于“更会推理”。
- 工程上更稳的验收对象是**最终答案、工具轨迹、引用与终态 verifier**，而不是要求所有推理模型暴露 raw CoT。
- 关联：[[088 GRPO 与可验证奖励|GRPO]]、[[090 RLAIF、宪法 AI 与过程奖励 PRM|PRM]]、[[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF/奖励黑客]]、[[099 剪枝、稀疏化与蒸馏压缩|蒸馏]]、[[110 下游基准：MMLU、GSM8K、HumanEval、MT-Bench|GSM8K/HumanEval]]、[[32 Agentic RL 与训练|Agentic RL]]
