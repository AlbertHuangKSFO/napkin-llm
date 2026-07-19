[[090 RLAIF、宪法 AI 与过程奖励 PRM|RLAIF、宪法 AI 与 PRM]]:它们都在回答同一个问题——RLHF 的奖励信号到底从哪来；但三者解决的是不同层次：RLAIF 换掉“谁来打偏好分”，宪法 AI 换掉“按什么原则修正输出”，PRM 换掉“奖励只看最终结果还是看中间步骤”。

## 直觉：不是都在“替代人类”，而是在改奖励颗粒度

[[083 奖励模型 RM|RM]] 与 [[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF]] 的瓶颈之一是人类偏好标注：慢、贵、风格不稳定。于是有三条常见改法：

- **RLAIF**：让 LLM 裁判代替人类偏好标注者。
- **宪法 AI**：把“无害原则”写成显式文本，再让模型自我批判、自我修订。
- **PRM**：不只看最后答案对不对，而是给“做到这一步时，这条路径还靠谱吗”打分。

这里也有个常见误区要纠正：**PRM 不是“普适优于 ORM 的通用真理”**。更准确的说法是：在像 MATH 这类可以明确标注中间步骤正确性的任务上，过程监督常常更有信息量；但换任务、换步骤切分方式、换搜索/聚合器，结论不一定自动成立。

![[post-rlaif-cai.png]]

## 小数字手算：为什么 PRM 能发现“过程错但最后蒙对”

看一道四步数学题，假设模型生成 4 个步骤，逐步正确性标签是

$$
\ell=\{1,1,0,1\}.
$$

如果你只做 ORM（Outcome Reward Model），只看最后答案恰好写对了，那么整条解法可能拿到

$$
r_{\text{ORM}}=1.
$$

但这会掩盖第 3 步已经走歪。

若使用 PRM，把每一步是否仍“在正确轨道上”建模为

$$
q_t=p_\phi(\ell_t=1\mid x,s_{\le t}),
$$

例如某条轨迹得到

$$
q=\{0.95,0.90,0.20,0.70\}.
$$

那么你可以用乘积近似“整条路径都靠谱”的概率：

$$
\prod_{t=1}^4 q_t
=0.95\times0.90\times0.20\times0.70
=0.1197.
$$

或者更保守，直接取最弱一步：

$$
\min_t q_t=0.20.
$$

这时即便最后答案蒙对，PRM 也会把“第 3 步已经明显跑偏”暴露出来。**这就是 PRM 的价值：它把信用分配从“只在终点算总账”改成“沿路分段记账”**。

## 公式推导：PRM 学的是“前缀条件下的下一步质量”

PRM 不是给孤立的某一步打分，而是给**题目 + 已有前缀 + 当前候选步**打分。更准确的形式应写成：

$$
q_t=p_\phi(\ell_t=1\mid x,s_{\le t}),
$$

其中 $x$ 是题目，$s_{\le t}$ 是到第 $t$ 步为止的推理前缀，$\ell_t$ 是“当前步是否把路径保持在正确轨道上”的标签。

训练损失常写成逐步二分类：

$$
\mathcal{L}_{\text{PRM}}
=-\sum_t
\Big[
\ell_t\log q_t+(1-\ell_t)\log(1-q_t)
\Big].
$$

注意这里的条件是 **$(x,s_{\le t})$**，而不是孤立地看某个 step 文本。少掉这个条件，就会把 PRM 错写成“判断一句话本身是否正确”，这不对，因为同一句操作在不同题目、不同前缀下意义完全不同。

整条路径怎么聚合，各家实现并不统一，常见有两类：

$$
\text{score}_{\prod}(y)=\prod_t q_t,
\qquad
\text{score}_{\min}(y)=\min_t q_t.
$$

前者更平滑，后者更像“木桶短板”。**PRM 的有效性不仅取决于模型本身，还取决于：步骤如何切分、标签怎么造、聚合器怎么选、以及它是被用于训练奖励还是用于推理时重排。**

![[cai-两阶段.png]]

**RLAIF** 则保持 RLHF 主流程不变，只把偏好标注者从人换成 LLM：

$$
(y_w,y_l)\xleftarrow{\text{LLM judge}}(A,B),
\qquad
\text{后续仍可接 RM / PPO / DPO}.
$$

**宪法 AI** 把“对齐原则”显式化，再让模型基于原则先批判、后修订。监督阶段做自我修订样本，RL 阶段再用 AI feedback 做偏好比较。它解决的是**原则可审计**与**减少人类暴露有害内容**的问题，不等于 PRM，也不等于普通 RLAIF。

## 能力边界：Lightman 2023 的结果是“特定设置有效”，不是“到处必胜”

OpenAI 的 *Let’s Verify Step by Step*（2023）展示了：在数学推理数据上，过程监督可优于只看最终答案的结果监督。PRM800K 仓库还公开了大规模 step-level 标注及 held-out 评测拆分。

但面试里更成熟的回答应当是：

- **在可清晰切步、可稳定标注中间正确性的任务上**，PRM 往往比 ORM 更有信息量；
- **在步骤边界模糊、答案开放、语言风格影响更大的任务上**，PRM 的标签质量和聚合规则会变得不稳；
- **PRM 不天然替代终态 verifier**，尤其在代码/Agent 任务里，最终环境状态仍比“看起来像正确步骤”更重要。

因此，PRM 更适合被描述成：**对某些推理任务极有价值的更细粒度奖励器，而不是一个跨任务普适压制 ORM 的金科玉律**。

**Math-Shepherd 为什么重要**：它用 rollouts 把“这一步未来还有没有希望”转成统计标签，降低了逐步人工标注成本。但这本质上仍然依赖任务可验证性和 rollout 质量，不是零前提自动成立。

![[prm-MathShepherd树.png]]

![[post-prm-vs-orm.png]]

## 代码/配置：PRM 的输入必须包含前缀

```python
# ---- RLAIF: 用 LLM 裁判替代人类偏好标注 ----
def ai_preference(prompt, resp_a, resp_b, rubric):
    # ❌ 固定 A/B 顺序，容易吃位置偏置
    # return judge(prompt, resp_a, resp_b, rubric)

    # ✅ 顺序交换 + 规则明确 + 分歧时丢弃或复判
    v1 = judge(prompt, resp_a, resp_b, rubric=rubric)
    v2 = judge(prompt, resp_b, resp_a, rubric=rubric)
    return aggregate_pairwise_votes(v1, flip(v2))


# ---- Constitutional AI: 先按原则自我批判，再自我修订 ----
def self_revise(prompt, draft, principle):
    critique = model(f"按原则指出问题：{principle}\n\n{draft}")
    revised = model(f"基于批判重写答案：\n原文：{draft}\n批判：{critique}")
    return revised


# ---- PRM: 题目 + 前缀 + 当前步 -> 当前步质量 ----
def prm_step_score(question, prefix_steps, next_step):
    # ❌ 错：只给 next_step 打分，忽略题目与前缀
    # return prm(next_step)

    # ✅ 对 (question, prefix_steps, next_step) 条件化
    return prm(question=question, prefix=prefix_steps, step=next_step)


def path_score(question, steps):
    prefix = []
    scores = []
    for step in steps:
        q = prm_step_score(question, prefix, step)
        scores.append(q)
        prefix.append(step)
    return min(scores)  # 也可以改成 prod(scores)
```

要点：

- PRM 的判断对象必须是“前缀条件下的下一步”，不是孤立 step。
- RLAIF 重点是**降低偏好标注的人力依赖**，但 judge 偏置会直接传给下游模型。
- 宪法 AI 重点是**显式原则 + 自我修订流程**，适合无害对齐语境。

## 面试高频

- 题库路由：[[LLM 面试题库]]
- **RLAIF、宪法 AI、PRM 分别解决什么问题？** RLAIF 换偏好标注者，宪法 AI 把对齐原则显式化并用来做自我修订，PRM 把奖励从“只看终点”细化到“沿途分段打分”。
- **PRM 和 ORM 的根本差异是什么？** ORM 对整条输出给一个终态分；PRM 对每个前缀上的下一步质量建模，核心是更细粒度的信用分配。
- **为什么说 PRM 不是普适优于 ORM？** 因为它依赖可稳定切步、可标注中间正确性、合适聚合器与具体任务；在开放式任务里，这些前提可能不成立。
- **PRM 的输入为什么必须带前缀？** 因为同一句 step 在不同题目或不同前序步骤下含义不同；孤立打分会丢失逻辑上下文。
- **Lightman 2023 证明了什么，没证明什么？** 证明了在其数学推理与 held-out 评测设置中，过程监督可优于结果监督；没证明“所有任务、所有 PRM 设计都优于 ORM”。
- **Math-Shepherd 的价值是什么？** 用 rollouts 自动近似步级标签，降低人类逐步标注成本，但前提仍是任务可验证且 rollout 质量足够。
- **RLAIF 的主要风险是什么？** 位置偏置、长度偏置、自我偏好和谄媚会被 judge 放大后传给下游模型。
- **宪法 AI 的主要优势是什么？** 原则显式、可审计，能减少人工编写有害样本与人工暴露风险。
- **代码/Agent 任务里，PRM 能替代终态验证吗？** 不能。PRM 更像路径质量估计器；真正的交付正确性仍要看测试、引用和环境终态。

## 关键事实

- Bai et al.（Anthropic）, *Constitutional AI: Harmlessness from AI Feedback*, 2022，arXiv **2212.08073**：提出显式宪法原则、自我批判/修订与 AI feedback 结合的对齐流程。
- Lightman et al.（OpenAI）, *Let’s Verify Step by Step*, 2023，arXiv **2305.20050**：展示在数学推理设定下，过程监督可优于结果监督，并发布 PRM800K。
- OpenAI `prm800k` 仓库（核验于 **2026-07-19**）说明其为 step-level correctness labels 数据，并在 held-out 500 题上评估大规模 ORM/PRM；这支持“该结论依赖具体评测设置”的限定说法。
- Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*, 2023，arXiv **2312.08935**：用 rollout 统计构造近似步级标签，降低 PRM 标注成本。
- Lee et al., *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*, 2023，arXiv **2309.00267**：展示 AI feedback 可用于替代部分人类偏好标注，但 judge 偏置仍是核心风险。
- 对推理/代码/Agent 任务，PRM 常用于“路径重排、搜索剪枝、奖励整形”；终态正确性仍需 [[088 GRPO 与可验证奖励|RLVR]] 或独立 verifier 兜底。
- 关联：[[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF]]、[[083 奖励模型 RM|RM]]、[[088 GRPO 与可验证奖励|GRPO / RLVR]]、[[089 推理模型与 RL：o1、R1 的长 CoT 与自我反思|推理模型]]、[[086 DPO 直接偏好优化(推导)|DPO]]、[[32 Agentic RL 与训练|Agentic RL]]
