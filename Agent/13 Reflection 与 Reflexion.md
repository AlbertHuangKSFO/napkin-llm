[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 都让模型根据反馈改下一次尝试，但层级不同：**Reflection** 是泛称的「产出—批评—修订」循环；**Reflexion** 是 Shinn et al. 提出的具体 agent 框架，用自然语言反思写入 episodic memory，再用于后续 trial。

关键澄清：Reflexion 的「reinforcement」发生在上下文与记忆层，原始流程**不更新模型参数**。它不是在线微调，也不能保证每轮都会进步。

## 直觉：批改一题，还是记住错因

Reflection 像老师在同一张草稿上圈出「单位错了」，学生立即重写；Reflexion 则会把「以后先检查单位」写进错题本，下一道相似题开始前先读错题本。前者重在当前输出修订，后者多了一条跨尝试可检索的文字记忆。

## 小数字手算：记忆只在反馈可区分时有用

设一个代码任务有 3 个测试，第一次输出通过 2 个，得分为 $r_1=2/3$。Evaluator 给出可行动反馈「边界输入为空列表时返回索引错误」，Self-Reflection 将它压缩为规则 $m_1$。第二次只在修复这一边界后通过 3 个测试，则 $r_2=3/3$，提升为：

$$
\Delta r=r_2-r_1=1/3.
$$

这只说明一条具体反馈如何被利用；若测试不覆盖真实错误、反思写错原因或任务本身不可重试，写更多记忆不会产生同样结论。

## 公式推导：更新的是上下文，不是权重

将 Actor 的一次尝试写为 $a_t=\pi_\theta(x,m_t)$，Evaluator 给出反馈 $f_t=E(x,a_t)$，反思器写入简洁经验 $r_t=R(x,a_t,f_t)$：

$$
m_{t+1}=\operatorname{select}(m_t\cup\{r_t\}),\qquad \theta_{t+1}=\theta_t.
$$

第二个等式正是与训练型 RL 的边界：Reflexion 用 $m_{t+1}$ 改变下一次 prompt 条件，而非通过梯度改变 $\theta$。`select` 必须控制长度和相关性，否则旧反思会污染上下文。

## 机制：Actor / Evaluator / Self-Reflection / Memory

![[Reflection 与 Reflexion.png]]

1. **Actor** 根据任务与已选记忆行动，例如生成代码、做网页操作或回答问题。
2. **Evaluator** 提供可用反馈；它可来自环境奖励、单元测试、规则校验或语言形式的评价。
3. **Self-Reflection** 把失败轨迹和反馈压成对下一次可执行的文字建议，而不是复述整个日志。
4. **Memory** 保存并检索这些建议，供后续 trial 使用。

![[Reflection 与 Reflexion-记忆回灌.png]]

## 原论文与可核验结论

Shinn et al. 的 *Reflexion: Language Agents with Verbal Reinforcement Learning* 发表在 NeurIPS 2023。论文报告在其 HumanEval 设置中，Reflexion 达到 **91% pass@1**，而论文所比较的 GPT-4 基线为 **80%**；这是特定模型、提示与评测协议下的结果，不能直接解释为任意代码 agent 的预期增益。

- 论文（2023）：https://arxiv.org/abs/2303.11366
- 作者提供的代码链接：https://github.com/noahshinn/reflexion

## 可运行的最小循环

```python
# ❌ 朴素单次作答：没有反馈，也没有可带到下一次尝试的经验。
def actor(_, memory):
    return "return 0 if empty else xs[0]" if "检查空列表" in memory else "return xs[0]"

def evaluate(_, answer):
    return ("if empty" in answer, "空列表要有返回值")

assert not evaluate("first", actor("first", []))[0]
assert actor("first", ["无关记忆"]) == "return xs[0]"

# ✅ 改进：将可行动的反馈写成 memory，再让下一次 Actor 读取它。
def reflexion(task, actor, evaluate, reflect, memory, trials=3):
    for _ in range(trials):
        answer = actor(task, memory)
        ok, feedback = evaluate(task, answer)
        if ok:
            return answer
        lesson = reflect(task, answer, feedback)
        if lesson and lesson not in memory:
            memory.append(lesson)
    raise RuntimeError("达到重试预算仍未通过")

expected = "return 0 if empty else xs[0]"
assert reflexion("first", actor, evaluate, lambda *_: "检查空列表", []) == expected
```

生产中应把 `memory` 变成有来源、可过期、可删除的结构化记录；不要无限追加自然语言反思。

## 对比：不要把词混用

| 维度 | Reflection（通用模式） | Reflexion（论文框架） | 训练型 RL |
|---|---|---|---|
| 改变什么 | 当前输出或下一轮提示 | 后续 trial 的文字记忆 | 模型参数或策略 |
| 是否要求跨 trial | 不要求 | 是其关键设计 | 通常需要采样与优化 |
| 反馈例子 | 审稿意见、schema 错误 | 环境/测试反馈加自我反思 | 奖励信号 |
| 主要风险 | 自我批评不可靠 | 记忆污染与错误归因 | 训练成本与奖励错设 |

## 何时用 / 边界

适合：有可验证反馈、允许重试、并且失败原因能压缩成可复用规则的任务，例如测试失败后的代码修订。

不适合：不能安全重试的写操作、反馈延迟很长、或 Evaluator 只给无解释且不稳定分数的任务。此时先改善验证器或增加人工审查，而不是堆叠反思轮数。

- 反思应引用具体失败证据，避免「更仔细一些」这类不可行动总结。
- 限制重试次数、token 和总工具预算；每轮重新验证，而不是相信模型宣称已修复。
- 对跨任务记忆做检索过滤和失效处理，防止旧规则覆盖当前约束。

## 面试高频

**Q：Reflection 与 Reflexion 的核心区别？**

答：Reflection 是广义的自评修订机制；Reflexion 是带 Actor、Evaluator、Self-Reflection 和 episodic memory 的具体框架，强调把语言化经验带到下一次尝试。

**Q：为什么称 verbal reinforcement learning，却不是参数 RL？**

答：反馈被转成文本记忆并写进下一次上下文，原始流程的模型权重不变。它利用语言反馈改变行为条件，而不是通过梯度更新策略。

## 关键事实

- Reflexion 论文发表于 NeurIPS 2023，核心是语言反思与跨 trial 的 episodic memory（Shinn et al., 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)）。
- 91% 与 80% 是论文 HumanEval 设置内的对比数字，引用时必须带上其基准语境（Shinn et al., 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)）。
- [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 解释「生成—评价—改进」抽象；Reflexion 在此之上强调记忆和多次尝试（Shinn et al., 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)）。
- [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]] 可把反思作为搜索过程的附加信息，但反思不等于可靠的价值函数（Shinn et al., 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)；Zhou et al., 2023, [arXiv:2310.04406](https://arxiv.org/abs/2310.04406)）。
