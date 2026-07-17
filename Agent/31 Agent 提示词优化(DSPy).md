[[31 Agent 提示词优化(DSPy)|DSPy 与 GEPA]]：把语言模型流程中的指令与示例视作待选择的参数，用独立评估集上的度量来选择编译产物；它优化的是提示与流程配置，不训练底座模型权重。

## 直觉：把“改文案”改成“验收后选配方”

手调 prompt 像厨师凭感觉改菜谱：换一句话、试一次、觉得不错就留下。DSPy 的切分更像工厂验收：人先写清输入/输出和验收标准，系统提出若干“菜谱”（指令、few-shot 示例或模块配置），再在未参与提案的样本上比较。原始 DSPy 工作把这件事称为为给定 metric 编译 LM pipeline；它并不承诺任何 metric 都可靠，也不替人定义“什么算好”。[DSPy, 2023](https://arxiv.org/abs/2310.03714)

`[[08 Evaluator-Optimizer|Evaluator-Optimizer]]`是在运行时反复生成与评判；DSPy/GEPA 通常是开发期的离线搜索，部署时只带选出的程序与版本化配置。它们与改权重的 `[[32 Agentic RL 与训练|Agentic RL]]` 是两条不同路线。

## 小数字手算：为什么必须留出验证样本

有 4 条**未用于提出候选**的验证题，二元 metric 是“答案与标注一致”为 1，否则为 0：

| 候选提示 | 四题得分 | 平均分 |
|---|---:|---:|
| A | $[1,0,1,0]$ | $(1+0+1+0)/4=0.50$ |
| B | $[1,1,1,0]$ | $(1+1+1+0)/4=0.75$ |

所以本轮选 B，而不是因为它的措辞“看起来更专业”。若用提出候选的同一小批样本来宣布胜利，A/B 都可能只是记住了那批题；这就是要把 train 与 validation 分开的原因。

## 公式推导：metric 搜索与 GEPA 的反思信号

令 $p$ 是一个可部署的提示/示例配置，$D_{dev}$ 是独立验证集，$m\in[0,1]$ 是可复算 metric，则最简单的编译目标是

$$
p^*=\arg\max_{p\in\mathcal P}\;\frac{1}{|D_{dev}|}\sum_{(x,y)\in D_{dev}}m\bigl(y,f_p(x)\bigr).
$$

`Signature → Module → metric → compile` 是把 $f_p$ 的控制流与 $p$ 的文本参数分开：人写前者和 $m$，优化器在候选空间 $\mathcal P$ 中搜索后者。它不是“自动知道业务正确性”。

GEPA（Genetic-Pareto）论文的方法是在候选失败时利用轨迹、测试 diff 或格式错误等**文本反馈**提出修订，并保留不被其他候选全面压制的 Pareto 集。若候选 $a,b$ 在多个评估目标 $j$ 上有分数 $s_j$，则

$$
a\succ b\iff \bigl(\forall j,\;s_j(a)\ge s_j(b)\bigr)\ \land\ \bigl(\exists j,\;s_j(a)>s_j(b)\bigr).
$$

GEPA 作者在该论文的六个任务、给定模型与预算下报告：相对所比较的 GRPO 设置**平均**高 6%、**最高**高 20%，并最多少用 35 倍 rollout；同一论文还报告相对 MIPROv2 **超过 10%**，例为 AIME-2025 上准确率 $+12$ 个百分点。两组数字的比较对象与统计口径不同，都是论文设置中的结果，不能外推成“GEPA 总会胜过 RL”或“总会胜过 MIPROv2”。[GEPA 预印本，2025；修订于 2026-02](https://arxiv.org/abs/2507.19457)

## 图：编译发生在部署之前

![[Agent 提示词优化(DSPy).png]]

![[DSPy-编译流程.png]]

图中的“编译产物”应连同模型名、数据版本、metric、随机种子/预算与验证分数一起记录；否则它不能被可靠复现或回归比较。

## 可运行代码：❌ 凭感觉定稿 vs ✅ 用验证 metric 选候选

下面的例子不调用模型，而把两个候选在同一验证集上的输出写死，专门展示可运行的选择逻辑；把 `predictions` 换成真实 LM 输出即可接入同一 metric。

```python
# Python 3 标准库；运行后打印 selected=search_with_metric, score=0.75
gold = ["4", "H2O", "Paris", "blue"]
predictions = {
    "❌ hand_tuned": ["4", "water", "Paris", "red"],
    "✅ search_with_metric": ["4", "H2O", "Paris", "red"],
}

def accuracy(expected, actual):
    return sum(a == b for a, b in zip(expected, actual)) / len(expected)

# ❌ 不把“我喜欢这句指令”误当作证据。
# chosen = "❌ hand_tuned"

# ✅ 候选由训练/反馈提出；由未参与提出的验证集选出。
scores = {name: accuracy(gold, out) for name, out in predictions.items()}
chosen = max(scores, key=scores.get)
assert chosen == "✅ search_with_metric"
assert scores[chosen] == 0.75
print(f"selected={chosen[2:]}, score={scores[chosen]:.2f}")
```

实际项目还要固定验证集，并为格式、安全性、成本和延迟增加独立 metric；单一准确率可能把“答对但不可用”的输出选为最优。

## 面试高频

**DSPy 和手写 prompt 的边界是什么？** DSPy 不消灭提示词，而是把提示/示例作为可搜索配置；价值来自可复算 metric、留出的验证集和可保存的产物。一次性、没有可靠验收的任务，直接写 prompt 可能更合适。

**GEPA 与 GRPO 谁更强？** 不能脱离任务、模型、奖励和 rollout 预算比较。GEPA 的论文比较的是其文本反思式提示进化与特定 RL 设置；GEPA 不改权重，GRPO 是策略更新算法，成本与风险边界不同。[GEPA, 2025](https://arxiv.org/abs/2507.19457)

**最常见失败是什么？** 把训练集分数当作上线证据，或把有漏洞的 metric 当作目标。先做 `[[38 Agent 评估与可观测性|评估与可观测性]]`，再优化。

## 关键事实

- DSPy 论文于 2023 年提出以声明式模块和编译器为给定 metric 优化 LM pipeline；论文中的提升是其案例的实验结果，不是跨任务保证。[DSPy, 2023](https://arxiv.org/abs/2310.03714)
- GEPA 预印本发表于 2025 年，方法名中的 Pareto 指多目标候选保留；作者报告的比较应分开记：对所比 GRPO 设置为平均 $+6\%$、最高 $+20\%$、最多 $35\times$ 少 rollout；对 MIPROv2 报告超过 $10\%$，例为 AIME-2025 准确率 $+12$ 个百分点。[GEPA, 2025；修订于 2026-02](https://arxiv.org/abs/2507.19457)
- 选型顺序：先定义可审计 metric 与独立验证集；提示层优化不能满足目标且确有可验证训练环境时，再评估 `[[32 Agentic RL 与训练|Agentic RL]]`，不要把二者混为同一种“自动改进”。
