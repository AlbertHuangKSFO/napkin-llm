[[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]] 把单条推理改成多个可评估的候选状态：展开分支、评价、保留有希望的路径，必要时回到较早状态继续探索。[[09 ReAct|ReAct]] 更像顺着一条轨迹交替行动；ToT 与 LATS 关注如何搜索多条轨迹。

其中 **ToT（Tree of Thoughts）** 搜索由文本「thought」构成的推理状态；**LATS（Language Agent Tree Search）** 用 MCTS 将语言模型的行动、环境反馈、价值估计和自我反思放进搜索循环。

## 直觉：走迷宫时保留岔路

单链推理像每个路口只选一扇门，走错才发现尽头；树搜索会记下多扇门、先探索更有希望的，再回到岔路。它不保证找到正确解，优势来自「有可靠的比较信号时不必押注第一个续写」。

## 小数字手算：beam 与 UCT 各在算什么

**ToT 的一层 beam。** 当前状态生成三个候选，Evaluator 给分 $0.8,0.3,0.7$；`beam=2` 只保留 $0.8$ 和 $0.7$，丢弃 $0.3$。这一步是剪枝，不等于证明被丢弃路径错误；Evaluator 误判会造成不可恢复的漏解。

**LATS 的一次选择。** 设某子节点累计价值 $Q=3$、访问次数 $N=4$、父节点访问次数 $N_p=16$、探索常数 $C=1.4$：

$$
\operatorname{UCT}=\frac{3}{4}+1.4\sqrt{\frac{\ln16}{4}}
=0.75+1.4\sqrt{0.693}
\approx1.916.
$$

第一项偏向平均价值高的节点，第二项让访问较少的节点仍有机会被探索。未访问节点通常先被赋予无限优先级或单独处理，不能直接除以零。

## 公式推导：成本由保留宽度决定

ToT 若每层对 $b$ 个保留状态各生成 $k$ 个候选、搜索深度为 $d$，仅生成/评分候选数的近似量级为：

$$
N_{\mathrm{eval}}\approx d\,b\,k.
$$

这是 beam 固定时的线性上界；**未剪枝**的完整 $k$ 叉树才会有 $1+k+\cdots+k^d=O(k^d)$ 个节点。因而成本控制要写清 `beam`、深度、每节点采样数和评价调用次数，不能笼统地声称所有树搜索必然指数增长。

LATS 常用的选择准则为：

$$
\operatorname{UCT}(n)=\frac{Q(n)}{N(n)}+C\sqrt{\frac{\ln N(\mathrm{parent})}{N(n)}}.
$$

反向传播把叶子评估写回路径上的 $Q,N$，让上式在后续迭代中改变选择。

## ToT：思维状态的搜索

![[树搜索：ToT 与 LATS-思维树.png]]

ToT 将连贯的一段「thought」作为节点而非逐 token 决策：

1. 分解任务，定义每步 thought 的粒度；
2. 从当前状态生成多个候选 thought；
3. 以 LLM 评价、投票或任务规则评估状态；
4. 用 BFS、DFS 或 beam search 保留/回溯分支。

Yao et al. 的原论文在 Game of 24 上报告 GPT-4 的 Chain-of-Thought 为 **4%**、其 ToT 设置为 **74%**；这是论文任务、提示和搜索配置下的结果。ToT 的论文主要讨论语言推理和搜索，并不把外部工具调用作为其定义条件。

- 论文（2023，NeurIPS 2023）：https://arxiv.org/abs/2305.10601
- 作者提供的代码链接：https://github.com/princeton-nlp/tree-of-thought-llm

## LATS：MCTS 加行动、反馈与反思

![[树搜索：ToT 与 LATS-MCTS.png]]

LATS 在每轮迭代中执行：

1. **Selection**：以 UCT 从根选择待扩展节点；
2. **Expansion**：生成候选行动和后继状态；
3. **Evaluation**：执行行动获得环境反馈，再用语言模型价值函数或任务反馈评估；
4. **Backpropagation**：将价值与访问统计沿路径回传。

论文还使用语言反思帮助之后的探索，因此它和 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 相连；但反思文本是额外信息，不会自动把不可靠的评价器变成可靠评价器。

Zhou, Yan, Shlapentokh-Rothman, Wang 与 Wang 的论文在多类任务上评估 LATS，并报告在其 GPT-4 HumanEval 设置中 **92.7% pass@1**，在其 GPT-3.5 WebShop 设置中平均分 **75.9**。这些数字只描述该论文协议，不能写成不带范围的「通用 SOTA」。

- 论文（2023 arXiv，ICML 2024）：https://arxiv.org/abs/2310.04406
- 作者提供的代码链接：https://github.com/lapisrocks/LanguageAgentTreeSearch

## 可运行的最小搜索骨架

```python
import math

# ❌ 贪心：只收下第一个候选；它既不比较分支，也不保留回退点。
def greedy_step(states, generate):
    return [generate(states[0], 1)[0]]

assert greedy_step([0], lambda _, __: [1, 2, 3]) == [1]

# ✅ 改进：beam 保留多条高分路径；UCT 在价值与探索之间选择。
def choose_uct(children, parent_visits, c=1.4):
    def score(node):
        if node["visits"] == 0:
            return math.inf
        exploit = node["value"] / node["visits"]
        explore = c * math.sqrt(math.log(parent_visits) / node["visits"])
        return exploit + explore
    return max(children, key=score)

def beam_step(states, generate, evaluate, beam=2, samples=3):
    candidates = [child for s in states for child in generate(s, samples)]
    return sorted(candidates, key=evaluate, reverse=True)[:beam]

assert beam_step([0], lambda _, __: [1, 2, 3], lambda x: x, beam=2) == [3, 2]
assert choose_uct([{"value": 3, "visits": 4}], parent_visits=16)["value"] == 3
```

真实 agent 必须把 `evaluate` 接到可审计的信号，并对工具副作用采用沙箱、只读接口或补偿机制；不能让 MCTS 用真实生产写操作反复试探。

## 对比：限定在搜索结构

| 维度 | [[09 ReAct\|ReAct]] | ToT | LATS |
|---|---|---|---|
| 状态 | 一条行动/观测轨迹 | 中间 thought | 带行动与环境反馈的搜索节点 |
| 分支与回退 | 实现可额外加入，但原循环偏单链 | 显式生成、评价、剪枝 | MCTS 的选择、扩展、回传 |
| 评价信号 | 通常由下一轮模型吸收观测 | LLM 评价或任务规则 | 价值函数与环境反馈 |
| 外部行动 | 可有 | 非定义重点 | 论文框架包含环境交互 |

[[12 LLMCompiler|LLMCompiler]] 的 DAG 调度解决的是「已知依赖下如何并行执行」，而树搜索解决的是「候选状态不确定时探索哪条路径」；两者可组合，但不是同一种图算法。

![[树搜索：ToT 与 LATS-谱系对比.png]]

## 何时用 / 边界

适合：任务确实需要试探与回退、可以定义可靠的状态/终局验证、并能给搜索设置成本上限，例如可在沙箱中运行的代码测试或约束满足问题。

不适合：线性且低风险的任务、没有可信评价信号的开放式任务、或不可撤销的生产写操作。

- 先验证 Evaluator；错误的评价会系统性保留坏分支或剪掉好分支。
- 明确限制深度、beam、MCTS 迭代次数、工具调用数和 token 预算；达到预算时返回「当前最优且未验证完全」，而非伪装为已找到解。
- 分支间必须使用可比较的分数尺度；否则 beam 和 UCT 都失去含义。

## 面试高频

**Q：ToT 与 LATS 的区别？**

答：ToT 把 thought 当状态，用生成、评价与搜索探索推理分支；LATS 在 MCTS 中把行动、环境反馈、价值估计和语言反思结合起来。前者的核心是审慎文本推理搜索，后者的核心是 agent 式 MCTS。

**Q：树搜索一定优于单链吗？**

答：不是。只有当评价信号可信且回退确有价值时，多分支探索才可能抵偿额外成本。评估器不可靠或任务线性时，搜索会更慢且可能更差。

## 关键事实

- ToT（Yao et al., NeurIPS 2023）报告了 Game of 24 中 4% 到 74% 的论文设置结果（Yao et al., 2023, [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)）。
- LATS（Zhou et al., ICML 2024）以 MCTS、环境反馈、LM value function 与 self-reflection 组织 agent 搜索（Zhou et al., 2023, [arXiv:2310.04406](https://arxiv.org/abs/2310.04406)）。
- 宽度受控的 beam search 的候选量约为 $d\,b\,k$；未剪枝树才呈指数节点增长（由本笔记「公式推导」直接展开；ToT 的搜索设定见 Yao et al., 2023, [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)）。
- 树搜索的上限由评价器/验证器决定，相关工程监控见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]，成本约束见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]（Yao et al., 2023, [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)；Zhou et al., 2023, [arXiv:2310.04406](https://arxiv.org/abs/2310.04406)）。
