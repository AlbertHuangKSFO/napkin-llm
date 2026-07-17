[[11 ReWOO|ReWOO]]（Reasoning WithOut Observation）把「先规划、再取证、最后作答」拆开：Planner 写出带 `#E1` 等变量的计划，Worker 调工具填值，Solver 只在最后看一次完整证据并回答。

它是 [[09 ReAct|ReAct]] 的一个明确取舍：不让每一步观测重新进入下一轮推理，换取更少的提示词重复；代价是计划期间看不到环境反馈。

## 直觉：先列采购单，再统一验货

把 Planner 想成先写好「买面粉、买鸡蛋、再计算总价」的采购单；Worker 只负责按单取货并填入价格；Solver 最后才检查采购单和收据、写出答案。[[09 ReAct|ReAct]] 则是每买一样就回厨房重新问一次下一步。因此 ReWOO 适合采购单大体可预见的任务，不适合「打开盒子后才知道下一步该做什么」的探索。

## 小数字手算：省在观测被少带几次

以下是**教学账本，不是论文基准**。设固定指令为 $P=120$ token，三次工具观测各为 $O_1=O_2=O_3=300$ token。若 ReAct 每次工具选择后都把已有观测放入下一次决策，最后还需一次模型调用作答，则提示词约为：

$$
120+(120+300)+(120+300+300)+(120+300+300+300)=2{,}280.
$$

若 ReWOO 用一次 $120$ token 的规划和一次携带三条证据的总结，则约为：

$$
120+(120+3\times300)=1{,}140.
$$

这里差额不大，因为只有三步；步数和观测长度增加时，早期观测在 ReAct 中会被重复携带更多次。实际 token 还取决于系统提示、工具格式和是否压缩证据，不能把这个算例当作固定倍率。

## 公式推导：重复项来自哪里

令第 $j$ 条观测长度为 $O_j$，规划/决策固定上下文近似为 $P$。一个 $n$ 步交错循环的提示词量可粗略写成：

$$
T_{\mathrm{interleaved}}\approx \sum_{i=0}^{n}\left(P+\sum_{j=1}^{i}O_j\right).
$$

交换求和顺序可见第 $j$ 条观测大约出现 $n-j+1$ 次：

$$
\sum_{i=0}^{n}\sum_{j=1}^{i}O_j
=\sum_{j=1}^{n}(n-j+1)O_j.
$$

ReWOO 的理想化提示词量则近似为一次规划、一次汇总和一次证据集合：

$$
T_{\mathrm{ReWOO}}\approx P_{\mathrm{plan}}+P_{\mathrm{solve}}+\sum_{j=1}^{n}O_j.
$$

这解释了它的收益来源，也说明其前提：最终 Solver 仍要能容纳并正确使用全部必要证据。

## 机制：Planner / Worker / Solver

![[ReWOO.png]]

1. **Planner（LLM）** 一次生成计划和变量，例如 `#E1 = Search[城市 A 人口]`、`#E2 = Calculate[#E1 * 0.05]`。变量引用表达数据依赖，而不是证据已经到手。
2. **Worker（工具执行层）** 按依赖顺序把 `#En` 替换成实际工具结果；没有依赖边的任务可以由调度器并发执行。Worker 是否完全不用 LLM 是该论文范式的设计点；工程实现也可额外加入解析或校验模块，但那应单独计入成本。
3. **Solver（LLM）** 接收问题、计划和已解析的证据表，进行一次最终合成。

![[ReWOO-占位DAG.png]]

## 原论文与可核验结论

Xu et al. 在 2023 年提出 ReWOO，并将其描述为把推理与外部观测解耦的模块化范式。论文报告：在 HotpotQA 设置中，相对其比较的 ReAct 配置达到 **5× token efficiency** 和 **4% accuracy improvement**；这些是该论文的特定实验结果，不能外推为所有模型、工具和提示词的通用倍率。论文还展示了以 GPT-3.5 规划数据微调 7B LLaMA 的实验。

- 论文（2023）：https://arxiv.org/abs/2305.18323
- 作者提供的代码链接：https://github.com/billxbf/ReWOO

## 可运行的最小骨架

```python
import re

VAR = re.compile(r"#E\d+")

steps = [('#E1', 'lookup', '苹果价格'), ('#E2', 'percent', '#E1')]
tools = {'lookup': lambda _: 10, 'percent': lambda x: float(x) * 0.05}

# ❌ 朴素串行：照抄参数执行，第二步拿到 '#E1' 而不是真实证据。
def naive_worker(steps, tools):
    evidence = {}
    for variable, tool_name, raw_arg in steps:
        evidence[variable] = tools[tool_name](raw_arg)
    return evidence

try:
    naive_worker(steps, tools)
except ValueError:
    pass  # float('#E1')：变量尚未替换，串行不等于依赖已解析。

# ✅ 改进：执行前把已就绪变量替换为真实证据。
def substitute(text, evidence):
    missing = [v for v in VAR.findall(text) if v not in evidence]
    if missing:
        raise ValueError(f"尚未就绪的依赖：{missing}")
    return VAR.sub(lambda m: str(evidence[m.group(0)]), text)

def worker(steps, tools):
    """steps 形如 [('#E1', 'lookup', 'Beijing'), ...]。"""
    evidence = {}
    for variable, tool_name, raw_arg in steps:
        if tool_name not in tools:
            raise KeyError(f"未知工具：{tool_name}")
        evidence[variable] = tools[tool_name](substitute(raw_arg, evidence))
    return evidence

assert worker(steps, tools) == {'#E1': 10, '#E2': 0.5}
```

生产实现应在执行前检查变量是否重复、是否有环以及工具失败如何表示；上例刻意只展示替换与失败可见性，不把 `eval` 用于工具参数。

## 对比：只比较架构假设

| 维度 | [[09 ReAct\|ReAct]] | [[10 Plan-and-Execute\|Plan-and-Execute]] | ReWOO |
|---|---|---|---|
| 何时读观测 | 每轮决策后 | 可在执行和重规划间读 | 论文范式中由 Solver 在末尾汇合 |
| 对中途意外的响应 | 可据最新观测继续行动 | 可选择重规划 | 原始流程不以中途重规划为核心 |
| 并行机会 | 取决于实现，交错循环常偏串行 | 取决于计划与调度 | 无数据依赖的变量可并发 |
| 主要优化目标 | 交替推理与行动 | 显式规划 | 避免重复携带观测 |

[[12 LLMCompiler|LLMCompiler]] 同样会表达依赖，但其论文重点是把函数调用调度成可并行的执行计划；不要把两者的报告指标直接横比，模型、工具、任务和计费方式不同。

## 何时用 / 边界

适合：工具步骤和依赖能在取证前大体写出、观测文本较长、且最后一次汇总仍能放进上下文的任务。

不适合：下一动作必须由刚返回的真实值决定、外部环境经常意外变化、或失败后必须立即调整策略的任务。这类任务更接近 [[09 ReAct|ReAct]] 的交错反馈优势。

- 先对变量图做拓扑检查；未定义引用、循环依赖和同名变量都应在调用工具前失败。
- 对证据保留来源、时间和错误状态；不要把空字符串当成正常答案继续替换。
- 若加入「失败即重规划」，这是合理的工程扩展，但不再是原始两次 LLM 调用的简单描述，应重新测量成本与成功率。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：ReWOO 为什么可能省 token？**

答：它让 Planner 在没有工具结果时写变量化计划，Worker 不把每条 observation 交回模型，Solver 最后一次性读证据。省的是交错循环中早期观测在多轮提示词里的重复出现；不是工具调用本身凭空消失。

**Q：它的主要失败模式是什么？**

答：盲规划。若下一步强依赖刚得到的事实，预先写好的变量计划可能失效；还要防变量依赖环、工具失败和最终证据过长。

## 关键事实

- ReWOO 是 **Reasoning WithOut Observation**，论文流程为 Planner、Worker、Solver 的分工（Xu et al., 2023, [arXiv:2305.18323](https://arxiv.org/abs/2305.18323)）。
- 变量引用把「计划时未知」与「执行后证据」分开；并发来自变量之间没有数据依赖，而非名称本身（Xu et al., 2023, [arXiv:2305.18323](https://arxiv.org/abs/2305.18323)）。
- 论文的 5× 与 4% 是 HotpotQA 中相对于论文比较设置的报告值，引用时应保留任务和比较对象（Xu et al., 2023, [arXiv:2305.18323](https://arxiv.org/abs/2305.18323)）。
- 相关的 [[12 LLMCompiler|LLMCompiler]] 优化函数调用并行；[[09 ReAct|ReAct]] 则优先逐步吸收观测。选择取决于反馈依赖，而不是范式名称（Xu et al., 2023, [arXiv:2305.18323](https://arxiv.org/abs/2305.18323)；Kim et al., 2023, [arXiv:2312.04511](https://arxiv.org/abs/2312.04511)）。
