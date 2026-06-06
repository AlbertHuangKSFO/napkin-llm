**Workflow** 是用**预设代码路径**编排 LLM 与工具的系统;**Agent** 是 LLM **动态决定**自己流程与工具用法的系统——区别只有一句话:**流程是人写死的,还是模型自己定的。**

这条线来自 [[01 什么是 AI Agent|什么是 AI Agent]] 里提到的自主性谱系。本篇把这条边界画清楚:两端各是什么、怎么取舍、什么时候该跨过去。

## 本质:谁掌握「下一步做什么」的控制权

- **Workflow**:控制权在**代码**手里。工程师预先把流程画成图(顺序、分支、循环都写死),LLM 只是其中某个节点上的「智能算子」。流程可预测、可复现。
- **Agent**:控制权在**模型**手里。给定目标和一组工具,模型自己在回路里决定调哪个工具、何时停下,见 [[03 Agent 核心循环|Agent 核心循环]]。流程不定,换灵活性。

两者都把 LLM 当组件,差别在**编排权归谁**。这正是 Anthropic《Building Effective Agents》(2024-12)给出的官方切分。

![[Workflow 与 Agent 的边界.svg]]

## 机制:从同一个任务看两条路

任务:「把一封英文邮件翻成中文,若含金额则同时换算成人民币。」

**Workflow 写法**(分支写死):
```python
def handle(email):
    zh = llm(f"翻译成中文:{email}")           # 节点1
    if has_amount(zh):                          # 分支:代码判断
        rate = get_fx_rate("USD", "CNY")        # 节点2:工具
        zh = llm(f"把金额按 {rate} 换算并标注:{zh}")  # 节点3
    return zh
```
流程图固定,你能画出来、能单测每个节点、成本可估。这就是 workflow——具体编排模式见 [[04 Prompt Chaining|Prompt Chaining]]、[[05 Routing|Routing]]、[[06 Parallelization|Parallelization]]、[[07 Orchestrator-Workers|Orchestrator-Workers]]、[[08 Evaluator-Optimizer|Evaluator-Optimizer]]。

**Agent 写法**(模型自己决定):
```python
agent(goal="翻译这封邮件,如有金额换算成人民币",
      tools=[translate, get_fx_rate, calculator])
# 模型自己决定:先翻译?先查汇率?要不要算?调几次?——流程不写死
```
模型在 [[09 ReAct|ReAct]] 式回路里自己排步骤。任务一变复杂(「顺便总结要点并起草回复」),agent 不用改代码就能适应,workflow 则要重画流程图。

## 取舍:一张账

| 维度 | Workflow | Agent |
|---|---|---|
| 编排权 | 代码(人) | 模型 |
| 可预测性 | 高,流程固定 | 低,步数不定 |
| 可靠性 | 高,易复现 | 较低,可能跑飞 |
| 成本/延迟 | 低且可估 | 高且波动(多轮调用) |
| 可调试 | 易,逐节点测 | 难,要看轨迹 |
| 灵活性 | 低,改流程要改码 | 高,改目标即可 |
| 适用任务 | 步骤可预定 | 步骤不可预定、开放 |

![[Workflow 与 Agent 的边界-决策树.svg]]

## 判定清单:能 workflow 就别上 agent

按顺序问自己,**一旦能停就停**(对应上面的决策树):

1. **单次 LLM 调用够吗?**(配上检索 + 几个示例)——够,就别搞编排。
2. **步骤能预先列全吗?**——能列全/分支可写死 → 用 **Workflow**。
3. **任务是否开放、步数不可预知、需多轮试错?**——是 → 才上 **Agent**。
4. 上了 agent,**护栏配齐了吗?**——最大步数、token 预算、人工审核点、沙箱,见 [[03 Agent 核心循环|Agent 核心循环]] 的停机条件。

> Anthropic 的核心建议:**追求能用的最简方案,只在确有必要时增加复杂度**。很多人以为自己需要 agent,其实一个 workflow(甚至单次调用)就够了。

## 边界其实是连续的

别把它当非黑即白:

- **受限 agent**:模型能选工具,但工具集小、步数封顶、关键动作要人确认——介于两者之间,工程上最常见。
- **workflow 里嵌 agent**:固定流程的某个节点交给一个小 agent 处理子问题(如 [[07 Orchestrator-Workers|Orchestrator-Workers]] 里 worker 可以是 agent)。
- **agent 里嵌 workflow**:agent 调用的某个「工具」内部其实是一段固定 workflow。

所以真实系统往往是混合体;「边界」是设计时的决策刻度,不是产品分类。

## 坑与反模式

- **默认上 agent**:被「autonomous」噱头带跑,简单任务也套自主回路 → 慢、贵、难调、不稳。
- **用 workflow 硬扛开放任务**:任务步骤其实不可预定,却拿一堆 if-else 死撑,分支爆炸、维护地狱——这时该换 agent。
- **混淆「多步」和「agent」**:[[04 Prompt Chaining|Prompt Chaining]] 也是多步,但流程写死,它是 workflow 不是 agent。判据永远是**控制权归谁**。
- **agent 无护栏**:见 [[01 什么是 AI Agent|什么是 AI Agent]] 反模式——没停机条件就是烧钱机器。

## 工业界实践

这条边界不是学术划分,而是生产里**架构选型的第一道分叉**——选错一边,要么烧钱不稳(该 workflow 却上了 agent),要么分支爆炸维护地狱(该 agent 却硬用 workflow)。

**框架在这条线上的站位:**

- **偏 workflow / 图编排**:**LangGraph** 把流程显式建成有状态的图,你能画出每个节点、单测每条边、回放每次执行——本质是「**带 LLM 节点的 workflow 引擎**」,但图里也能放循环和条件边,于是同一框架既能写纯 workflow 也能写 agent,边界由你画图的方式决定。**Apache Airflow / Prefect / Temporal** 这类传统编排器也越来越多被用来串 LLM 步骤,属于「workflow 一侧」的重武器。
- **偏 agent / 自主回路**:**OpenAI Agents SDK**、**Claude Agent SDK**、**CrewAI**、**AutoGen** 默认把「下一步」交给模型,你只给目标和工具集。
- **同一框架横跨两侧**:这正是当下趋势——LangGraph、PydanticAI 都允许你在「写死分支」和「让模型决定」之间自由滑动,因为真实系统几乎都是混合体。

**生产里最常见的形态是「workflow 主干 + agent 子节点」:**

```python
# 主干是确定性 workflow(可预测、可单测、成本可估)
def pipeline(ticket):
    category = classify(ticket)            # 节点1:Routing,见 05
    if category == "refund":
        return refund_workflow(ticket)     # 固定分支:纯 workflow
    elif category == "bug":
        # 这一个节点交给受限 agent 处理开放子问题
        return debug_agent.run(ticket,
                               tools=[read_logs, run_tests, search_kb],
                               max_steps=15)   # ← 护栏:步数封顶
    else:
        return human_escalate(ticket)
```

这种「把不确定性**关进一个有护栏的小盒子**」的做法,是工业界用得最多的折中:主流程保持可预测,只在确需自主探索的局部放 agent。

**规模化与成本/延迟权衡:**
- workflow 成本**可预先估算**(步数固定),agent 成本**按轨迹波动**且常高一个数量级——预算敏感的批量任务尽量 workflow 化。
- 「workflow 里嵌 agent」时,务必给 agent 子节点单独配 max_steps + token 预算 + 超时,防止一个节点拖垮整条 SLA。

**可观测与运维:** workflow 的可观测是「看 DAG 哪个节点失败」,粒度天然清晰;agent 的可观测是「看一条不定长的轨迹」,要专门的 trace 工具(LangSmith / LangFuse)还原每步 thought/tool_call/observation,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。这也是「能 workflow 就别上 agent」在运维层面的现实理由。

**真实踩坑:**
- **把 LangGraph 当 agent 框架,结果写成了 workflow 还不自知**——很多人画了一张全是固定边的图,以为自己「做了 agent」,其实控制权全在代码,这没问题但要心里有数。
- **agent 子节点没设超时**:模型在局部反复试错,把上游 workflow 的延迟预算吃光。
- **该用 Routing 的地方上了全自主 agent**:输入明明有清晰类别,却让模型每次重新「想」该走哪条路,既慢又不稳,见 [[05 Routing|Routing]]。

## 面试高频

**Q1:Workflow 和 Agent 的唯一本质区别是什么?**
标准答:**「下一步做什么」由代码决定(workflow)还是由模型动态决定(agent)**——即编排权归谁。其它(多步、用工具、调 LLM)都不是判据。
- 陷阱:面试官给你一个「调了好几次 LLM、还用了工具」的系统问是不是 agent。别被表象骗——若流程是写死的(如 [[04 Prompt Chaining|Prompt Chaining]]),它就是 workflow。

**Q2:给定一个任务,你怎么决定用 workflow 还是 agent?**
标准答:按顺序问——① 单次调用够吗?② 步骤能预先列全/分支可写死吗?能 → workflow;③ 任务开放、步数不可预知、需多轮试错吗?是 → 才上 agent;④ 上了 agent 护栏配齐了吗?
- 追问:**为什么默认不上 agent?** 因为 agent 用可预测性/可靠性/成本换灵活性,多数任务不值这个代价。

**Q3:举个「workflow 里嵌 agent」和「agent 里嵌 workflow」的例子。**
标准答:前者——客服主流程是固定 workflow,但「排查这个 bug」这一节点交给受限 agent;后者——agent 调用的某个「工具」内部其实是一段固定 workflow(如「下单」工具内部是写死的多步流程)。说明真实系统是混合体。

**Q4:LangGraph 既能写 workflow 又能写 agent,这说明什么?**
标准答:说明边界是**连续谱系而非产品分类**——同一个图引擎,边全写死就是 workflow,加了条件边/循环让模型决定走向就偏 agent。框架只是工具,控制权归谁取决于你怎么用它。

**Q5(陷阱):"多步" 等于 "需要 agent" 吗?**
标准答:不等于。Prompt Chaining、Routing 都是多步但流程写死,是 workflow。只有当**步数和走向到运行时才由模型决定**时才是 agent。

## 知识拓展

- **官方出处**:这条切分来自 **Anthropic《Building Effective Agents》(2024-12)**,它把 agentic 系统分为两大类——**workflows**(LLM 与工具被预定义代码路径编排)和 **agents**(LLM 动态主导自身流程与工具),并强调「**追求能用的最简方案,只在确有必要时增加复杂度**」。
- **五大 workflow 模式**:workflow 一侧被进一步细分为五种可组合模式,正好是本域 04–08:[[04 Prompt Chaining|Prompt Chaining]](串行)、[[05 Routing|Routing]](分流)、[[06 Parallelization|Parallelization]](并行)、[[07 Orchestrator-Workers|Orchestrator-Workers]](动态分发)、[[08 Evaluator-Optimizer|Evaluator-Optimizer]](评估闭环)。
- **edge case:Orchestrator-Workers 是 workflow 还是 agent?** 它在边界上——orchestrator 用 LLM **动态决定**要派几个 worker、派什么任务(像 agent),但整体编排骨架仍是预设的(像 workflow)。Anthropic 把它归到 workflow,但它是「最像 agent 的 workflow」,见 [[07 Orchestrator-Workers|Orchestrator-Workers]]。
- **反模式总览**:① 默认上 agent(被 "autonomous" 噱头带跑);② 用 workflow 硬扛开放任务(if-else 分支爆炸);③ 混淆「多步」与「agent」;④ agent 无护栏。
- **深层联系**:本篇承接 [[01 什么是 AI Agent|什么是 AI Agent]] 的自主性谱系,下接 [[03 Agent 核心循环|Agent 核心循环]](agent 那一端的运行机制)。理解这条边界,是读懂后面所有架构([[09 ReAct|ReAct]]、[[10 Plan-and-Execute|Plan-and-Execute]]、[[22 多智能体系统|多智能体系统]])的前提——它们全是 agent 一侧的不同填法。

## 关键事实速记

- 唯一判据:**「下一步做什么」由代码决定(workflow)还是模型决定(agent)**。
- workflow 用可预测/可靠/便宜,换不了灵活;agent 反之。
- 工程默认:**能不上 agent 就不上**;先单次调用 → 再 workflow → 实在不行才 agent。
- 边界是连续谱系,真实系统常是 workflow 与 agent 互相嵌套的混合体。
- 生产最常见形态:**workflow 主干 + 有护栏的 agent 子节点**,把不确定性关进小盒子。
