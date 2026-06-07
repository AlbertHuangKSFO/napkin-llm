**AI Agent** 是以 LLM 为「大脑」、能感知环境、自主推理、调用工具采取行动、并根据反馈不断迭代直到完成任务的系统——它不是回答一句话,而是「自己想办法把活干完」。

要分清它和 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]]:狭义的 agent 指模型**自己动态决定**流程与工具用法的那一端;本篇先讲清楚 agent 这个整体概念长什么样、由什么组成、和它的近亲有何不同。

## 本质:一个感知-推理-行动的闭环

把 agent 拆开,只有四个部件 + 一根回路:

- **大脑(LLM)**:核心决策器。每一步读当前上下文,输出「我接下来要做什么」——可能是一段思考(thought),也可能是一次工具调用(tool call)。
- **工具(Tools)**:大脑伸向世界的手。搜索、跑代码、读写文件、调 API……见 [[15 Function Calling 工具调用|Function Calling 工具调用]] 与 [[16 工具设计与工具层|工具设计与工具层]]。
- **记忆(Memory)**:短期=当前任务的上下文窗口;长期=跨会话的经验/事实存储。见 [[19 Agent 记忆系统|Agent 记忆系统]]。
- **环境(Environment)**:动作真正落地的地方(文件系统、浏览器、数据库、用户),也是反馈(observation)的来源。

闭环是关键:**感知环境 → 推理下一步 → 执行动作 → 拿到反馈 → 再感知**,循环往复。正是这根回路把「会说话的模型」变成「会干活的系统」。这根回路的机制细节见 [[03 Agent 核心循环|Agent 核心循环]]。

![[什么是 AI Agent.png]]

## 机制:一次完整的 agent 运行(分步)

1. **接任务**:用户给一个高层目标(「帮我把这个仓库的测试修绿」),目标开放、步骤未知。
2. **观察**:模型读到当前状态(报错日志、文件树)。
3. **推理 + 决策**:模型产出思考 + 决定调哪个工具(`run_tests`、`read_file`、`edit_file`)。
4. **行动**:runtime(即 [[23 Agent Harness 概览|Agent Harness 概览]])执行工具,得到结果。
5. **回灌反馈**:把工具结果(observation)塞回上下文,模型据此修正下一步。
6. **判停**:任务达成 / 超步数 / 超预算 / 触发人工审核时停下。

跟单次问答最大的差别:agent **不预先知道要做几步**,步数由任务和反馈动态决定。

## 来源:这个词怎么定下来的

- 「agent」概念源自经典 AI(Russell & Norvig《AIMA》:agent = 感知环境并行动的实体)。
- LLM 时代的现代用法被 **Anthropic《Building Effective Agents》(2024-12)** 收紧:agent 特指「LLM 动态主导自身流程与工具使用」的系统,与 workflow 区分开。
- 工程范式上游是 **ReAct(Yao et al., 2022)**,首次把「推理 + 行动」交错进同一个 LLM 回路,见 [[09 ReAct|ReAct]]。

## 最小可跑骨架(伪代码)

```python
def agent(goal, tools, llm, max_steps=20):
    messages = [{"role": "user", "content": goal}]
    for _ in range(max_steps):
        resp = llm(messages, tools=tools)          # 大脑:决定下一步
        if resp.tool_calls:                         # 决策=调工具
            for call in resp.tool_calls:
                obs = tools[call.name](**call.args) # 行动 + 拿反馈
                messages.append(tool_result(call, obs))  # 回灌
        else:
            return resp.content                     # 没有工具调用 = 收尾,任务完成
    return "达到最大步数,未完成"
```

这就是一个能跑的最小 agent。把 `llm`/`tools` 换成真实实现即可——剩下的所有花样(规划、反思、多智能体)都是在这个回路上加料。

## 对比:agent vs chatbot vs RPA

| 维度 | Chatbot | RPA(机器人流程自动化) | AI Agent |
|---|---|---|---|
| 核心 | 单轮/多轮对话 | 硬编码脚本 | LLM 大脑 + 回路 |
| 流程 | 无流程 | 人写死的固定流程 | 模型动态决定 |
| 工具 | 一般无 | 固定的录制动作 | 动态选择调用 |
| 应对变化 | 不能 | 流程变就失效 | 据反馈自适应 |
| 适用 | 答疑 | 规则明确、稳定重复 | 任务开放、步骤不可预定 |

![[什么是 AI Agent-自主性谱系.png]]

注意:RPA 和 chatbot 不是「弱 agent」,而是**另一类东西**——它们没有「据反馈改下一步」的回路。自主性是一条谱系(见上图),从单次调用 → workflow → 受限 agent → 全自主 agent 渐变。

## 何时真的需要 agent

✅ **该上 agent**:任务开放、步骤无法预先列全、需要多轮试错与工具交互(写代码改 bug、深度调研、操作浏览器完成订单)。

❌ **别上 agent**:流程固定可预定、对成本/延迟/可靠性敏感。这时用 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里的固定编排更稳更便宜。

> 原则(Anthropic):**先用最简方案,能不上 agent 就不上**。agent 用灵活换可预测性,代价是更高的 token 成本、更慢、更难调试。

## 坑与反模式

- **为 agent 而 agent**:简单分类/抽取任务硬套自主回路,徒增成本和不确定性。
- **没有停机条件**:回路跑飞、烧光预算——必须设最大步数/token 预算/熔断,见 [[03 Agent 核心循环|Agent 核心循环]]。
- **工具设计糟糕**:工具描述含糊、返回噪声大,模型决策就崩;工具是 agent 的上限,见 [[16 工具设计与工具层|工具设计与工具层]]。
- **上下文塞爆**:长程任务里把所有历史堆进窗口,信噪比下降,见 [[20 上下文工程|上下文工程]]。

## 工业界实践

真实生产里几乎没人从零手写那个 while 循环——而是站在某个 **agent 框架 / harness** 上,把精力花在工具、提示、护栏上。

**主流框架与服务(给名字 + 定位):**

- **LangGraph**(LangChain 出品):把 agent 建模成**有状态的图**(节点 = 步骤,边 = 转移),2025-05 发布 1.0 GA、2025-12 出 v1.1,被 Uber、LinkedIn、Klarna 等近 400 家公司用于生产。卖点是**持久化状态(checkpoint)+ human-in-the-loop 中断 + LangSmith 可观测**。适合需要可控流程、可回放的复杂 agent。
- **OpenAI Agents SDK**(2025-03-11 发布,是实验性 Swarm 的生产化继任):轻量 `Agent + handoff + guardrail` 抽象,底层跑 **Responses API**(2025-03 发布,合并了 Chat Completions 的简洁与 Assistants 的工具能力)。注意:旧的 **Assistants API 已于 2025-08-26 宣布弃用,2026-08-26 下线**,新项目直接用 Responses/Agents SDK。
- **Claude Agent SDK**(原名 Claude Code SDK,2025-09 改名):把 Claude Code 背后的 **agent harness** 抽出来给你嵌进自己的应用——`Client SDK` 要你自己写 tool loop,`Agent SDK` 则让 Claude **自主跑完循环、自带工具执行与上下文管理**,见 [[23 Agent Harness 概览|Agent Harness 概览]]。
- **LlamaIndex**(偏数据/检索型 agent)、**AutoGen**(微软,多 agent 对话编排)、**CrewAI**(角色化多 agent 协作)、**PydanticAI / Mastra / Vercel AI SDK**(类型安全、前端友好)各占一块生态,详细横评见 [[37 Agent 框架对比|Agent 框架对比]] 与 [[39 Agent 开源生态全景|Agent 开源生态全景]]。

**典型架构与数据流:** 用户目标 → harness 组装上下文(system prompt + 工具定义 + 历史)→ 模型出 tool_call → 沙箱内执行工具 → observation 截断后回灌 → 循环,直到停机。生产里这条链路外面还会包一层**编排/持久化**(见下「durable execution」)、一层**可观测**(trace 每一步)、一层**护栏**(沙箱、审批、预算)。

**规模化与成本/延迟权衡:**
- agent 是「多轮 LLM 调用」,成本和延迟天然比单次问答高一个数量级;生产里靠**模型分级**(规划用大模型、批量子步用小模型)、**prompt 缓存**、**并行工具调用**压成本,系统性方法见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
- 长程任务要做**持久化执行(durable execution)**:用 Temporal / Restate / DBOS / LangGraph checkpointer 把每步落盘,崩溃后从断点恢复而非从头重跑,见 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]。

**可观测与运维:** 生产 agent 必须能**回放轨迹**(每步的 thought / tool_call / observation),否则线上出错无从排查。主流方案 LangSmith、LangFuse、Arize Phoenix、OpenTelemetry GenAI 语义约定,详见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**真实踩坑与最佳实践:**
- **先别上框架**:Anthropic 明确建议很多场景直接用裸 LLM API 几行代码就够;框架的抽象有时反而挡住你看清 prompt 与上下文。
- **工具是上限**:再强的循环,工具描述含糊、返回噪声大,agent 就崩,见 [[16 工具设计与工具层|工具设计与工具层]]。
- **上下文是战场**:长程任务的成败八成在上下文管理(截断、压缩、外置记忆),见 [[20 上下文工程|上下文工程]]、[[21 上下文压缩与卸载|上下文压缩与卸载]]。

```python
# 用框架 vs 裸写:两种心智
# 裸写(Anthropic 推荐的起点):你掌控每一行
from anthropic import Anthropic
client = Anthropic()
resp = client.messages.create(model="claude-...", tools=TOOLS, messages=msgs)

# 框架(LangGraph):你描述图,框架管状态/持久化/可观测
from langgraph.graph import StateGraph
g = StateGraph(AgentState)
g.add_node("agent", call_model)
g.add_node("tools", run_tools)
g.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
app = g.compile(checkpointer=checkpointer)   # ← 持久化,可断点续跑
```

## 面试高频

**Q1:一句话定义 AI Agent,它和 chatbot / workflow 的本质区别?**
标准答:Agent = 以 LLM 为大脑、能调工具、据环境反馈**自主迭代**直到完成开放任务的系统。与 chatbot 区别在**有「据反馈改下一步」的闭环**;与 workflow 区别在**「下一步做什么」由模型动态决定而非代码写死**(见 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]])。
- 追问:**那 RPA 算弱 agent 吗?** 不算——RPA 是死脚本,没有据反馈自适应的回路,是另一类东西。
- 陷阱:别把「多步 LLM 调用」就叫 agent;[[04 Prompt Chaining|Prompt Chaining]] 也多步,但流程写死,是 workflow。判据永远是**控制权归谁**。

**Q2:Agent 由哪几部分组成?**
标准答:大脑(LLM)、工具、记忆(短期=上下文窗口/长期=外置存储)、环境(动作落地 + 反馈来源),加一根 `感知→推理→行动→反馈` 的闭环。
- 追问:**记忆为什么单独拎出来?** 因为模型本身无状态,跨轮/跨会话的「连续」全靠外部把历史累积喂回去(机制见 [[03 Agent 核心循环|Agent 核心循环]])。

**Q3:什么时候该上 agent,什么时候不该?**
标准答:任务**开放、步骤无法预先列全、需多轮试错**才上 agent;流程固定、对成本/延迟/可靠性敏感就用 workflow 或单次调用。Anthropic 原则:**能不上 agent 就不上**。
- 陷阱:面试官常引诱你「什么都用 agent」,要主动说出 agent 的代价(贵、慢、难调、不稳)。

**Q4:Agent 为什么会「跑飞」,怎么防?**
标准答:循环没有出口 → 烧光预算。必配**停机条件**:max_steps、token 预算、人工审核点、报错熔断,见 [[03 Agent 核心循环|Agent 核心循环]]。

**Q5(进阶):自主性谱系是什么?**
标准答:从「单次调用 → workflow → 受限 agent(工具集小/步数封顶/关键动作要人确认)→ 全自主 agent」连续渐变,不是开关。工程上大多数生产系统落在 workflow 与受限 agent 之间。

## 知识拓展

- **理论根**:「agent」概念出自经典 AI(Russell & Norvig《AIMA》:能感知环境并行动的实体);LLM 时代的现代收紧定义来自 **Anthropic《Building Effective Agents》(2024-12)**;工程范式上游是 **ReAct(Yao et al., 2022,arXiv 2210.03629)**,首次把推理与行动交错进同一回路,见 [[09 ReAct|ReAct]]。2026 年主流编码 agent(Claude Code、Cursor、Aider)内部仍跑 ReAct 式循环。
- **「augmented LLM」是地基**:Anthropic 把一切 agentic 系统的最小积木定义为**增强版 LLM**——LLM + 检索 + 工具 + 记忆。Agent 只是在这块积木上加了「自主编排」这层。
- **边界与反模式**:最常见反模式是**为 agent 而 agent**(简单分类/抽取硬套自主回路)和**无护栏 agent**(没停机条件 = 烧钱机器)。判断锚点永远是那句:**步骤能否预先定全?**
- **深层联系**:本篇是整个 Agent 域的总入口——核心机制展开在 [[03 Agent 核心循环|Agent 核心循环]];与 workflow 的切分在 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]];多 agent 协作扩展见 [[22 多智能体系统|多智能体系统]];把 agent 用于具体场景(编码/研究/操作浏览器)见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]、[[29 Deep Research Agent|Deep Research Agent]]、[[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]]。
- **前沿方向**:用 RL 直接训练 agent 行为(而非只靠 prompt)是 2024–2026 热点,见 [[32 Agentic RL 与训练|Agentic RL 与训练]];让 agent 在长程任务中自我改进见 [[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进(Ralph loop)]]。

## 关键事实速记

- Agent = LLM 大脑 + 工具 + 记忆 + 环境回路;灵魂是「据反馈迭代」的闭环。
- 与 chatbot(无回路)、RPA(死脚本)的本质区别在于**流程由模型动态决定**。
- 自主性是谱系,不是开关;工程上多数任务落在 workflow 与受限 agent 之间。
- 判定是否用 agent 的关键问题:**步骤能否预先定全?** 能则 workflow,不能才 agent。
- 生产框架按定位记:LangGraph(图+持久化)、OpenAI Agents SDK(轻量+Responses API)、Claude Agent SDK(harness 抽取)、LlamaIndex/AutoGen/CrewAI(数据/多 agent)。
