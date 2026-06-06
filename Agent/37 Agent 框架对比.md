[[37 Agent 框架对比|Agent 框架对比]] 的本质是:主流框架并不是「谁更好」,而是各自在**「控制粒度 ↔ 抽象层级」这条一维谱**上占了一段——你写得越多、能改的越多(高控制低抽象),框架就越像一套积木;你写得越少、开箱即用越多(低控制高抽象),框架就越像一个黑盒。选框架=选你想停在这条谱的哪一段。

## 本质:一条谱,不是一张排行榜
所有「LangGraph 和 CrewAI 哪个好」的争论都问错了问题。正确的问法是:**这个任务,你需要多细的控制?你愿意为此写多少样板代码?**

- **高控制 / 低抽象**:框架把「[[03 Agent 核心循环|Agent 核心循环]]」「状态」「分支」全摊开给你,你像搭电路一样显式连线。优点是每一步可见、可改、可调试;代价是样板多、上手慢。代表:**LangGraph**。
- **低控制 / 高抽象**:框架把循环、协作、停机都内建,你只声明「有哪些角色 / 工具 / 目标」。优点是几行起步、协作开箱即用;代价是出错时难定位、想改中间一步要和框架的抽象搏斗。代表:**CrewAI**。
- 中间地带:**AutoGen**(对话式多体)、**OpenAI Agents SDK**(轻量声明式)、**Claude Agent SDK**(把 [[23 Agent Harness 概览|Agent Harness 概览]] 的 harness 能力 SDK 化)。

记住这条谱,后面所有对比都只是它的展开。

![[Agent 框架对比.svg]]

## 五个框架逐个拆

### LangGraph(高控 / 低抽:把 agent 建成图)
**心智模型**:agent = 一张**有向图 / 状态机**。节点是函数(常是一次 LLM 调用或一次工具执行),边是控制流,**状态(State)显式地在节点间流动**。它是 LangChain 团队为「LangChain 的链太线性、装不下循环和分支」而做的下层引擎。

- **控制粒度**:最细。循环、条件分支、人审中断(`interrupt`)、检查点(checkpoint)持久化、断点续跑、时间旅行回放——全是一等公民。
- **为什么独特**:它不替你决定「下一步做什么」,你在图里写死或写成条件边。所以它能精确实现 [[10 Plan-and-Execute|Plan-and-Execute]]、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]]、[[07 Orchestrator-Workers|Orchestrator-Workers]] 这些**有明确骨架**的模式,而不会「失控乱跑」。
- **状态管理**:一等公民。`StateGraph` 定义一个带 schema 的共享状态(常用 reducer 合并),天然支持持久化与恢复——这是它做长程、可中断 agent 的杀手锏。
- **多体支持**:能,但要你自己用「子图 / 多节点」搭,不是开箱的「角色」抽象。
- **学习曲线**:陡。要理解图、状态、reducer、检查点这套概念。
- **适合**:复杂但**确定流程**的生产系统;需要人审、需要断点续跑、需要精确控制每一步的场景。

### CrewAI(低控 / 高抽:角色 + 任务的多体协作)
**心智模型**:你**招一个团队(Crew)**,每个成员是一个有 `role / goal / backstory` 的 Agent,你把工作拆成 `Task` 派给他们,框架负责把活按顺序(sequential)或层级(hierarchical)跑完。

- **控制粒度**:粗。你描述「谁、目标是什么、用什么工具」,**怎么循环、怎么交接由框架兜底**。
- **为什么独特**:抽象贴近人类组织直觉——「研究员→作家→审校」。Demo 友好,几十行就能跑出一个多角色流水线。
- **多体支持**:这是它的主场。sequential / hierarchical(有一个 manager agent 派活,本质是 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的封装)两种内建协作。
- **状态管理**:弱。靠 Task 之间传 output、靠 context 串联,没有 LangGraph 那种显式可持久化状态机。
- **学习曲线**:最平。
- **适合**:快速原型、流程相对固定的多角色内容生产;**不适合**需要精细控制、复杂分支或强可靠性的生产核心链路。

### AutoGen(中间:微软,对话式多 agent)
**心智模型**:多个 agent **通过「对话」协作**。最经典的是 `AssistantAgent` ↔ `UserProxyAgent` 两者来回发消息,`UserProxy` 还能执行代码;多体则用 `GroupChat` + `GroupChatManager` 决定「下一个谁说话」。

- **控制粒度**:中。编排的核心抽象是**「发言权调度」(谁下一个说话、何时终止)**,而不是图或循环。
- **为什么独特**:把多 agent 协作建模成**会话**,天然适合「写代码—执行—看报错—改」这类需要来回的任务(UserProxy 能跑代码并回灌结果,等于内建了一个执行回路)。新版 AutoGen(0.4+)做了事件驱动的 actor 架构重写,更工程化。
- **多体支持**:强,且是「对话式」这一独特范式;但「靠对话收敛」有时不如显式图可控。
- **状态管理**:以消息历史为主。
- **学习曲线**:中。
- **适合**:需要 agent 间多轮协商、需要内建代码执行回路的研究 / 编码任务。

### OpenAI Agents SDK(中间偏高抽:轻量、handoffs、guardrails)
**心智模型**:声明一个 `Agent(instructions, tools, handoffs)`,调 `Runner.run()`,框架替你跑 [[03 Agent 核心循环|Agent 核心循环]]。它是 OpenAI 把早期实验项目 Swarm 产品化的结果。

- **控制粒度**:中。你声明「是什么」,循环、工具调度、停机交给框架——但比 CrewAI 更贴近原始的「单 agent + 工具」,不强加「角色 / Crew」这层组织抽象。
- **三个核心原语**:
  - **handoffs**:一个 agent 可以把控制权**移交**给另一个 agent(本质是把「换 agent」做成一种特殊工具调用)——这是它做多体的方式,轻量、显式。
  - **guardrails**:在输入 / 输出上挂校验,不合规就提前中断,是内建的安全/质量闸。
  - **sessions / tracing**:内建会话记忆与追踪。
- **多体支持**:靠 handoffs,去中心、链式移交,不是中心编排。
- **学习曲线**:平缓。
- **适合**:想要轻量、少魔法、和 OpenAI 生态贴合的生产 agent;尤其是「分诊 → 移交专家 agent」这类 [[05 Routing|Routing]] + handoff 形态。

### Claude Agent SDK(把 harness SDK 化)
**心智模型**:Anthropic 把 **Claude Code 背后的 [[23 Agent Harness 概览|Agent Harness 概览]]**(文件读写、bash、[[24 Agentic Search：grep vs 向量检索|代码搜索]]、[[17 MCP 模型上下文协议|MCP 模型上下文协议]] 接入、子代理、权限沙箱、上下文压缩)抽出来,做成可编程 SDK。它原名 Claude Code SDK,2025 年更名以示「不止写代码」。

- **控制粒度**:中。你不必自己写循环——**循环、工具编排、上下文管理(自动压缩长对话)是 harness 内建的**;你控制的是工具集、权限、系统提示、子代理定义。这正是 [[23 Agent Harness 概览|Agent Harness 概览]] 里说的 **HaaS(Harness as a Service)**。
- **为什么独特**:它自带一套**经过 Claude Code 实战打磨的工具与上下文工程**(参见 [[20 上下文工程|上下文工程]])。你不是从零搭 agent,而是站在一个成熟 harness 上。内建 **[[26 Sub-agents 与 Agent Teams|subagents]]**、**[[25 Agent Skills(SKILL.md)|Agent Skills(SKILL.md)]]**、MCP、hooks。
- **多体支持**:内建 subagents(主 agent 派子 agent,本质是 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的工程化加强)。
- **状态管理**:会话 + 自动上下文压缩;长任务友好。
- **学习曲线**:中。概念不多,但要理解 harness/权限/MCP 模型。
- **适合**:要直接复用「Claude Code 级」工程能力(尤其代码、文件、终端类自主任务)的场景;以及想要 Anthropic 官方上下文工程默认值的团队。

## 核心对比表(本篇的心脏)
| 维度 | LangGraph | CrewAI | AutoGen | OpenAI Agents SDK | Claude Agent SDK |
|---|---|---|---|---|---|
| 抽象层级 | 低(图/状态机) | 高(角色/任务) | 中(对话/会话) | 中(声明式 agent) | 中高(harness 内建) |
| 控制粒度 | **最细** | 粗 | 中(发言权调度) | 中 | 中(工具/权限可控,循环内建) |
| 核心抽象 | 节点+边+State | Agent role + Task + Crew | 多 agent 对话 + GroupChat | Agent + tools + **handoffs** | harness + tools + subagents |
| 多体支持 | 自己搭子图 | **内建**(seq/hier) | **内建**(对话式) | handoffs(链式移交) | 内建 subagents |
| 可观测性 | LangSmith 深度集成 | 基础(可接外部) | 内建 logging,可接 | 内建 tracing | 内建,可接 OTel |
| 状态管理 | **一等公民**(可持久化/续跑) | 弱(Task 传 output) | 消息历史 | sessions | 会话+自动压缩 |
| 人审/中断 | **原生** interrupt/checkpoint | 弱 | 靠 UserProxy | guardrails | hooks/权限 |
| 学习曲线 | 陡 | **最平** | 中 | 平缓 | 中 |
| 厂商 | LangChain | CrewAI Inc. | 微软 | OpenAI | Anthropic |
| 典型场景 | 复杂确定流程、强可靠生产 | 快速多角色原型 | 多轮协商/代码执行 | 轻量生产、分诊移交 | 复用 Claude Code 工程能力 |

![[Agent 框架对比-代码风格对照.svg]]

## 最小代码风格对照(同一任务的四种写法)
同样是「先研究、再写作」,看抽象层级如何决定你写什么。

**LangGraph(画图,显式状态与边):**
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class S(TypedDict):
    topic: str; notes: str; draft: str

g = StateGraph(S)
g.add_node("research", lambda s: {"notes": llm(f"研究:{s['topic']}")})
g.add_node("write",    lambda s: {"draft": llm(f"基于笔记写作:{s['notes']}")})
g.add_edge("research", "write")
g.add_edge("write", END)
g.set_entry_point("research")
app = g.compile()                 # 你掌控每一条边
app.invoke({"topic": "agent 框架"})
```

**CrewAI(招角色,派任务):**
```python
from crewai import Agent, Task, Crew
researcher = Agent(role="研究员", goal="收集资料", backstory="资深分析师")
writer     = Agent(role="作家",   goal="写成文章", backstory="科技作者")
t1 = Task(description="研究 agent 框架", agent=researcher)
t2 = Task(description="据资料成文",     agent=writer, context=[t1])
Crew(agents=[researcher, writer], tasks=[t1, t2]).kickoff()  # 循环框架兜底
```

**OpenAI Agents SDK(声明 agent + handoff):**
```python
from agents import Agent, Runner
writer = Agent(name="Writer", instructions="把研究笔记写成文章")
researcher = Agent(name="Researcher", instructions="研究主题后移交给 Writer",
                   handoffs=[writer])      # handoff = 把控制权交出去
Runner.run_sync(researcher, "研究 agent 框架并成文")
```

**Claude Agent SDK(配工具/子代理,循环交给 harness):**
```python
from claude_agent_sdk import query, ClaudeAgentOptions
# harness 自带文件/bash/搜索工具与上下文压缩,你只声明意图与权限
opts = ClaudeAgentOptions(system_prompt="研究后写成 markdown", allowed_tools=["Read","Write","WebSearch"])
async for msg in query(prompt="研究 agent 框架并写成 report.md", options=opts):
    print(msg)
```
看出差别没有:**LangGraph 你写「流程」,CrewAI 你写「组织」,SDK 你写「意图」**。越往下抽象越高、代码越少、可控越少。

## 何时选哪个(决策建议)
- **流程明确、要强可靠 / 人审 / 断点续跑** → **LangGraph**。它是唯一把「状态可持久化 + 中断恢复」做成一等公民的;生产核心链路首选。
- **快速验证一个多角色想法、Demo / 内容流水线** → **CrewAI**。最快出活,但别把它放进高可靠生产核心。
- **任务本身就是多 agent 来回协商、或需要内建代码执行回路** → **AutoGen**。
- **想要轻量、贴 OpenAI 生态、做「分诊→移交专家」** → **OpenAI Agents SDK**。handoffs + guardrails 的组合很顺手。
- **要直接吃到「Claude Code 级」的文件/终端/搜索/上下文工程能力**(尤其代码与自主任务)→ **Claude Agent SDK**。
- **一个朴素但常对的建议**:能用 [[02 Workflow 与 Agent 的边界|workflow]] 解决就别上 agent 框架;能用裸 [[15 Function Calling 工具调用|Function Calling 工具调用]] + 一个 while 循环解决,就别急着引框架。框架是为「复杂度真到了那一步」准备的。

## 坑
- **坑一:被抽象绑架**。CrewAI 这类高抽象框架,顺着它的范式很爽,一旦要做框架没预设的事(改中间一步、加奇怪分支),就要和抽象搏斗,常比裸写还累。
- **坑二:把 Demo 框架塞进生产**。很多框架 Demo 惊艳、生产翻车——错误处理、重试、可观测性、成本控制这些「无聊的工程」才是生产门槛,而它们恰恰是高抽象框架最弱的地方。
- **坑三:多体不等于更好**。多数任务一个 agent + 好工具就够了(见 [[22 多智能体系统|多智能体系统]] 的成本/协调代价讨论)。别因为框架支持多体就上多体。
- **坑四:可观测性是后悔药**。选型时容易忽略,出问题时最致命——见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。LangGraph+LangSmith、各 SDK 内建 tracing 的差距,要在选型期就算进去。
- **坑五:版本动荡**。这些框架都在高速迭代(AutoGen 大改过架构、OpenAI SDK 由 Swarm 演化、Claude SDK 改过名),教程极易过期,认准官方最新文档。

## 关键事实
- 一句话记忆:**LangGraph 写流程、CrewAI 写组织、AutoGen 写对话、OpenAI SDK 写意图+移交、Claude SDK 站在 harness 上写意图**。
- 这条谱不是「越右越先进」——**高抽象的代价永远是失控时的不可见**;成熟团队常选偏左,因为可调试性在生产里比上手快值钱。
- 框架的本质是把 [[03 Agent 核心循环|Agent 核心循环]]、[[07 Orchestrator-Workers|Orchestrator-Workers]]、[[05 Routing|Routing]] 等模式**封装成可复用的工程件**;理解了这些底层模式,框架只是它们的不同打包方式。
- Claude Agent SDK 是 [[23 Agent Harness 概览|Agent Harness 概览]] 所说 **HaaS** 的最直接体现:harness 本身成了产品。
- 选型不是一次性的:原型期可用 CrewAI 验证想法,生产期常重写到 LangGraph 或裸 SDK 以换可控与可观测——参见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 把这件事做成闭环。

## 工业界实践

**生产里框架不是「单选」,而是「分阶段组合」。** 2026 一个反复出现的企业模式:**用 CrewAI 做研究/综合阶段、把结构化 JSON 交给 LangGraph 做执行阶段**——前者搭得快、贴组织直觉,后者状态机稳、失败节点能优雅恢复。也就是说,本篇那条「谱」在真实系统里往往是**两段拼接**,而非停在一点。

**采用度与基准(2026 实测,供谈资,别背死):**
- **GitHub 热度**:CrewAI 约 31k star(从 2024 年 1 月 2.8k 暴涨 ~1000%,绝对增速最快);LangGraph 约 12.8k 但**企业采用更快**。按 Gartner,2026 Q1 在 1000+ 员工公司的生产架构文档里,**LangGraph 占 agent 框架引用的 34%**——星数不等于生产份额。
- **任务成功率基准**(第三方,口径各异):中等任务 LangGraph ~76% > CrewAI ~71% > AutoGen ~68%;复杂任务 LangGraph ~62% > CrewAI ~54%,差距来自 LangGraph 的图状态机**对失败节点的优雅处理**(可重试、可续跑)。
- **成本**:CrewAI 的多角色协作有 ~18% 的 token 开销(3-agent crew 比等价 LangGraph 实现多花 token)——多体的协调代价是真实的(呼应 [[22 多智能体系统|多智能体系统]])。

**部署侧的关键差异——LangGraph Platform(2026 已 GA)。** LangGraph 不只是库,还有配套的**持久化执行平台**:
- 内建 **task queue + 状态持久化 + 断点续跑**,把本篇说的「checkpoint 一等公民」做成了托管服务,长程/有状态 agent 不用自己搭 [[34 Agent 部署与持久化执行|持久化执行]]。
- 部署形态:**Cloud SaaS**(Plus $49/月单部署、Pro $99/月五部署)、**Hybrid**(SaaS 控制面 + 自托管数据面,Enterprise)、**完全自托管**(数据不出 VPC)。
- 成本陷阱:生产常驻有 **standby 计费(~$0.0036/分钟,空闲也算)**,高可用常驻服务这笔钱可能超过按节点执行的费用——选型时要把它算进 TCO。
- 开源库本身 MIT、永久免费,但自己跑要自备基础设施、调度、扩缩容。

**其余框架的生产现状:**
- **CrewAI**:除开源库外有商业 **AMP / Enterprise** 平台,补齐了部署、监控、连接器;Flows(事件驱动、更可控的低层 API)缓解了「纯 Crew 太黑盒」的批评。
- **AutoGen**:0.4+ 事件驱动 actor 架构重写后更工程化;微软同时推 **AutoGen + Semantic Kernel** 双线,企业版能力向 SK 收敛。
- **OpenAI Agents SDK**:2026 加了 **sandbox agents**(在沙箱里跑命令/改代码的长程任务),向「成品编码 agent」靠拢;吃 OpenAI 全家桶时 tracing/guardrails 最顺。
- **Claude Agent SDK**:复用 Claude Code 实战打磨的 harness(工具集、自动上下文压缩、subagents、Skills、hooks),代码/终端/文件类自主任务开箱即强。

**可观测必须同步选型(别等出事)。** LangGraph 配 LangSmith 最丝滑,但会**框架锁定**;追求中立则全栈走 **OpenTelemetry GenAI 语义约定** + Langfuse/Phoenix,埋点一次、后端随便换。这点在选型期就要算进去,详见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

## 面试高频

**Q1:LangGraph、CrewAI、AutoGen 怎么选?给一句话心智模型。**
**LangGraph 写流程(图/状态机,最细控制)、CrewAI 写组织(角色+任务,最高抽象)、AutoGen 写对话(发言权调度)**。再加两个:OpenAI Agents SDK 写「意图 + handoff 移交」、Claude Agent SDK「站在 harness 上写意图」。一维谱:控制粒度 ↔ 抽象层级,**不是排行榜**。
- *追问:这条谱越往高抽象越先进吗?* 不是。高抽象的代价永远是**失控时的不可见**;成熟团队常选偏左(LangGraph/裸 SDK),因为生产里可调试性比上手快值钱。
- *陷阱:「CrewAI 最简单,生产就用它?」* 不。CrewAI 状态管理弱、错误难定位,适合原型/demo;放进高可靠生产核心链路常翻车,典型路径是原型 CrewAI → 生产重写 LangGraph。

**Q2:LangGraph 的「杀手锏」是什么?为什么生产派偏爱它?**
**状态(State)是一等公民 + 可持久化 checkpoint + 中断恢复**。它能精确实现 [[10 Plan-and-Execute|Plan-and-Execute]]、[[13 Reflection 与 Reflexion|Reflection]]、[[07 Orchestrator-Workers|Orchestrator-Workers]] 这些有明确骨架的模式而不失控乱跑;失败节点能优雅重试、断点续跑、时间旅行回放。这正是长程、可中断、要人审(`interrupt`)的生产系统所需。
- *追问:那它的代价?* 学习曲线陡(图/状态/reducer/checkpoint 一套概念),样板多,上手慢。

**Q3:什么时候根本不该用 agent 框架?**
能用 [[02 Workflow 与 Agent 的边界|workflow]] 解决(固定的 [[05 Routing|Routing]]/[[04 Prompt Chaining|Prompt Chaining]])就别上 agent;能用裸 [[15 Function Calling 工具调用|Function Calling]] + 一个 while 循环解决就别急着引框架。框架是负债——每加一层多一份升级成本和锁定风险。多数任务一个 agent + 好工具就够,别因为框架支持多体就上多体。
- *陷阱:「多 agent 一定比单 agent 强?」* 不。多体有真实的 token 开销(~18%)和协调代价(见 [[22 多智能体系统|多智能体系统]]),多数任务单 agent 更省更稳。

**Q4:handoff、guardrail、subagent 各属于哪个框架,本质是什么?**
**handoff**(OpenAI Agents SDK)= 把「换 agent」做成一种特殊工具调用,链式去中心移交,做 [[05 Routing|分诊→专家]] 很顺;**guardrail**(OpenAI SDK)= 输入/输出校验闸,不合规提前中断;**subagent**(Claude Agent SDK)= 主 agent 派子 agent,本质是 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的工程化。三者底层都是同几个模式的不同封装。

## 知识拓展

**框架本质 = 把模式打包。** 理解了 [[03 Agent 核心循环|Agent 核心循环]]、[[07 Orchestrator-Workers|Orchestrator-Workers]]、[[05 Routing|Routing]]、[[10 Plan-and-Execute|Plan-and-Execute]] 这些底层模式,框架只是它们的不同打包方式——所以**先学模式,再学框架**,换框架的成本就低。Anthropic 的 "Building effective agents"(2024)反复强调:从最简单的 workflow 起步,只在复杂度真到了那步才上 agent。

**新兴方向(带年份):**
- **CodeAct 范式**(2024):让模型**写 Python 代码当动作**(而非填 JSON tool call),一段代码能组合多个工具、带控制流,表达力更强;`smolagents`(HuggingFace,核心约 1000 行)主打这个,见 [[39 Agent 开源生态全景|Agent 开源生态全景]] ① 层。
- **类型安全派**:`pydantic-ai` 把 Pydantic 类型校验带进 agent(FastAPI 式手感),重结构化输出、想看清每一步时强。
- **协议化解耦**:框架内部怎么写在收敛,框架之间靠 [[17 MCP 模型上下文协议|MCP]](Agent↔工具)和 [[30 A2A 协议|A2A]](Agent↔Agent)互通——长期看「用哪个框架」会比「框架间能不能互操作」次要。

**反模式:**
- **被抽象绑架**:顺着 CrewAI 的范式很爽,一旦要做它没预设的事(改中间一步、加奇怪分支),和抽象搏斗常比裸写还累。
- **把 demo 框架塞进生产**:错误处理、重试、可观测、成本控制这些「无聊工程」才是生产门槛,恰是高抽象框架最弱处。
- **版本动荡当事实**:这些框架高速迭代、还改名搬家(AutoGen→ag2、Swarm→OpenAI Agents SDK、Claude Code SDK→Claude Agent SDK),教程极易过期,认官方最新文档与 **pip 包名**(见 [[39 Agent 开源生态全景|Agent 开源生态全景]] 的「以 pip 名为锚」)。

**相关链接:** 完整生态分层与同层竞品见 [[39 Agent 开源生态全景|Agent 开源生态全景]](本篇是其 ① 编排层的代码级展开);harness 视角见 [[23 Agent Harness 概览|Agent Harness 概览]];部署持久化见 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]];选型闭环交给 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
