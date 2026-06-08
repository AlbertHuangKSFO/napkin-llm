**Agent 核心循环**就是一个 `observe → think → act → observe` 的回路(tool-use loop):模型读上下文、决定下一步、runtime 执行、把结果回灌,再循环——直到触发停机条件。

它是 [[01 什么是 AI Agent|什么是 AI Agent]] 里那根「感知-推理-行动」闭环的工程化骨架,也是 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 中 agent 那一端的运行机制。把这个循环讲透,后面所有架构([[09 ReAct|ReAct]]、[[10 Plan-and-Execute|Plan-and-Execute]]、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]])都只是在它上面加料。

## 本质:一个带停机条件的 while 循环

去掉所有花哨,agent 运行时只有一个循环 + 三个状态:

- **observe(观察)**:把当前上下文 + 上一步产生的 observation 一起读进来。
- **think(思考)**:模型据此产出一段思考(thought)并决定下一步——通常是一次工具调用(tool call),也可能直接收尾。
- **act(行动)**:runtime(即 [[23 Agent Harness 概览|Agent Harness 概览]])真正执行那次工具调用,得到结果。

结果作为新的 observation 回灌,进入下一轮。**没有 tool call = 任务完成,正常退出**;否则继续转,直到撞上某个停机条件。

![[Agent 核心循环.png]]

## 机制:一轮里到底发生了什么(分步)

把单轮拆到帧级,跑这个循环的不是模型自己,而是模型和 harness 一来一回:

1. **harness → 模型**:把累积上下文(目标 + 历史 + 最新 observation)和**工具定义**一起发给模型。
2. **模型推理**:模型内部决定「下一步该干嘛」。
3. **模型 → harness**:吐出 thought + 一个结构化的 `tool_call(name, args)`(机制见 [[15 Function Calling 工具调用|Function Calling 工具调用]])。
4. **harness 校验**:检查工具名/参数是否合法、是否超频、是否需人工批准。
5. **harness → 工具**:真正执行 `tool(args)`。
6. **工具 → harness**:返回结果或报错。
7. **harness 处理**:把结果**截断/格式化**成 observation(原始输出可能巨大,这步极关键,见 [[20 上下文工程|上下文工程]])。
8. **harness → 模型**:observation 回灌进上下文,**回到第 1 步**开始下一轮。

注意:**模型本身是无状态的**,它不记得上一轮——是 harness 把历史一路累积着喂回去,才形成「连续干活」的错觉。

![[Agent 核心循环-时序.png]]

## 停机条件:不设就是烧钱机器

循环必须有出口,否则会跑飞、烧光预算。常见五种:

| 停机条件 | 触发时机 | 性质 |
|---|---|---|
| 任务完成 | 模型不再发 tool call(或调用 `finish`) | 正常出口 |
| 超最大步数 | 轮数 ≥ max_steps | 熔断 |
| 超 token 预算 | 累计 token ≥ 预算 | 熔断 |
| 触发人工审核 | 命中敏感动作(删库/付款) | 暂停转人工 |
| 报错熔断 | 连续 N 次工具失败 | 防死循环 |

工程上**至少配 max_steps + token 预算**;涉及外部副作用(写文件、调真实 API)时再加人工审核点。

表格看不出拓扑——把五个出口画回 while 循环上,正常出口与熔断/暂停一目了然:

![[Agent核心循环-停机状态机.png]]

## 最小可跑骨架(伪代码)

```python
def run_loop(goal, tools, llm, max_steps=20, token_budget=100_000):
    ctx = [{"role": "user", "content": goal}]   # 累积上下文
    used = 0
    for step in range(max_steps):                # ← 停机①:最大步数
        resp = llm(ctx, tools=tools)             # observe + think
        used += resp.usage.total_tokens
        if used > token_budget:                  # ← 停机②:token 预算
            return "超预算,中止"
        if not resp.tool_calls:                  # ← 停机③:无 tool call = 完成
            return resp.content
        ctx.append(resp.message)
        for call in resp.tool_calls:             # act
            if needs_human_review(call):         # ← 停机④:人工审核
                return pause_for_human(call)
            try:
                obs = tools[call.name](**call.args)
            except Exception as e:               # ← 停机⑤:报错熔断(简化)
                obs = f"工具报错:{e}"
            ctx.append(tool_result(call, truncate(obs)))  # 回灌(注意截断)
    return "达到最大步数,未完成"
```

这就是一切 agent 的心脏。三态(observe/think/act)、回灌、五个出口——全在这里。

## 与 ReAct 的关系

[[09 ReAct|ReAct]] **不是另一个循环,而是这个循环的一种 prompt 实现**:它用「Thought → Action → Observation」的文本模板,引导模型在每轮里**先显式写出推理再行动**。

- 核心循环 = 抽象骨架(observe/think/act,与实现无关)。
- ReAct = 在「think」这步强制产出可见的 reasoning trace 的具体玩法。
- 现代 Function Calling 把 action 结构化成 JSON tool_call,本质仍是同一个循环;ReAct 的「思考」退化成模型的内部推理或 thinking 块。

换句话说:**ReAct 是循环的一种填法,循环是 ReAct 的上位概念。** 其它填法还有 [[10 Plan-and-Execute|Plan-and-Execute]](先规划整条路线再逐步执行)、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]](在循环里加自我批判)。

## 何时用 / 坑 / 反模式

✅ **该用循环**:任务需多步工具交互、步数不可预定——这正是 agent 的定义场景。

**坑:**
- **无停机条件**:最常见、最致命,务必设步数 + 预算上限。
- **observation 不截断**:工具吐回几万 token 原始日志直接灌进上下文,几轮就爆窗口、推高成本、淹没信号——必须在第 7 步做摘要/截断,见 [[20 上下文工程|上下文工程]]。
- **上下文只增不减**:长程任务里历史无限累积,信噪比崩塌——需要压缩/外置记忆,见 [[19 Agent 记忆系统|Agent 记忆系统]]。
- **错误不回灌**:工具报错时直接抛异常退出,而不是把错误信息当 observation 喂回去让模型自我纠正——白白浪费 agent 的纠错能力。

**反模式:**
- 用核心循环硬扛**流程可预定**的任务——那是 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里该用 workflow 的场景。
- 把「多步骤」误当「需要循环」:有些多步其实是固定链,用 [[04 Prompt Chaining|Prompt Chaining]] 更稳。

## 工业界实践

生产里那个「心脏」循环很少由你亲手写——它被框架封进 **agent harness / runtime**,你只配工具、提示、护栏。

**谁来跑这个循环:**

- **Claude Agent SDK**(原 Claude Code SDK,2025-09 改名):Anthropic 把 Claude Code 背后那套 **agent harness 抽出来**——`Agent SDK` 让 Claude **自主跑完循环、自带工具执行与上下文管理**;若用更底层的 `Client SDK`,则要你自己写 tool loop。同一套循环既驱动 Claude Code 也驱动你的自定义 agent,见 [[23 Agent Harness 概览|Agent Harness 概览]]。
- **OpenAI Agents SDK**:官方文档把这叫 **agent loop**——runner 反复「调模型 → 看输出 → 执行工具 → (或切换 agent / 返回结果)」,直到模型给出无 tool call 的终答。底层是 **Responses API**(2025-03 发布)。
- **LangGraph**:把循环显式建成图里的一条带条件边的环(`agent` 节点 ↔ `tools` 节点),好处是**每一轮都是一个 checkpoint**,可持久化、可中断、可回放。

**生产级循环比伪代码多出的关键件:**
- **持久化执行(durable execution)**:长程任务必须把每步落盘,崩溃后从断点恢复而非从头重跑。LangGraph 用 checkpointer 在应用层防故障;Temporal / Restate 在基础设施层防故障(机器宕机也能续);DBOS 把状态直接存进 Postgres,「天生 durable」无需显式打点。OpenAI Agents SDK 文档也指向 DBOS 集成做长程/人在环。深入见 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]。
- **observation 处理层**:工具原始输出常是几万 token 日志,必须在回灌前**截断/摘要**,否则几轮爆窗口。这是循环里最被低估的一步,机制见 [[20 上下文工程|上下文工程]]、[[21 上下文压缩与卸载|上下文压缩与卸载]]。
- **并行工具调用**:现代模型一轮能吐多个 tool_call,harness 并发执行能显著降延迟,见 [[06 Parallelization|Parallelization]]、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
- **可观测**:每一轮的 thought / tool_call / observation 都要可回放,否则线上 agent 出错无从查。LangSmith / LangFuse / Arize Phoenix,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

```python
# 生产循环 = 伪代码骨架 + 持久化 + 可观测 + observation 处理
# LangGraph 风格:每一轮自动成为可恢复 checkpoint
def agent_node(state):           # observe + think
    resp = llm.invoke(state["messages"], tools=TOOLS)
    return {"messages": [resp]}

def tools_node(state):           # act + 回灌(注意截断)
    results = run_tools_parallel(state["messages"][-1].tool_calls)
    return {"messages": [truncate(r) for r in results]}

graph.add_conditional_edges("agent", lambda s:
    "end" if not s["messages"][-1].tool_calls else "tools",
    {"tools": "tools", "end": END})
app = graph.compile(checkpointer=PostgresSaver(...))   # ← 崩溃可续跑
```

**真实踩坑与最佳实践:**
- **错误要回灌而非抛出**:工具报错时,把错误信息当 observation 喂回去让模型自我纠正——ReAct 的核心优势正是「靠对 observation 的推理从坏的 tool call 里恢复」。直接抛异常退出 = 白白浪费纠错能力。
- **停机条件是底线**:至少 max_steps + token 预算;有外部副作用(写文件/调真实 API/付款)再加人工审核点和沙箱。
- **上下文只增不减是慢性死亡**:长程任务历史无限累积,信噪比崩塌——需压缩/外置记忆,见 [[19 Agent 记忆系统|Agent 记忆系统]]。

## 面试高频

**Q1:画出 / 描述 agent 的核心循环。**
标准答:`observe(读上下文+上一步 observation)→ think(模型出 thought + tool_call)→ act(harness 执行工具)→ 把结果回灌 → 再 observe`,直到触发停机条件。**没有 tool call = 任务完成,正常退出。**
- 追问:**模型自己记得上一轮吗?** 不记得——**模型是无状态的**,是 harness 把历史一路累积着喂回去,才形成「连续干活」的错觉。这是高频追问,务必答对。

**Q2:一轮里 harness 和模型怎么分工?**
标准答:harness 组装上下文+工具定义发给模型 → 模型出 tool_call → harness **校验**(工具名/参数合法?超频?需审批?)→ 执行工具 → 把结果**截断/格式化**成 observation → 回灌。模型只负责「决定」,harness 负责「执行+管状态+管护栏」。

**Q3:Agent 为什么会跑飞?停机条件有哪些?**
标准答:循环没出口就烧光预算。五种停机:① 任务完成(无 tool call)② 超 max_steps ③ 超 token 预算 ④ 触发人工审核(敏感动作)⑤ 报错熔断(连续失败)。**至少配步数 + 预算。**
- 陷阱:有人只答「任务完成」就停——那是正常出口,熔断类的步数/预算才是防跑飞的关键。

**Q4:ReAct 和核心循环是什么关系?**
标准答:**ReAct 不是另一个循环,而是这个循环的一种 prompt 实现**——它用「Thought → Action → Observation」文本模板强制模型每轮先显式写推理再行动。核心循环是上位抽象,ReAct 是其一种填法;现代 Function Calling 把 action 结构化成 JSON tool_call,本质仍是同一循环,见 [[09 ReAct|ReAct]] 与 [[15 Function Calling 工具调用|Function Calling 工具调用]]。

**Q5(进阶):observation 不截断会怎样?为什么这是最致命的坑之一?**
标准答:工具吐回几万 token 原始日志直接灌进上下文 → 几轮就爆窗口、推高成本、淹没信号导致模型决策崩。必须在回灌前做摘要/截断,见 [[20 上下文工程|上下文工程]]。

## 知识拓展

- **「模型无状态」的深意**:这是理解一切 agent 工程的钥匙。所谓「记忆」「连续」「断点续跑」全是 harness 在管理外部状态——所以才有上下文工程([[20 上下文工程|上下文工程]])、记忆系统([[19 Agent 记忆系统|Agent 记忆系统]])、持久化执行([[34 Agent 部署与持久化执行|Agent 部署与持久化执行]])这些独立话题。
- **循环之上的「加料」全景**:同一根 `observe→think→act` 循环,加不同料就成不同架构——加显式推理 = [[09 ReAct|ReAct]];先规划整条路线再执行 = [[10 Plan-and-Execute|Plan-and-Execute]];规划与执行解耦省 token = [[11 ReWOO|ReWOO]];把步骤编译成 DAG 并行 = [[12 LLMCompiler|LLMCompiler]];加自我批判 = [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]];把循环展开成搜索树 = [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]]。
- **durable execution 是 2025–2026 的工程主线**:随着 agent 跑得越来越长(分钟到小时甚至天),「崩溃从断点恢复」从可选变成刚需。Temporal(基础设施级)、LangGraph checkpointer(应用级)、DBOS(Postgres 原生)各代表一种思路;生产常是「应用级 + 基础设施级」双层。
- **反模式**:① 无停机条件;② observation 不截断;③ 上下文只增不减;④ 错误不回灌(抛异常退出);⑤ 用核心循环硬扛流程可预定的任务(那是 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里该用 workflow 的场景)。
- **历史脉络**:循环的工程范式上游是 **ReAct(Yao et al., 2022,arXiv 2210.03629)**;它报告 ReAct 在知识密集问答上胜过纯 CoT 和纯 act 基线,优势主要来自「靠对 observation 推理从坏 tool call 中恢复」——这正是「错误回灌」最佳实践的理论依据。

## 关键事实速记

- Agent = 一个带停机条件的 `observe → think → act` while 循环;模型无状态,靠 harness 累积上下文。
- 一轮 = 模型出 tool_call → harness 执行 → observation 回灌;**「无 tool call」是正常出口**。
- 停机五件:任务完成 / 超步数 / 超预算 / 人工审核 / 报错熔断;至少配步数 + 预算。
- [[09 ReAct|ReAct]] 是这个循环的一种 prompt 实现,不是另一种东西。
- 两个最致命的工程坑:**不设停机条件** 和 **observation 不截断**。
- 生产循环比伪代码多三件:**持久化执行(断点续跑)+ observation 截断 + 可观测回放**。
