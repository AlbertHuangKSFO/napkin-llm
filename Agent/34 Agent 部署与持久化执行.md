[[34 Agent 部署与持久化执行|Agent 部署与持久化执行]] 的本质是:一个 agent 不是无状态的 HTTP 请求,而是一个**长跑、可中断、需要恢复的有状态工作流**——它可能跑几小时甚至几天,中途要等工具返回、等外部事件、等人审批,期间进程随时可能崩溃/被重启/被扩缩容杀掉。把它当无状态服务部署,一崩就丢光进度;正确做法是用 **durable execution(持久化执行)**:把**每一步的状态持久化、崩溃后重放恢复到崩前一刻、并保证步骤幂等**,让长跑 agent 可崩溃、可暂停、可恢复。

这是 [[03 Agent 核心循环|Agent 核心循环]] 从"能在 notebook 里跑通"到"能在生产里稳定长跑数天"之间那道最被低估的工程鸿沟,也是 [[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进(Ralph loop)]] 里"git commit 当检查点"这一穷人做法的工业级正解。

## 本质:agent 是有状态工作流,不是无状态请求

传统 Web 服务的心智是**无状态请求-响应**:请求进来,几百毫秒内算完返回,进程内存里的东西用完即弃。这套模型对 agent **彻底失效**,原因有三:

1. **长跑**:一个 [[29 Deep Research Agent|Deep Research Agent]] 或 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]] 任务动辄几十上百步、跑几分钟到几天,远超任何 HTTP 超时。
2. **有状态**:agent 的进度(已走到第几步、各步结果、计划、记忆)是**必须跨步保留**的核心资产,一旦丢失等于从头再来——而 LLM 调用又贵又慢,重跑代价极高。
3. **会等待**:agent 常要**长时间挂起**——等一个慢工具、等外部 webhook、等人审批一个高风险动作(Human-in-the-Loop)。这期间若让进程空占内存死等,既浪费又脆弱(一崩就全没)。

把 agent 塞进无状态 serverless / 普通 HTTP handler 的后果:**进程一死,内存里的全部进度归零,从头重跑**——重复烧 token、重复执行有副作用的操作(可能发两次邮件、下两次单)、用户体验崩坏。

**Durable execution** 就是为这类问题发明的范式。一句话定义:**让一个函数能在崩溃后,从它上次离开的地方精确恢复继续执行,就像崩溃从未发生过。** 它把"程序执行的进度"本身变成持久化、可恢复的资产。2025 年起,agentic AI、长跑 saga、事件驱动任务都收敛到了"一个崩溃后能精确续跑的函数"这同一个原语上。

## 机制:事件溯源 / 重放 + 持久状态 + 中断恢复 + 幂等

![[Agent 部署与持久化执行.svg]]

durable execution 的核心是**确定性重放(deterministic replay)+ 持久日志(event log)**,工作流分四个机制:

**① 持久日志 / 检查点**:agent 的执行被拆成一连串**步骤(step / activity)**。每完成一步,就把"这一步的输入 + 返回结果"**持久化写入一个日志**(通常落 Postgres 或专用存储)。step 之间的状态也随之落盘。

**② 崩溃后重放恢复(replay)**:进程崩溃后,一个新进程拉起,**不从零开始,而是重新执行工作流代码**;但走到每个已记录的 step 时,引擎**不真的再调一遍 LLM / 工具,而是直接从日志里取出上次的返回值喂回去**——于是代码被"快进"到崩溃前那一刻,只在第一个**没有日志记录的 step**(崩溃中断处)才真正重新执行。效果:跑了三天的 agent,崩了之后几秒内恢复到崩前状态续跑,**已花的 LLM/工具调用一分钱不重复花**。

**③ 持久状态 + 中断 / 恢复**:agent 要等人审批或外部事件时,引擎把**当前位置 + 变量 + "在等什么"**持久化,然后**释放执行资源、挂起**;worker 本身保持无状态,等待期间不空占内存。等审批/事件到达,调度器**唤醒**工作流、拉一个新 worker、从检查点 **resume** 继续。这就是 Human-in-the-Loop 审批、长时等待、定时唤醒在生产 agent 里的落地方式。

![[Agent 部署与持久化执行-中断恢复架构.svg]]

**④ 幂等(idempotency)**:重放的前提是**有副作用的步骤不能被做两遍**。崩溃可能发生在"工具已执行、但结果还没写进日志"的窗口里,恢复时该 step 会被重试。因此工具调用要么**天然幂等**(同样输入多次执行结果一致),要么靠**幂等键 / 日志去重**保证"恰好一次"的语义。这是 durable execution 区别于"简单 retry"的关键——它给的是 **exactly-once 效果**,而非 at-least-once。

把这四件事合起来,就得到一个**可崩溃、可暂停、可恢复、不重复副作用**的长跑 agent 运行时。

## 来源 / 技术谱系

durable execution 不是为 agent 发明的,而是分布式系统里"可靠工作流"的成熟范式(其思想可追溯到 workflow engine 与 event sourcing),2025–2026 因 agent 长跑需求而第二次走红。主流玩家:

- **Temporal**(`temporalio`):定义了这个品类的标杆,提供 **exactly-once 语义**和可无限期运行的工作流,在 Netflix、Uber 等大规模生产环境身经百战,一致性保证最强;其 Python SDK 已被用来构建"能扛崩溃、自动重试 LLM 调用、跑好几天"的 agent,并有与 OpenAI Agents SDK 的集成。
- **DBOS**(`dbos-transact-py`):**最轻量**的一派——直接构建在 **Postgres** 之上,给普通函数加几个装饰器即可获得持久工作流与队列,**无需单独运维一个 workflow 编排器**,运维足迹最小;提供 `DBOSAgent` 包装器把任意 Pydantic AI agent 变持久。
- **Restate**(`restate`):比 Temporal 模型更简单、更易运维的 durable execution。
- **Inngest**:事件驱动工作流,平台层负责重试与可观测;原生 TS,也支持 Python。
- **LangGraph checkpointer**(`langgraph`):把一次 agent 运行当作**有状态图执行**而非普通函数调用,每次状态转移自动 checkpoint(可插拔 Memory / SQLite / **Postgres** saver),让暂停/恢复、时间旅行调试、多实例水平扩展成为一等特性。

## 机制对照伪码:从"会丢状态"到"可恢复"

最朴素的 agent 循环——崩溃即全丢:

```python
def agent_run(goal):
    plan = llm_plan(goal)          # 贵
    results = []
    for step in plan:              # 中途崩溃 → plan、results 全在内存,全没
        results.append(call_tool(step))   # 可能重复执行副作用
    return llm_summarize(results)
```

用 durable execution(以 Temporal 风格伪码示意)后,每步成为可重放的 activity:

```python
from temporalio import workflow, activity

@activity.defn
async def llm_plan(goal: str) -> list[str]: ...      # 结果会被持久化
@activity.defn
async def call_tool(step: str) -> str: ...           # 需幂等;结果入日志

@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, goal: str) -> str:
        plan = await workflow.execute_activity(       # ← 重放时:有日志则直接取结果,不重调
            llm_plan, goal, start_to_close_timeout=TIMEOUT)
        results = []
        for step in plan:
            r = await workflow.execute_activity(       # 崩溃恢复后:已完成的 step 跳过
                call_tool, step, start_to_close_timeout=TIMEOUT)
            results.append(r)
        # 人审中断点:挂起,持久化,等 signal 回来再续(等待期间不占 worker)
        approved = await workflow.wait_condition(lambda: self._approved)
        return await workflow.execute_activity(llm_summarize, results, ...)

    @workflow.signal
    def approve(self):                                # 外部/人触发:唤醒并 resume
        self._approved = True
```

对应的 LangGraph 风格(checkpointer + interrupt)——同一思想的另一种 API:

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(DB_URL) as cp:
    cp.setup()                                   # 首次用需建表
    graph = build_agent_graph().compile(checkpointer=cp)
    cfg = {"configurable": {"thread_id": "task-42"}}   # 线程级寻址,断点续传
    # 每次状态转移自动写检查点;进程崩了换台机器、用同一 thread_id 即从最近检查点续
    graph.invoke({"goal": goal}, cfg)
    # 命中 interrupt() → 暂停;人审通过后:
    graph.invoke(Command(resume="approved"), cfg)      # 从中断点 resume
```

要点:① **每个 activity / 节点的结果被持久化**,重放只快进不重算;② **中断点**把控制权交还人/外部并挂起,worker 等待期间无状态、不占资源;③ **幂等** + start_to_close_timeout 处理"做了一半崩了"的窗口;④ 用 **thread_id / workflow_id** 寻址一个长跑实例,换机器也能续。

## 对比:无状态 serverless vs durable execution

| 维度 | 无状态 serverless / 普通 HTTP | **durable execution** |
|---|---|---|
| 进度持久化 | 内存里,**进程死即丢** | 每步落持久日志,**可恢复** |
| 崩溃恢复 | 从头重跑(重烧 token、重复副作用) | 从崩前一刻**重放续跑**,不重复已花的调用 |
| 长时等待 | 占内存死等 or 超时失败 | **挂起 + 释放资源**,事件回来再唤醒 |
| 人审 / HITL | 难,得自己搭状态机 | **中断点**原生支持,暂停数小时无压力 |
| 副作用语义 | at-most/at-least once,易重复 | **exactly-once 效果**(靠幂等 + 日志去重) |
| 运维 | 简单,但不适合长跑有状态 | 需后端(Postgres 等),换来可靠长跑 |
| 适合 | 短、无状态、可重试的请求 | **长跑、有状态、需恢复**的 agent |

一句话:无状态 serverless 赌"任务足够短、崩了重跑代价低";agent 任务又长又贵又有副作用,这个赌注不成立,所以要 durable execution。

## 何时用 / 坑

**该用**:任何**要在生产长跑、且崩溃重跑代价高**的 agent——多步研究、端到端编码、跨服务编排、需要 Human-in-the-Loop 审批或长时等待外部事件的工作流。只要"重头再来"会重烧大量 token 或重复有副作用的操作,就该上 durable execution。

**不该用 / 可简化**:**短、无副作用、重跑无所谓**的一次性调用(单轮问答、无状态分类),上 durable execution 是过度工程;此时普通 serverless + 简单 retry 就够。原型阶段也可先用 LangGraph 的 MemorySaver,生产再换 Postgres saver。

**常见坑**:
- **非确定性破坏重放**:重放要求工作流主体逻辑**确定**(同样的日志重放出同样的执行路径)。在工作流主体里直接用 `random()`、`now()`、读全局可变状态会让重放路径与原路径分叉、引擎报错。所有不确定/有副作用的东西必须包进 **activity / 节点**,不能裸写在编排逻辑里。
- **幂等没做**:工具不幂等 + 崩在"已执行未记录"的窗口 → 副作用执行两遍(发两次邮件、下两次单)。副作用步骤必须幂等键或日志去重。
- **检查点反序列化的安全**:从数据库恢复检查点时若不限制可反序列化的类型,数据库被攻破可导致代码执行。LangGraph 可设 `LANGGRAPH_STRICT_MSGPACK=true` 或 `allowed_msgpack_modules` 白名单限制——这是真实的 Prompt Injection / 供应链相关风险面,别忽视。
- **状态膨胀**:每步都把全量上下文塞进检查点,长程会让检查点暴大、恢复变慢。配合 [[21 上下文压缩与卸载|上下文压缩与卸载]]:检查点只存必要的最小状态 + 外部记忆指针,而非整段对话历史。
- **版本演进(determinism + 代码升级)**:工作流跑到一半你改了代码,旧实例重放可能对不上新逻辑。Temporal 等提供 versioning / patching 机制,需提前规划。
- **后端成了单点**:Postgres / event store 挂了全停。它的可用性即 agent 的可用性,要按关键依赖对待。

## 关键事实

- agent 是**长跑、有状态、可中断、需恢复**的工作流,不能当无状态 HTTP;生产部署的正解是 **durable execution**。
- 四大机制:**持久日志/检查点 → 崩溃重放恢复 → 中断/挂起/resume → 幂等(exactly-once 效果)**。
- 重放精髓:已完成的 step **从日志取结果、不重新调** LLM/工具 → 崩溃恢复**不重复烧 token、不重复副作用**。
- **中断点**让 agent 可暂停数小时等人审批或外部事件,worker 等待期间无状态、不占资源——这是 Human-in-the-Loop 在生产里的落地方式。
- 重放要求工作流主体**确定**:不确定/有副作用的东西必须封进 activity/节点,不能裸写进编排逻辑。
- 代表技术:**Temporal**(标杆、最强一致)、**DBOS**(最轻、Postgres 原生)、**Restate**(更简单)、**Inngest**(事件驱动)、**LangGraph checkpointer**(把 agent 当有状态图)。
- 与 [[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进(Ralph loop)]] 的关系:Ralph 用 `git commit` 当穷人版检查点,durable execution 是它的工业级正解;两者都靠"持久化进度 + 可恢复"扛长程。

## 主流开源实现 / Python 库

- **`temporalio/sdk-python`**(GitHub,pip `temporalio`):Temporal 官方 Python SDK;品类标杆,exactly-once、可无限期运行,有与 OpenAI Agents SDK 的 durable agent 集成。
- **`dbos-inc/dbos-transact-py`**(GitHub,pip `dbos`):DBOS Transact,基于 **Postgres** 的超轻量持久工作流,装饰器即可加持久化与队列,无需独立编排器;提供 `DBOSAgent` 让 Pydantic AI agent 变持久。2026 年活跃(已发 Go SDK、Databricks Lakebase 合作)。
- **`langchain-ai/langgraph`**(GitHub,pip `langgraph`)+ **`langgraph-checkpoint-postgres`**(pip):把 agent 当有状态图,自动 checkpoint + interrupt/resume;生产用 **PostgresSaver**(2026-05 随 LangGraph 1.2 系列更新),原型用 MemorySaver/SqliteSaver。
- **`restatedev/restate`**(GitHub,Python SDK `restate-sdk`):Restate,比 Temporal 模型更简单、更易运维的 durable execution。
- **`inngest/inngest`**(GitHub,Python SDK `inngest`):Inngest,事件驱动的持久工作流,平台层托管重试与可观测性。
- 相关生态:**`hatchet-dev/hatchet`**(Postgres 之上的分布式任务/工作流引擎)也常被列入 2026 durable execution 候选,适合需要任务队列 + 持久编排的 agent 后端。

## 工业界实践

**谁在用、怎么选型。** durable execution 在 agent 之前就是 Netflix、Uber、Stripe、Coinbase 等做长跑业务编排(订单 saga、对账、跨服务事务)的成熟范式;2025–2026 因 agent 长跑需求第二次走红,选型大致分三档:

- **要最强一致 + 已有平台团队 → Temporal**:exactly-once、可无限期运行、versioning/patching 完备,身经百战;代价是要独立运维一个 Temporal 集群(或用 Temporal Cloud),学习曲线陡(workflow/activity 的确定性约束严格)。大厂或对正确性零容忍(金融、交易类 agent)的场景首选。
- **要最轻、不想多运维一个编排器 → DBOS**:直接构建在 **Postgres** 之上,给普通函数加装饰器即得持久工作流 + 队列,运维足迹最小;适合"已经有 Postgres、不想引入重型基建"的中小团队。`DBOSAgent` 可把 Pydantic AI agent 一键变持久。
- **已在用 LangGraph 写 agent → 直接上它的 checkpointer**:把 agent 当有状态图,自动 checkpoint + interrupt/resume,与 agent 编排无缝。原型用 MemorySaver/SqliteSaver,**生产换 PostgresSaver**。这是"agent 框架自带 durable"的路线,改动最小。
- **事件驱动、想要托管重试与可观测 → Inngest**;**想要更简模型、更易运维 → Restate**。

**典型生产架构。** 一个生产长跑 agent 通常长这样:**无状态 worker 池**(可随意扩缩容/被杀)+ **持久后端**(Postgres / event store,存检查点与日志)+ **调度器/编排器**(决定唤醒哪个挂起的工作流)。请求进来创建一个工作流实例(带 `workflow_id` / `thread_id`),worker 拉起执行,每步落日志;遇人审或长等待就**挂起 + 释放 worker**,事件回来由调度器换个新 worker 从检查点 resume。关键性质:**worker 无状态、状态全在后端**,所以扩缩容/滚动升级/机器故障都不丢进度。

**规模化、成本与延迟。**
- **成本面**:durable execution 省的是"崩溃重跑的重复 token"——一个跑了三天的 agent 崩了,重放只快进不重算,**已花的 LLM/工具调用一分钱不重花**,这对动辄几十上百次大模型调用的 agent 是巨大节省(联动 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]] 的"压重复计算")。
- **延迟面**:正常路径下每步多一次"写日志"的开销(通常几毫秒到几十毫秒),相对 LLM 调用的秒级延迟可忽略;真正的延迟收益在**恢复**——崩溃后几秒续跑 vs 从零重跑几分钟到几天。
- **吞吐瓶颈**:持久后端(Postgres)的写入是规模化瓶颈,高 QPS 下需分库/分区、检查点瘦身、批量写日志。

**可观测与运维。** durable runtime 自带强可观测性是它相对"手搓状态机"的核心优势:**每个工作流实例的完整执行历史可查**(走到第几步、每步输入输出、为何挂起、重试几次)、卡住的实例可定位、可"时间旅行"调试(重放到任意历史点)。运维重点:① 监控**挂起过久**的实例(可能死等一个永不到来的事件);② 监控**重试风暴**(某 activity 反复失败);③ **后端可用性 = agent 可用性**,按关键依赖对待(Postgres 主从/备份);④ 检查点**状态膨胀**告警,防恢复变慢。

**踩坑与最佳实践。**
- **确定性纪律**:工作流主体里**绝不**裸写 `random()` / `now()` / 读全局可变状态 / 直接 HTTP——全部封进 activity/节点,否则重放路径分叉、引擎报错。这是新手最常翻车的点。
- **幂等优先设计工具**:能天然幂等就天然幂等(用幂等键、PUT 而非 POST、唯一约束去重),把"恰好一次"的保证下沉到工具层而非只靠 runtime。
- **检查点只存最小必要状态**:别把整段对话历史塞进检查点,只存"决策必需的最小状态 + 外部记忆指针"(联动 [[21 上下文压缩与卸载|上下文压缩与卸载]]),否则长程检查点暴大、恢复变慢。
- **反序列化安全**:从 DB 恢复检查点要限制可反序列化类型(LangGraph `LANGGRAPH_STRICT_MSGPACK=true` / `allowed_msgpack_modules` 白名单),否则后端被攻破可致 RCE。
- **版本演进提前规划**:工作流跑一半改了代码,旧实例重放可能对不上新逻辑;用 Temporal 的 patching / LangGraph 的兼容策略,别在有 in-flight 实例时做破坏性改动。

## 面试高频

**Q1:为什么不能把长跑 agent 当无状态 HTTP 服务部署?**
标准答:agent 是**长跑(几分钟到几天)、有状态(进度是核心资产)、会长时间等待(慢工具/外部事件/人审)**的工作流,而无状态服务的心智是"几百毫秒算完即弃、内存用完丢"。把 agent 塞进无状态 serverless,**进程一死内存里全部进度归零、从头重跑**——重烧 token、重复执行有副作用的操作(发两次邮件、下两次单)。正解是 durable execution。

**Q2:durable execution 的"崩溃恢复"到底怎么做到不重复烧 token?**
标准答:靠**持久日志 + 确定性重放**。执行被拆成一连串 step/activity,每完成一步就把"输入+返回结果"写进日志。崩溃后新进程**重新执行工作流代码**,但走到每个**已记录的 step 时不真的再调 LLM/工具,而是直接从日志取上次结果喂回去**,把代码"快进"到崩前一刻,只在第一个**没有日志的 step**(中断处)才真正重新执行。所以已完成的调用一分钱不重花。
- **追问:这要求什么前提?** 工作流主体**确定**(同样日志重放出同样路径)+ 有副作用的 step **幂等**。

**Q3:durable execution 给的是 at-least-once 还是 exactly-once?差别在哪?**
标准答:给的是 **exactly-once 效果**(而非简单 retry 的 at-least-once)。崩溃可能发生在"工具已执行、结果还没写进日志"的窗口,恢复时该 step 会被重试——若工具不幂等就会做两遍。所以 exactly-once 不是 runtime 单独能给的,而是 **runtime 的日志去重 + 工具自身幂等(幂等键)** 合起来达成的"恰好一次"语义。这正是 durable execution 区别于"加个 retry 装饰器"的关键。

**Q4:agent 要等人审批几小时,系统里发生了什么?(中断/恢复机制)**
标准答:引擎把**当前位置 + 变量 + "在等什么"**持久化,然后**释放执行资源、挂起工作流**,worker 保持无状态、等待期间不空占内存。审批信号(signal / 外部事件)到达时,调度器**唤醒**工作流、拉一个新 worker、从检查点 **resume** 继续。这就是 Human-in-the-Loop 审批、长等待、定时唤醒在生产 agent 里的落地方式——暂停数小时甚至数天毫无压力。

**Q5:重放为什么要求"确定性"?哪些写法会破坏它?**
标准答:重放是"用同一份日志重新跑一遍工作流代码,期望走出同一条执行路径"。若主体逻辑里有**非确定性**(`random()`、`now()`、读全局可变状态、网络抖动),重放路径会与原路径分叉,引擎对不上日志就报错。修法:把**所有不确定/有副作用的东西封进 activity/节点**,工作流主体只做确定性的编排。
- **陷阱**:面试者常说"重放就是再跑一遍"——要点出"已记录的 step 不重跑、只取结果",否则就是普通重试不是 durable replay。

**Q6:Ralph loop 和 durable execution 是什么关系?**
标准答:[[33 长程任务与自我改进(Ralph loop)|Ralph loop]] 用 `git commit` 当**穷人版检查点**,崩了从最近 commit 续;durable execution 是它的**工业级正解**——更细的持久化粒度、确定性重放、exactly-once 副作用、原生人审中断。两者都靠"持久化进度 + 可恢复"扛长程,区别在可靠性等级与副作用保证。

**Q7:Temporal / DBOS / LangGraph checkpointer 怎么选?**
标准答:要最强一致 + 有平台团队 → **Temporal**;要最轻、只想加装饰器、已有 Postgres → **DBOS**;已用 LangGraph 写 agent、想改动最小 → **LangGraph checkpointer(生产用 PostgresSaver)**。事件驱动要托管重试 → Inngest;要更简模型 → Restate。

## 知识拓展

**技术谱系:从 event sourcing 到 durable execution。** durable execution 的思想根在**事件溯源(event sourcing)**——不存"当前状态"而存"导致状态的事件序列",任意时刻可重放事件重建状态。durable runtime 把这套用到**函数执行**上:把"程序执行的进度"本身变成可持久化、可重放的事件日志。它也吸收了 **saga 模式**(长事务拆成可补偿的子事务)和 **workflow engine**(如更早的 AWS Step Functions、Cadence——Temporal 的前身)的经验。2025 年起,agentic AI、长跑 saga、事件驱动任务收敛到了"一个崩溃后能精确续跑的函数"这同一个原语。

**与 agent 其它模块的联动。**
- **检查点 × 上下文管理**:检查点该存什么直接关系 [[21 上下文压缩与卸载|上下文压缩与卸载]]——只存最小状态 + 外部记忆指针,把大块历史卸载到外部存储,检查点才不会暴大。
- **中断恢复 × 记忆系统**:resume 时从 [[19 Agent 记忆系统|Agent 记忆系统]] 的外部记忆重新拉取上下文,而非把全量历史塞进检查点。
- **durable × 成本优化**:重放"不重花已完成调用"本质是 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]] 里"压重复计算"的一种——只不过省的是崩溃重跑的重复,而非同前缀的重复。

**前沿与边界(带年份)。**
- **2025–2026 的 "durable agents" 浪潮**:Temporal、DBOS、LangGraph 都在 2025–2026 推出/强化了"把 agent 包成持久工作流"的官方集成(如 Temporal × OpenAI Agents SDK、DBOS × Pydantic AI、LangGraph 1.x 的 PostgresSaver),标志这个方向从"分布式系统范式"正式被 agent 生态吸收为一等公民。
- **反序列化即安全面**:从持久后端恢复状态时的不受限反序列化,是 2025 年被反复提及的真实风险面(与提示注入 / 供应链攻击相关),LangGraph 的 msgpack 白名单是直接回应。
- **边界**:durable execution **不是免费的可靠性**——它要求你接受确定性约束(编排逻辑不能裸写副作用)、运维一个持久后端、并提前规划版本演进。对**短、无副作用、重跑无所谓**的一次性调用(单轮问答、无状态分类),上它是**过度工程**,普通 serverless + 简单 retry 就够。

**反模式。**
- **把不确定逻辑裸写进工作流主体**:最常见的翻车,重放直接炸。
- **指望 runtime 单独给 exactly-once**:不做工具幂等,崩在"已执行未记录"窗口仍会双发副作用。
- **检查点塞整段对话历史**:长程检查点暴大、恢复变慢、后端写爆。
- **忽视后端单点**:Postgres/event store 挂了全停,却没按关键依赖做高可用与备份。
- **有 in-flight 实例时做破坏性代码升级**:旧实例重放对不上新逻辑,需 versioning/patching 而非硬改。

延伸联动:[[33 长程任务与自我改进(Ralph loop)|Ralph loop]](穷人版检查点)、[[21 上下文压缩与卸载|上下文压缩与卸载]](检查点瘦身)、[[19 Agent 记忆系统|Agent 记忆系统]](resume 时重拉上下文)、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]](重放省重复调用)。
