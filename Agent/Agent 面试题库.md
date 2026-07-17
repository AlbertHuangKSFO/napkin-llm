[[Agent|Agent 学习地图]] 的面试路由：按题型进入对应原子笔记的 `#面试高频`，只用于定位，不在这里重复答案。

## 基础与选型

- Agent 与 workflow 的边界如何判断？追问：什么任务应先用 workflow？[[02 Workflow 与 Agent 的边界#面试高频|去答题]]
- LLM Agent 的最小定义是什么？追问：与聊天机器人、自动化脚本的差异？[[01 什么是 AI Agent#面试高频|去答题]]
- 一个 Agent 核心循环由什么驱动，如何定义可处理的终态？[[03 Agent 核心循环#面试高频|去答题]]
- ReAct、Plan-and-Execute、ReWOO 分别适合什么任务形状？追问：何时需要重规划？[[09 ReAct#面试高频|ReAct]] [[10 Plan-and-Execute#面试高频|Plan-and-Execute]] [[11 ReWOO#面试高频|ReWOO]]
- Parallelization 与 Orchestrator-Workers 的拆分时机有什么不同？[[06 Parallelization#面试高频|Parallelization]] [[07 Orchestrator-Workers#面试高频|Orchestrator-Workers]]
- 选 Agent 框架时应比较哪些运行时语义，而非只看生态热度？[[37 Agent 框架对比#面试高频|去答题]]

## 工具/协议

- Function Calling 中模型实际做了什么？`strict` 能解决哪些问题、不能解决哪些问题？[[15 Function Calling 工具调用#面试高频|去答题]]
- 如何设计一个模型容易选对、填对、出错后可纠正的工具接口？[[16 工具设计与工具层#面试高频|去答题]]
- MCP 与 Function Calling 是否等价？MCP 的发现、调用与授权如何分开理解？[[17 MCP 模型上下文协议#面试高频|去答题]]
- 工具发现、检索、选择、执行分别在哪里失败？如何做动态加载？[[18 工具检索与动态加载#面试高频|去答题]]
- A2A 与 MCP 的协作边界是什么？两者何时一起使用？[[30 A2A 协议#面试高频|去答题]]
- 浏览器 Agent 何时是合适的适配层，何时仍应优先 API 或 MCP？[[27 计算机使用与浏览器 Agent#面试高频|去答题]]

## 上下文/Skills

- Prompt engineering 与 context engineering 的边界是什么？上下文应如何按轮组装？[[20 上下文工程#面试高频|去答题]]
- Agent 记忆如何兼顾跨会话可用性、租户隔离、授权和时效？[[19 Agent 记忆系统#面试高频|去答题]]
- Compaction 为什么不意味着 Agent 可以无限运行？失败信息该怎样保留？[[21 上下文压缩与卸载#面试高频|去答题]]
- 代码或知识检索为何不能只押注向量检索？`rg`、AST/LSP、BM25、embedding 如何选？[[24 Agentic Search：grep vs 向量检索#面试高频|去答题]]
- Skill 与 MCP 的职责如何区分？如何验证一个 Skill 的匹配、路径与权限？[[25 Agent Skills(SKILL.md)#面试高频|去答题]]
- Prompt Contract 与 Skill 如何分工，Skill 能否扩张本次运行的权限？[[41 Agent 指令设计：Prompt Contract 与行为边界#面试高频|去答题]]

## 代码 Agent

- 代码 Agent 的“读 issue—改仓库—测试—检查 diff”闭环如何设计？何时应交给人？[[28 代码 Agent 与 SWE-bench#面试高频|去答题]]
- SWE-agent 与 SWE-bench 的关系是什么？引用基准分数时必须交代哪些条件？[[28 代码 Agent 与 SWE-bench#面试高频|去答题]]
- 如何为代码 Agent 定义可靠性评估、独立验证器与反作弊边界？[[42 代码 Agent 评估：可靠性、验证器与反作弊#面试高频|去答题]]
- 多文件代码任务如何选择搜索策略，并用可定位证据约束修改范围？[[24 Agentic Search：grep vs 向量检索#面试高频|去答题]]
- 长程编码任务怎样从可恢复状态继续，而不是依赖同一段对话记忆？[[33 长程任务与自我改进(Ralph loop)#面试高频|去答题]] [[40 Loop Engineering：可验证的 Agent 外循环#面试高频|Loop Engineering]]

## 评估/可靠性

- 为什么最终文本正确仍可能是一次 Agent 失败？终态应由谁验证？[[38 Agent 评估与可观测性#面试高频|去答题]]
- LLM-as-judge 与确定性/权威 verifier 如何分工？何时需要人工复核？[[38 Agent 评估与可观测性#面试高频|去答题]]
- Loop Engineering 与 Prompt Engineering 的差别是什么？外循环应怎样触发、停止与恢复？[[40 Loop Engineering：可验证的 Agent 外循环#面试高频|去答题]]
- 为什么 tests passed 还不足以自动上线？批准与 revision 应怎样绑定？[[40 Loop Engineering：可验证的 Agent 外循环#面试高频|去答题]]
- 持久化执行中的 replay、activity 与外部副作用分别能保证什么？[[34 Agent 部署与持久化执行#面试高频|去答题]]
- 成本与延迟应如何在合格任务率、关键路径与失败风险下衡量？[[35 Agent 成本与延迟优化#面试高频|去答题]]

## 安全/系统设计

- 为什么 prompt 不是访问控制？最小运行时边界由哪些部件组成？[[41 Agent 指令设计：Prompt Contract 与行为边界#面试高频|去答题]]
- 面对提示注入，如何划分不可信数据与系统指令，并阻止工具输出越权？[[20 上下文工程#面试高频|去答题]] [[41 Agent 指令设计：Prompt Contract 与行为边界#面试高频|Prompt Contract]]
- 模型提出工具调用后，身份、资源归属、状态转换、限额与确认应在哪里校验？[[15 Function Calling 工具调用#面试高频|去答题]] [[16 工具设计与工具层#面试高频|工具层]]
- 为什么 Agent session 不能只存容器内存？审计、恢复与幂等如何协作？[[23 Agent Harness 概览#面试高频|去答题]] [[34 Agent 部署与持久化执行#面试高频|持久化执行]]
- 多智能体如何定义责任边界、汇总机制与故障隔离？[[22 多智能体系统#面试高频|去答题]] [[26 Sub-agents 与 Agent Teams#面试高频|协作拓扑]]
- Agentic RAG 如何避免“多检索一次”就被误称为 Agent，并控制证据与停止条件？[[36 Agentic RAG#面试高频|去答题]]
