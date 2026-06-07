[[23 Agent Harness 概览|Agent Harness 概览]] 的本质:harness 是**包裹模型推理循环的「非模型」运行时**——它不产生 token,却决定模型产生的 token 能不能变成真实世界里完成的任务。一句话拍死:**2026 的 AI 工程,大部分其实是 harness 工程;模型趋于商品化,差异来自你怎么包它——同一个模型、不同 harness,一个能开 PR,另一个空转**。

## 本质:模型是引擎,harness 是整辆车
把 LLM 想成一台发动机:它做的事很纯粹——给定一段上下文,预测下一个 token。但「预测下一个 token」和「帮我改完这个仓库的 bug 并提交 PR」之间隔着一道巨大的鸿沟。填这道鸿沟的所有东西,就是 harness。

harness(直译「马具/挽具」)是**环绕模型推理的那一整圈运行时代码**:它决定什么时候调模型、把什么塞进上下文、模型要调工具时谁去执行、执行结果怎么喂回去、什么时候该停、出错怎么重试、危险操作要不要拦、会话状态存哪。这些**全是「非模型」的工程**——换个模型权重,这一圈逻辑基本不变;但这一圈写得好不好,直接决定同一个模型是「能干活」还是「在兜圈子」。

![[Agent Harness 概览.png]]

这就引出本篇最反直觉的论点:**模型在快速商品化,真正拉开差距的是 harness**。2026 年,顶级模型之间的推理能力差距在收窄,且随时可替换;但把同一个 Claude / GPT 放进两个不同的 harness 里,产出可以天差地别——一个能读懂任务、改完代码、跑测试、开出 PR;另一个在重复读同一个文件、丢失任务目标、最后空转超时。**「同一个模型、不同 harness,一个开 PR,一个空转」** 不是修辞,是工程现实。

[[03 Agent 核心循环|Agent 核心循环]] 描述的「思考→行动→观察」是 agent 的灵魂,但那个循环本身得有人来驱动、来兜底、来管上下文——驱动它的就是 harness。可以说:**核心循环是 what,harness 是 how it actually runs**。

## 机制:harness 的四大职责
把 harness 拆开,它要解决四类问题。下面逐个讲透「各自要解决什么、写不好会怎样」。

### ① Loop —— 推理循环的调度
**要解决**:模型一次推理只产出一步(一段思考、一次工具调用、或一个最终答案)。谁来判断「这是中间步骤还是最终答案」?谁来把工具结果拼回上下文、再发起下一轮?谁来决定「转了 30 圈还没收尾,该熔断了」?这就是 loop 层。

- 解析模型输出:是要调工具,还是给最终答案?
- 终止判断:任务完成了吗?要不要再来一轮?设最大步数防死循环。
- 重试与降级:工具报错、模型输出不合法 schema,是重试、换路径,还是放弃?

**写不好的后果**:循环不会收尾(空转烧钱)、或过早收尾(任务没做完就交差)。这是 [[09 ReAct|ReAct]]、[[10 Plan-and-Execute|Plan-and-Execute]] 这些范式落地时最先踩的坑——范式是对的,但驱动范式的 loop 没兜住边界条件。

### ② Tool —— 工具的调度与执行
**要解决**:模型只会「说」它想调哪个工具、传什么参数(一段结构化文本)。真正去执行——发 HTTP、跑 shell、查数据库——是 harness 的活。它还要把执行结果**格式化后喂回模型**,且格式必须和模型期望的 schema 严丝合缝。

- 工具注册:告诉模型有哪些工具、各自的参数 schema(见 [[15 Function Calling 工具调用|Function Calling 工具调用]])。
- 参数解析与校验:模型给的参数可能缺字段、类型错,harness 得校验甚至纠偏。
- 执行与回填:执行工具,把结果(可能很长)裁剪、格式化后塞回上下文。

**写不好的后果**:**schema 错位**——模型以为工具返回 A 格式,harness 给了 B 格式,模型读不懂,后面全崩。工具层设计本身见 [[16 工具设计与工具层|工具设计与工具层]];harness 是让那套设计真正跑起来的执行体。

### ③ Context —— 上下文管理
**要解决**:上下文窗口有限,而一个长任务会累积海量历史(几十轮工具调用、大段文件内容)。塞不下怎么办?塞进去的东西过时了怎么办?关键信息被淹没怎么办?这是 [[20 上下文工程|上下文工程]] 在 harness 里的落地点。

- 裁剪与压缩:历史太长时,摘要旧轮次、丢掉无关内容。
- 记忆注入:在合适时机把 [[19 Agent 记忆系统|Agent 记忆系统]] 里的相关记忆拉进当前上下文。
- 防漂移:确保任务目标、关键约束始终在场,不被后续噪声挤掉。

**写不好的后果**:**上下文漂移(context drift)**——模型逐渐忘了最初要干什么,开始答非所问;或**状态退化**——本该记住的中间结论被压缩掉了。这是企业级 agent 失败的头号原因之一。

### ④ Safety —— 安全、权限与持久化
**要解决**:agent 要执行真实操作(删文件、调付费 API、发邮件),不能让它无边界乱来。同时一个跑几小时的任务不能因为进程重启就全丢。

- 沙箱:危险操作隔离在受限环境里执行。
- 权限与审批:高风险动作需要人确认(human-in-the-loop),或限定白名单。
- 会话持久化:状态落盘,支持断点续跑、审计回放。

**写不好的后果**:要么太松(agent 把生产库删了),要么太紧(每步都要人点确认,失去自动化价值),要么没持久化(长任务一崩全白干)。

![[Agent Harness 概览-结果分叉.png]]

## 关键数据:为什么说大部分是 harness 工程
**有分析称 65% 的企业 AI 失败源于 harness 缺陷(上下文漂移、schema 错位、状态退化),而非模型推理能力不足**。这个数字的含义很尖锐:当一个 AI 项目失败,人们第一反应是「模型不够聪明」,但更多时候,是包在模型外面的那一圈运行时没做好——上下文管乱了、工具 schema 对不齐、长任务状态退化了。换模型解决不了这些;换 harness 才行。

这正是论点的数据支撑:**模型趋于商品化,差异来自你怎么包它**。你买不到「更好的推理」来救一个 harness 烂的系统,但你能通过把 harness 做扎实,让一个普通模型稳定交付。

## 趋势:HaaS(Harness-as-a-Service)
既然 harness 这么关键又这么难手搓,自然的演化是:**把 harness 做成服务/SDK,开箱即给**。这就是 **HaaS(Harness-as-a-Service)**。

![[Agent Harness 概览-HaaS栈.png]]

代表产品:**Claude Agent SDK、Codex SDK、OpenAI Agents SDK** ——它们都把 **loop、工具、上下文管理、hooks、沙箱**这几层开箱即给。你不再需要从零手写 agent 循环、自己管上下文压缩、自己搭沙箱;SDK 把这些 harness 职责封装好,你只写「业务逻辑 + 工具 + 提示」,底层模型还能替换。

这件事的战略意义:它把「会做 agent」的门槛从「能手搓一整套运行时」降到「会用 SDK 拼装」,同时把 harness 的最佳实践(怎么防漂移、怎么压上下文)固化进基础设施。对个人开发者,意味着 [[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]]、[[25 Agent Skills(SKILL.md)|Agent Skills(SKILL.md)]] 这类能力直接可用;对行业,意味着竞争焦点从「模型」进一步上移到「怎么用 harness 把模型组织成能干活的系统」。

## 可跑最小示例(伪代码:一个最朴素的 harness)
下面这段把四大职责都点到,展示 harness「不产生 token、却决定一切」的本质:

```python
def harness(task, tools, max_steps=30):
    ctx = build_initial_context(task)          # ③ Context:组初始上下文
    for step in range(max_steps):              # ① Loop:循环 + 熔断上限
        out = llm(ctx, tools=tool_schemas(tools))  # ② Tool:把工具 schema 告诉模型
        if out.is_final():                     # ① Loop:终止判断
            return out.answer
        call = out.tool_call                   # 模型只是"说"要调哪个工具
        if needs_approval(call):               # ④ Safety:高危操作要人确认
            if not ask_human(call):
                ctx = append(ctx, "操作被拒绝")
                continue
        try:
            result = run_in_sandbox(tools[call.name], call.args)  # ④ Safety:沙箱执行
        except SchemaError:
            ctx = append(ctx, "参数不合法,请修正")  # ② Tool:schema 校验/纠偏
            continue
        ctx = append(ctx, format_result(result))  # ② Tool:结果回填
        ctx = compress_if_too_long(ctx)        # ③ Context:防漂移/防溢出
        save_session(ctx)                      # ④ Safety:持久化,支持续跑
    return "达到步数上限,未完成"                # ① Loop:兜底
```

注意:**这段代码里没有一行是「模型推理」**——`llm()` 那一行才是模型,其余全是 harness。但任务能不能成,几乎全押在这些「非模型」的行上。

## 对比表:模型 vs harness 各管什么
| 维度 | 模型(LLM) | Harness(运行时) |
|---|---|---|
| 产出 | token / 工具调用意图 / 最终答案 | 不产 token;调度、执行、管理 |
| 是否易替换 | 是,趋于商品化 | 是差异化所在,难替换 |
| 失败表现 | 推理错、幻觉 | 上下文漂移、schema 错位、状态退化 |
| 四职责 | —— | ① Loop ② Tool ③ Context ④ Safety |
| 提升手段 | 换更强模型 | 把运行时工程做扎实 / 上 HaaS |

## 何时关注 harness / 坑
- **何时重点投 harness**:任务要跑多步、要调真实工具、要跑很久、要管大量上下文——也就是任何真正的 agent。越是长程、越是高风险,harness 占成败的权重越大。
- **坑一:把失败归错给模型**。系统不灵就想换更大的模型,往往治标——先查是不是上下文漂移 / schema 错位 / 状态退化(这三样是 harness 病)。
- **坑二:重复造轮子**。2026 年大概率不该从零手写 loop / 沙箱 / 上下文压缩,先看 Claude Agent SDK、OpenAI Agents SDK 等 HaaS 能不能直接用。
- **坑三:忽视 Safety 层**。demo 阶段不上沙箱/权限没事,一旦让 agent 碰生产环境,缺这层就是事故。
- **坑四:上下文「塞满即最优」的误解**。塞得越多不等于越好,关键是 [[20 上下文工程|上下文工程]] —— 让对的信息在场、把噪声挤走;harness 的 context 层就是干这个的。
- **坑五:没有可观测性**。harness 跑出问题,没日志没回放就无法定位,必须配 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

## 2026 关键事实
- **harness 是「包裹模型推理循环的非模型运行时」**:协调工具调度、上下文管理、安全/权限、会话持久化。
- 核心论点:**2026 的 AI 工程,大部分其实是 harness 工程;模型趋于商品化,差异来自你怎么包它——同一个模型、不同 harness,一个能开 PR,另一个空转**。
- 关键数据:**有分析称 65% 的企业 AI 失败源于 harness 缺陷(上下文漂移、schema 错位、状态退化),而非模型推理能力不足**。
- 趋势 **HaaS(Harness-as-a-Service)**:**Claude Agent SDK、Codex SDK、OpenAI Agents SDK** 都把 loop、工具、上下文管理、hooks、沙箱开箱即给。
- harness 四大职责:**① Loop(推理循环调度)② Tool(工具调度执行)③ Context(上下文管理)④ Safety(安全/权限/持久化)**——记住这四象限,就抓住了 harness 的全貌。
- 关联:[[03 Agent 核心循环|Agent 核心循环]] 是被 harness 驱动的灵魂;[[20 上下文工程|上下文工程]] 与 [[16 工具设计与工具层|工具设计与工具层]] 是 harness 两大职责的深入;[[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]]、[[25 Agent Skills(SKILL.md)|Agent Skills(SKILL.md)]] 是现代 harness/SDK 提供的高级能力;[[38 Agent 评估与可观测性|Agent 评估与可观测性]] 是 harness 的体检手段。

## 工业界实践

**「harness > 模型」不是修辞,有硬数据**。SWE-bench 上,**同一个模型只改 scaffold(harness),分数能从 42% 跳到 78%**;SWE-bench Pro 上,scaffold 变化对同一模型造成 **22+ 个百分点的摆动**。另一组数据:仅靠更好的工具编排、错误恢复、每步上下文压实,完成率从 34% 提到 71%,**没换模型**。OpenHands、SWE-agent 这些 scaffold 在相同模型上表现差异显著——这就是「同一个模型、不同 harness,一个开 PR 一个空转」的实证。

**主流 HaaS(Harness-as-a-Service)产品与定位**:
- **Claude Agent SDK**(原 Claude Code SDK 改名,体现已超出编码):内置 compaction、subagents-as-tools、hooks、沙箱。2025-09-29 上线 **memory tool**(beta,客户端执行,跨会话存取)。
- **OpenAI Agents SDK**:loop + handoff + guardrails + tracing 开箱即给。
- **Codex SDK / Managed Agents**:把「大脑(模型)和手(执行)」解耦,session 提供活在 context window 之外的持久 context 对象。
- 开源 harness:**OpenHands**(原 OpenDevin)、**SWE-agent**(为 SWE-bench 而生的最小 agent-computer interface)、LangGraph(把四职责显式建图)。

**长程任务的生产架构(Anthropic「Effective harnesses for long-running agents」)**:核心是**跨多个上下文窗口接力**——一个 **initializer agent** 首次运行搭好环境,一个 **coding agent** 每个 session 只做增量进展、并为下个 session 留下清晰 artifacts(把状态写进文件,而非全压在上下文里)。这把 ④Safety 的「会话持久化」和 ③Context 的「卸载到文件」做成了长程任务的脊梁,对应 [[21 上下文压缩与卸载|上下文压缩与卸载]] 与 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]。

**规模化与成本/延迟权衡**:harness 决定每轮往上下文塞多少(直接定 token 账单)、并行不并行(定墙钟时间)、重试几次(定延迟尾部)、压缩频率(定 [[102 KV-Cache|KV-Cache]] 命中)。一个写得糙的 loop 可能在同一文件上空转十几轮——成本全烧在 harness 的低质量决策上,而非模型。

**可观测运维**:harness 是埋点的天然位置——每轮记 step 数、tool 调用、token 占用、重试/熔断、是否触发审批。没有 trace 就无法定位「为什么空转/为什么漂移」,必须配 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**踩坑与最佳实践**:① 别从零手搓,先看 HaaS 能不能直接用;② demo 期可不上沙箱,碰生产必须上 ④Safety;③ schema 校验+纠偏放在 ②Tool 回填前,避免一处错位后面全崩;④ 防漂移:任务目标/关键约束始终在场,别被噪声挤掉。

## 面试高频

- **Q:什么是 agent harness?它和模型什么关系?**
  A:harness 是**包裹模型推理循环的「非模型」运行时**——不产 token,却决定模型的 token 能否变成完成的任务。模型是引擎,harness 是整辆车。核心论点:**2026 模型趋于商品化,差异来自怎么包它**。
- **Q:harness 有哪四大职责?**(必背)
  A:**① Loop**(推理循环调度:解析输出/终止判断/重试熔断)**② Tool**(工具调度执行:注册 schema/参数校验/结果回填)**③ Context**(上下文管理:裁剪压缩/记忆注入/防漂移)**④ Safety**(沙箱/审批 human-in-the-loop/会话持久化)。
- **Q:为什么说「换更大模型救不了 harness 烂的系统」?**
  A:企业 AI 失败大量源于 harness 病——**上下文漂移、schema 错位、状态退化**(有分析称约 65%),这三样换模型治不了。证据:SWE-bench 上同模型改 scaffold 能从 42%→78%。
- **Q:harness 写不好分别会怎样?**(按职责答)
  A:Loop 烂→空转烧钱或过早收尾;Tool 烂→schema 错位后面全崩;Context 烂→漂移(忘了最初要干啥)/状态退化(中间结论被压没);Safety 烂→太松删生产库 / 太紧每步要人点 / 没持久化一崩全白干。
- **Q:HaaS 是什么?代表产品?**
  A:Harness-as-a-Service——把 loop/工具/上下文管理/hooks/沙箱做成 SDK 开箱即给。代表:Claude Agent SDK、OpenAI Agents SDK、Codex SDK。意义:把「会做 agent」门槛从「手搓运行时」降到「会用 SDK 拼装」。
- **Q:系统不灵,你先查模型还是先查 harness?**(陷阱)
  A:先查 harness 三病(漂移/错位/退化),别急着归错给模型。这是本篇最反直觉、也最实用的结论。

## 知识拓展

- **进阶方向**:跨上下文窗口的**长程 harness**(initializer + 增量 coding agent + artifacts 接力);harness 内置的自我纠错(重试/降级/回滚);多 agent harness(把 ②Tool 的「subagents-as-tools」做成一等公民,见 [[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]])。
- **相关前沿(带年份)**:SWE-agent(2024,首次系统化「agent-computer interface」)、OpenHands/OpenDevin(2024)、Anthropic「Effective harnesses for long-running agents」与「Building agents with the Claude Agent SDK」(2025)、Managed Agents「解耦大脑与手」(2025)、SWE-bench Pro scaffold 摆动研究(2025)。
- **边界与反模式**:重复造轮子手搓 loop/沙箱;把失败归错给模型;忽视 Safety 层就碰生产;以为「上下文塞满即最优」(关键是让对的信息在场,见 [[20 上下文工程|上下文工程]]);没有可观测性无法定位。
- **本域关联**:harness 驱动的灵魂是 [[03 Agent 核心循环|Agent 核心循环]];②Tool/③Context 的深入是 [[16 工具设计与工具层|工具设计与工具层]] [[15 Function Calling 工具调用|Function Calling 工具调用]] [[20 上下文工程|上下文工程]] [[21 上下文压缩与卸载|上下文压缩与卸载]];卸载取回靠 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]];④Safety 的持久化延伸到 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]];成本/延迟权衡见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]];体检靠 [[38 Agent 评估与可观测性|Agent 评估与可观测性]];编码 harness 的标尺是 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]。
