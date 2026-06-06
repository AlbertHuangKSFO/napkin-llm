[[09 ReAct|ReAct]] 把「推理(Reason)」和「行动(Act)」交错进同一个循环:模型不再一口气想完再动手,而是想一步(Thought)、动一步(Action)、看一眼真实结果(Observation),再想下一步——用外部观测把自己从幻觉里拽回来。

它是 [[03 Agent 核心循环|Agent 核心循环]] 最经典、最被广泛实现的具体形态,几乎所有现代 agent 框架的默认推理范式都源自它。

## 本质:为什么要把"想"和"做"焊在一起

在 ReAct 之前,LLM 解复杂任务有两条路,各有致命短板:

- **纯推理(Chain-of-Thought)**:让模型在脑内一步步想("先算 A,再算 B…")。问题是它**只在参数里取知识**,拿不到外部世界的新事实;一旦中间某步记错,后面整条链顺着错下去,且**无从核对**——这就是"推理性幻觉"。
- **纯行动(Act-only)**:让模型直接吐工具调用序列(search→click→…),不解释为什么。问题是它**没有推理来引导动作**,面对多跳、需要分解的任务就盲目乱点,也无法根据中途结果调整策略。

ReAct 的洞见:**这两者本该互补**。

- 推理(Thought)给行动**定方向**:决定下一步该查什么、该用哪个工具、信息够不够。
- 行动(Action)+ 观测(Observation)给推理**喂事实**:把外部世界的真实返回注回上下文,作为下一轮推理的锚点。

于是幻觉被压住了——因为每隔一步就有一次真实 Observation 来校准;过程也**可追溯**——整条 Thought/Action/Observation 轨迹就是模型的"工作底稿",出错能定位到哪一跳。

## 机制:Thought → Action → Observation 的循环

![[ReAct.svg]]

一轮迭代分三段,循环往复直到模型主动收尾:

1. **Thought(推理)**:模型用自然语言写出当前的思考——任务进展到哪、还缺什么、下一步打算干嘛。这一段**不与外界交互**,纯粹是把规划显式化(也方便人读)。
2. **Action(行动)**:模型从 Thought 收敛出一个**结构化动作**,通常形如 `工具名[参数]`,例如 `Search[Apple Remote]`、`Lookup[设备]`、或终止动作 `Finish[答案]`。
3. **Observation(观测)**:外部环境(检索器 / API / 代码执行器…)执行该 Action,把**真实返回**作为 Observation 拼回上下文。

然后这段 `Thought_i / Action_i / Observation_i` 会附加到 prompt 里,模型据此生成 `Thought_{i+1}`……如此滚动,直到某一轮 Action 是 `Finish[...]`,循环终止、产出答案。

控制循环的是**外层程序(harness)**,不是模型:程序负责解析模型输出里的 Action、真正去调工具、把 Observation 塞回去、并设最大步数防止死循环。模型只管在每一步生成"下一段 Thought + Action"。

> 注意:原始 ReAct 用的是**文本协议**(模型吐 `Action: Search[...]` 字符串,靠正则解析)。今天的等价实现几乎都改用 [[15 Function Calling 工具调用|Function Calling 工具调用]],让模型输出结构化 tool call、由 runtime 执行——本质完全一致,只是把"解析 Action 文本"换成了"解析 tool_calls 字段",更鲁棒。

### 一段真实 trace 长什么样

![[ReAct-trace示例.svg]]

这是论文里的经典多跳例子:问"Apple Remote 最初设计来控制的程序,后来还能被哪些设备控制"。模型先 `Search[Apple Remote]` 拿到"控制 Front Row",再 `Search[Front Row]` 拿到"也可由键盘功能键控制",两条真实 Observation 集齐后才 `Finish`。纯 CoT 在第二跳会直接"脑补"一个设备名而答错——这正是 Observation 的价值。

## 原论文

**Yao et al., _ReAct: Synergizing Reasoning and Acting in Language Models_**(2022 年 10 月 arXiv,**ICLR 2023** 接收)。作者来自 Princeton 与 Google Research(Shunyu Yao、Jeffrey Zhao、Dian Yu、Nan Du、Izhak Shafran、Karthik Narasimhan、Yuan Cao)。

论文核心实验:在知识密集型问答(HotpotQA、FEVER)上,ReAct 接入维基检索 API,显著压低了纯 CoT 的幻觉;在交互决策环境(ALFWorld 具身任务、WebShop 网购)上,ReAct 大幅超过 Act-only 基线。最强的配置是 **ReAct + CoT-SC 互为补充**——能查就查、查不到就回退到内部推理。"ReAct"这个名字就是 **Reason + Act** 的合写。

## 可跑的最小实现

下面是不依赖任何框架、纯 prompt + while 循环的 ReAct 骨架(文本协议版,便于看清机制):

```python
import re

SYSTEM = """你按 ReAct 范式解题。每轮严格输出:
Thought: <你的推理>
Action: <工具名>[<参数>]
可用工具:Search[query]、Calculate[expr]、Finish[answer]
拿到 Observation 后继续下一轮,信息够了就用 Finish 给出答案。"""

def search(q):    ...   # 接你的检索器 / API
def calculate(e): return str(eval(e))

TOOLS = {"Search": search, "Calculate": calculate}
ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.S)

def react(question, llm, max_steps=8):
    scratchpad = f"Question: {question}\n"
    for _ in range(max_steps):
        # 只让模型生成到 Observation 之前(用 stop 截断)
        out = llm(SYSTEM, scratchpad, stop=["Observation:"])
        scratchpad += out
        m = ACTION_RE.search(out)
        if not m:                       # 没给合法 Action,提示纠正
            scratchpad += "\nObservation: 动作格式无效,请重出。\n"
            continue
        tool, arg = m.group(1), m.group(2).strip()
        if tool == "Finish":
            return arg                  # 终止,返回答案
        obs = TOOLS.get(tool, lambda x: "未知工具")(arg)
        scratchpad += f"\nObservation: {obs}\n"   # 真实结果回灌
    return "达到最大步数仍未完成"
```

要点:① 用 `stop=["Observation:"]` 让模型**只生成 Thought+Action**,Observation 由你的代码填,防止模型自己伪造观测;② `max_steps` 是必须的护栏;③ 解析失败要把错误当成 Observation 回灌,让模型自我纠正。生产环境把这套文本协议换成 [[15 Function Calling 工具调用|Function Calling 工具调用]] 即可。

## 对比:ReAct 在 agent 谱系里的位置

| 范式 | 推理与行动关系 | 何时问大模型 | Token / 延迟 | 抗幻觉 | 适合 |
|---|---|---|---|---|---|
| 纯 CoT | 只推理,不行动 | 一次 | 最省 | 差(无外部校准) | 纯逻辑/数学,无需外部信息 |
| Act-only | 只行动,不推理 | 每步 | 中 | 中 | 动作简单、无需分解 |
| **ReAct** | **每步交错** | **每一步** | 较高(每步重发全上下文) | **强**(每步真实观测) | 多跳、需边走边看的探索式任务 |
| [[10 Plan-and-Execute|Plan-and-Execute]] | 先规划再批量执行 | 规划1次+偏差时 | 较省 | 中 | 长程、步骤可预先排布 |
| [[11 ReWOO|ReWOO]] | 规划/取证/合成三段解耦 | 仅 Plan+Solve | 最省(中间不回灌) | 中(蓝图错难纠) | 步骤可静态规划、要省 token |

ReAct 的**软肋**正是它的代价:每一步都要把"到目前为止的全部 Thought/Action/Observation"重新发给大模型 → **token 随步数线性甚至更快膨胀、延迟高**;长程任务里上下文越滚越长,容易迷失方向。[[10 Plan-and-Execute|Plan-and-Execute]] 和 [[11 ReWOO|ReWOO]] 正是为压住这个成本而生——把"每步都问模型"砍成"只在关键点问"。

往上走,如果要让 agent **从失败中学习**就接 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]];如果要**探索多条推理/行动路径**而非单线前进,就上 [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]](LATS 本质就是给 ReAct 套上蒙特卡洛树搜索)。

## 何时用 / 坑

**该用 ReAct 的场景**:任务需要**边走边看**——下一步该干嘛严重依赖上一步的真实结果(多跳检索、网页浏览、需反复试探的环境交互、数据库探索)。这类任务无法预先排好完整计划,只能交错推进。

**不该用的场景**:步骤能**预先静态排布**的长流程(用 [[10 Plan-and-Execute|Plan-and-Execute]] 更省更稳);或纯内部推理、根本不碰外部信息的题(纯 CoT 即可,套 ReAct 反而浪费)。

**常见坑**:
- **死循环 / 不收敛**:模型反复查同一个东西或绕圈。必须设 `max_steps`,并在 prompt 里鼓励"信息够了就 Finish"。
- **伪造 Observation**:不加 `stop` 截断,模型会自己脑补出 Observation 内容,彻底废掉抗幻觉的初衷。Observation **必须**由你的代码真实执行后填入。
- **上下文爆炸**:步数多了 token 线性增长且越来越贵、越慢;长 trace 还会让模型"忘了"最初目标。对策:压缩历史 Observation、或换成 Plan-based 范式。
- **动作解析脆弱**:文本协议靠正则,模型一手抖格式就崩。生产请用 [[15 Function Calling 工具调用|Function Calling 工具调用]] 的结构化 tool call。
- **错误传播**:某步 Action 选错工具/错参数,后续会顺着错的 Observation 走偏;需要的话叠加 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 做自我修正。

## 关键事实

- ReAct = **Reason + Act**,把 CoT 的"想"和工具使用的"做"统一进一个交错循环。
- 三件套:**Thought(脑内推理,不碰外界)/ Action(结构化动作)/ Observation(外部真实返回)**。
- 抗幻觉的根源:**每隔一步就有一次真实 Observation 校准**;可追溯的根源:整条 trace 就是工作底稿。
- 循环由**外层 harness 控制**,模型只负责每步生成 Thought+Action;`max_steps` 与 `stop` 截断是两个必备护栏。
- 代价:**每步重发全上下文 → token/延迟随步数膨胀**,这是 [[10 Plan-and-Execute|Plan-and-Execute]]、[[11 ReWOO|ReWOO]]、[[12 LLMCompiler|LLMCompiler]] 要解决的核心问题。
- 现代实现普遍用 [[15 Function Calling 工具调用|Function Calling 工具调用]] 替代原始文本协议,机制不变、更鲁棒。

## 主流开源实现 / Python 库

- **`langchain-ai/langgraph`** —— prebuilt 的 `create_react_agent`,几行就起一个标准 tool-calling ReAct agent,**当下最主流首选**。注意:LangGraph v1.0(2025-10)后官方正逐步把它迁向 `langchain` 包里的 `create_agent`(带 middleware,更灵活),旧函数标了 deprecation。
- **`langchain` 旧版 `AgentExecutor` / `initialize_agent`** —— legacy,官方已建议改用上面的 `create_react_agent`,新项目别用。
- **`run-llama/llama_index`** —— `ReActAgent`(`llama_index.core.agent.workflow`),RAG/文档检索场景里最顺手,2026 仍活跃。
- **`huggingface/smolagents`**(pip `smolagents`)—— 极简(核心 ~1000 行)。其 `CodeAgent` 把 Action 写成 **Python 代码**而非 JSON(仍是 ReAct 骨架),`ToolCallingAgent` 才是经典 JSON 工具调用版;模型无关。

首选:做通用 agent 用 LangGraph(留意 `create_react_agent`→`create_agent` 迁移);偏检索用 LlamaIndex;想要极轻量/代码即动作用 smolagents。

## 工业界实践

ReAct 早已不是"论文范式"而是**几乎所有生产 agent 的默认骨架**——只是名字被各家系统隐去。一个生产级 tool-calling agent 的主循环,本质就是 ReAct:模型生成 `tool_calls` → runtime 执行 → 把 `tool` 角色的结果消息拼回 → 再次推理,直到模型不再发起调用而是直接出文本(等价于 `Finish`)。

**主流落地形态(具体名 + 定位)**:
- **OpenAI / Anthropic / Gemini 的原生工具调用 API**:把 ReAct 的"Action 解析"从正则换成结构化 `tool_calls` 字段,runtime 执行后用 `role:"tool"`(OpenAI)/ `tool_result` content block(Anthropic)回灌。**这是今天 ReAct 的事实标准实现**,见 [[15 Function Calling 工具调用|Function Calling 工具调用]]。
- **LangGraph `create_react_agent` / LangChain `create_agent`**:生产里最常见的"开箱即用" ReAct 编排,自带 checkpointer(可中断/恢复)、中间件(限流、护栏)。
- **Claude Code / Cursor / Devin 等代码 agent**:核心循环就是 ReAct——读文件(Observation)→ 想(Thought)→ 改/跑(Action)→ 看报错(Observation),见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]、[[23 Agent Harness 概览|Agent Harness 概览]]。
- **AWS Bedrock Agents / Azure AI Agent Service**:云厂商把 ReAct 循环 + 工具注册 + trace 托管化,企业少写胶水。

**规模化与成本/延迟**:ReAct 的成本模型是"每步重发全上下文",因此 token 随步数**超线性**膨胀(第 N 步 prompt 含前 N-1 步全部 Observation)。生产三板斧压成本:① **Prompt Caching**——把稳定的 system + 工具定义 + 历史前缀缓存,命中后这部分按 ~10% 计费,对长 trace 省得最狠,见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]];② **历史 Observation 压缩/卸载**——老的检索片段截断、摘要或写到外部存储,只留指针,见 [[21 上下文压缩与卸载|上下文压缩与卸载]];③ **小模型跑简单步、大模型跑难步**的分层路由。延迟上,单步串行往返是硬伤,无依赖的多工具调用要么用 provider 的 **parallel tool calls**(一轮里发多个 `tool_calls`),要么直接换 [[12 LLMCompiler|LLMCompiler]]。

**可观测与运维**:ReAct 的整条 Thought/Action/Observation **trace 就是天然的可观测对象**。生产标配 **LangSmith / Langfuse / Arize Phoenix / OpenLLMetry(OpenTelemetry GenAI 语义约定)** 把每一步记成 span:输入/输出 token、工具名、参数、耗时、报错。排障时直接看"哪一跳的 Observation 偏了 / 哪一步选错工具",见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**踩坑与最佳实践**:
- **必设 `max_steps` / `recursion_limit`**:防死循环烧钱,这是上线红线。
- **Observation 必须由代码真实执行后填**,绝不让模型自己续写(否则抗幻觉初衷尽失)。
- **工具结果要瘦身**:别把 50KB 的 API 原始 JSON 整个回灌,先抽字段/截断,否则上下文几步就爆。
- **结构化错误回灌**:工具报错时把错误信息当 Observation 喂回去让模型自纠,而不是直接抛异常终止。
- **超时/重试/限流**包在 runtime 层,模型不该感知这些基础设施细节。

```python
# 生产级 ReAct 主循环(Anthropic 风格,带护栏)
def agent_loop(messages, tools, max_steps=12):
    for _ in range(max_steps):
        resp = client.messages.create(model="...", tools=tools,
                                       messages=messages)   # 每步重发全上下文(命中 cache)
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:                       # 模型不再调工具 == Finish
            return resp
        results = []
        for tu in tool_uses:                    # 可并行执行
            try:
                out = run_tool(tu.name, tu.input)
            except Exception as e:
                out = f"ERROR: {e}"             # 错误也回灌,让模型自纠
            results.append({"type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": truncate(out)})  # 结果瘦身
        messages.append({"role": "user", "content": results})
    return "达到最大步数"                         # 护栏兜底
```

## 面试高频

**Q1:ReAct 到底解决了什么问题?为什么不直接用 CoT?**
标准答:CoT 只在参数内推理、拿不到外部新事实,且中间记错无从核对(推理性幻觉);Act-only 没有推理引导动作会盲目乱点。ReAct 让 Thought 给 Action 定方向、Observation 给 Thought 喂真实事实,**每隔一步就有一次真实观测校准**,既压幻觉又让过程可追溯。
- 追问"那为什么 Observation 能抗幻觉":因为它是外部环境的**真实返回**而非模型脑补,把模型从参数记忆拉回现实锚点。
- 陷阱:面试官可能问"ReAct 是不是就一定比 CoT 准"——不是,纯逻辑/数学等不碰外部信息的题,ReAct 反而多花 token,论文最强配置是 **ReAct + CoT-SC 互为补充**(能查就查、查不到回退内部推理)。

**Q2:ReAct 一轮的三段分别是什么?谁来控制循环?**
标准答:Thought(脑内推理,不碰外界)/ Action(结构化动作,如 `Search[q]` 或 `Finish[ans]`)/ Observation(外部真实返回)。循环由**外层 harness(程序)控制**,不是模型——程序解析 Action、真正调工具、回灌 Observation、设 `max_steps`;模型只负责每步生成 Thought+Action。
- 追问"`stop` 截断有什么用":让模型只生成到 Observation 之前,防止它自己伪造观测内容。
- 陷阱:很多人误以为"模型自己在跑循环"——错,模型每次只前进一步,是 harness 在驱动。

**Q3:ReAct 最大的代价是什么?怎么优化?**
标准答:**每步重发全上下文 → token 与延迟随步数膨胀**,长 trace 还易迷失目标。优化:Prompt Caching 缓存历史前缀、压缩/卸载老 Observation、小模型跑简单步;若步骤可预排就换 [[10 Plan-and-Execute|Plan-and-Execute]] / [[11 ReWOO|ReWOO]](减少模型介入次数),若有并行结构就换 [[12 LLMCompiler|LLMCompiler]]。
- 追问"为什么不能像 ReWOO 那样只调两次模型":因为 ReAct 面向**边走边看、下一步依赖上一步真实结果**的探索式任务,无法预先排好全程。

**Q4:原始 ReAct 用文本协议,今天怎么实现?**
标准答:今天几乎都用 [[15 Function Calling 工具调用|Function Calling 工具调用]],模型输出结构化 `tool_calls`、runtime 执行,**机制完全一致**,只是把"正则解析 Action 文本"换成"解析 tool_calls 字段",更鲁棒、不怕格式抖动。

**Q5:ReAct 会死循环吗?怎么防?**
标准答:会——模型反复查同一东西或绕圈。防:必设 `max_steps` 护栏 + 在 prompt 里鼓励"信息够了就 Finish" + 检测重复动作。这是上线必做项。

## 知识拓展

**谱系定位**:ReAct 是这条"减少大模型介入次数"主线的**起点**:[[09 ReAct|ReAct]](每步交错观测,最贵最灵活)→ [[10 Plan-and-Execute|Plan-and-Execute]](规划集中一次)→ [[11 ReWOO|ReWOO]](观测不回灌、固定调 2 次)→ [[12 LLMCompiler|LLMCompiler]](编译成 DAG 并行)。理解 ReAct 的"每步重发全上下文"为何贵,才懂后三者各自在省什么。

**向上的进阶方向**:
- **自我修正**:ReAct 单线前进、错了顺着错,叠加 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 让 agent 从失败 trace 里总结教训再重试。
- **多路径探索**:ReAct 是单线 DFS,[[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]] 把它扩成树——**LATS 本质就是给 ReAct 套上蒙特卡洛树搜索**,用价值函数评估不同行动分支。
- **训练而非提示**:[[32 Agentic RL 与训练|Agentic RL 与训练]] 直接用 RL 在多轮工具交互上训模型,让"该不该调工具、调哪个"成为学到的策略而非靠 prompt 约束。

**相关论文/前沿**:
- **Yao et al., _ReAct_(ICLR 2023)**——本范式原点。
- **Shinn et al., _Reflexion_(NeurIPS 2023)**——给 ReAct 加"语言化的失败反思 + 记忆",在 ReAct trace 上做迭代自改。
- **Schick et al., _Toolformer_(2023)**——从另一侧切入:不靠 prompt 约束,而是自监督地教模型**在哪插入 API 调用**,与 ReAct 的"靠提示驱动工具使用"形成对照。
- **ReAct 的"上下文滚雪球"问题**催生了上下文工程整条线,见 [[20 上下文工程|上下文工程]]、[[21 上下文压缩与卸载|上下文压缩与卸载]]。

**边界与反模式**:
- **反模式 1:对纯推理题硬套 ReAct**——无外部信息可查,白白多花 token 还引入工具失败面,直接 CoT 即可。
- **反模式 2:把 50KB 工具原始返回整个回灌**——几步就把上下文撑爆,且模型被噪声淹没。
- **反模式 3:让模型自己续写 Observation**——抗幻觉的根都拔了。
- **边界**:ReAct 假设"每步串行往返",对天然可并行的多工具任务(查 N 个实体再比较)是结构性浪费,该用 [[12 LLMCompiler|LLMCompiler]] 或 provider 的 parallel tool calls。
