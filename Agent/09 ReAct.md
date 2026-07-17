[[09 ReAct|ReAct]] 把「推理(Reason)」和「行动(Act)」交错进同一个循环:模型不再一口气想完再动手,而是在需要时决定下一步行动、执行工具、读取真实结果,再据此继续——用外部观测校准下一步决策。

它是 [[03 Agent 核心循环|Agent 核心循环]] 的经典具体形态;今天的结构化工具调用也常采用同构的「模型提议调用 → runtime 执行 → 结果回灌」控制环。

## 本质:为什么要把"想"和"做"焊在一起

在 ReAct 之前,LLM 解复杂任务有两条路,各有致命短板:

- **纯推理(Chain-of-Thought)**:让模型在脑内一步步想("先算 A,再算 B…")。问题是它**只在参数里取知识**,拿不到外部世界的新事实;一旦中间某步记错,后面整条链顺着错下去,且**无从核对**——这就是"推理性幻觉"。
- **纯行动(Act-only)**:让模型直接吐工具调用序列(search→click→…),不解释为什么。问题是它**没有推理来引导动作**,面对多跳、需要分解的任务就盲目乱点,也无法根据中途结果调整策略。

ReAct 的洞见:**这两者本该互补**。

- 推理/决策给行动**定方向**:决定下一步该查什么、该用哪个工具、信息够不够。
- 行动(Action)+ 观测(Observation)给推理**喂事实**:把外部世界的真实返回注回上下文,作为下一轮推理的锚点。

真实 Observation 能降低「把外部事实凭空补全」的风险,但不保证正确:工具选错、参数错、来源错或结果被误读仍会传播。可追溯性也不等于保存原始思维过程——生产系统应把可验证的行动和证据做成 trace。

**生活类比:侦探查案。**侦探不会先在办公室把整案想完,而是写一条可公开的下一步目的「核对车牌归属」,去查登记库(Action),拿到登记结果(Observation),再决定要不要查监控。案卷保存的是这条目的、查询记录、结果和证据链,而不是侦探脑中的逐字自言自语。

## 机制:论文中的 Thought → Action → Observation 循环

![[ReAct.png]]

原论文的一轮迭代分三段,循环往复直到模型主动收尾:

1. **Thought(推理轨迹)**:论文让模型输出可见的自然语言 reasoning trace——任务进展到哪、还缺什么、下一步打算干嘛。这是当时的研究 prompt 接口,**不等同于**任意模型的私有内部 CoT。
2. **Action(行动)**:模型从 Thought 收敛出一个**结构化动作**,通常形如 `工具名[参数]`,例如 `Search[Apple Remote]`、`Lookup[设备]`、或终止动作 `Finish[答案]`。
3. **Observation(观测)**:外部环境(检索器 / API / 代码执行器…)执行该 Action,把**真实返回**作为 Observation 拼回上下文。

然后这段 `Thought_i / Action_i / Observation_i` 会附加到 prompt 里,模型据此生成下一轮输出……如此滚动,直到某一轮 Action 是 `Finish[...]`,循环终止、产出答案。

控制循环的是**外层程序(harness)**,不是模型:程序负责解析模型输出里的 Action、真正去调工具、把 Observation 塞回去、并设最大步数防止死循环。模型每次只产生当前轮的决策/动作或收尾。

### 论文 trace 不等于生产日志

论文里的显式 Thought 适合解释 ReAct 的研究机制,但生产实现**不应要求、记录或向用户暴露原始 chain-of-thought**。模型提供商可能根本不返回它;即使某个模型返回,它也可能含有不可靠、敏感或不适合展示的内容。更稳妥的做法是记录可审计的外显事实:

- **简短决策摘要**:例如「为核对发布日期,下一步调用新闻检索」;它是面向人和运维的说明,不是正确性的证据。
- **结构化动作**:工具名、经脱敏的参数、调用 ID、权限/人审决定。
- **状态与结果**:计划/任务状态变更、工具返回的状态码、耗时、重试和错误。
- **证据**:检索到的来源标识、引用片段或数据版本,并把最终结论关联到证据。

把这些字段记为 span/trace,才能在不依赖原始 CoT 的前提下排查「选错工具、工具失败还是证据不足」,详见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。OpenAI 在 2024 年说明其推理模型不向用户展示原始 CoT,2025 年也明确建议应用不要直接展示 CoT;这里采用的是这一类面向用户的安全边界,而不是否认论文的显式 reasoning trace。

> 注意:原始 ReAct 用的是**文本协议**(模型吐 `Action: Search[...]` 字符串,靠正则解析)。许多现代实现改用 [[15 Function Calling 工具调用|Function Calling 工具调用]],让模型输出结构化 tool call、由 runtime 执行——控制环相同,只是把"解析 Action 文本"换成了结构化字段校验,通常更易做类型与权限检查。

### 一段真实 trace 长什么样

![[ReAct-trace示例.png]]

这是论文里的经典多跳例子:问"Apple Remote 最初设计来控制的程序,后来还能被哪些设备控制"。模型先 `Search[Apple Remote]` 拿到"控制 Front Row",再 `Search[Front Row]` 拿到"也可由键盘功能键控制",两条真实 Observation 集齐后才 `Finish`。它说明了 Observation 的价值:第二跳依据检索结果而非只依赖参数知识。

## 原论文

**Yao et al., _ReAct: Synergizing Reasoning and Acting in Language Models_**(2022 年 10 月 arXiv,**ICLR 2023** 接收)。作者来自 Princeton 与 Google Research(Shunyu Yao、Jeffrey Zhao、Dian Yu、Nan Du、Izhak Shafran、Karthik Narasimhan、Yuan Cao)。

论文在 HotpotQA、FEVER、ALFWorld、WebShop 上报告实验,核心主张是将 reasoning traces 与任务动作交错,使动作能获取外部信息、推理能更新行动计划。各基准、模型和提示设置下的数值不可直接外推到今天的生产系统;应在自己的工具、模型与风险约束下做评估。"ReAct"这个名字就是 **Reason + Act** 的合写。

## 可跑的最小实现:❌ 原始 Thought 入审计 vs ✅ 安全事件

下面用一个确定性脚本模拟「模型提议动作 → runtime 执行 → 结果回灌」。它不调用外部 API,可直接运行;真实系统只需将 `SAFE_TURNS` 换成经校验的结构化 tool call。

```python
CATALOG = {
    "Apple Remote": {
        "result": "最初用于控制 Front Row",
        "evidence": "catalog:apple-remote:v1",
    }
}

def lookup(query):
    record = CATALOG.get(query)
    if record is None:
        return {"ok": False, "result": None, "evidence": None,
                "error": f"未找到: {query}"}
    return {"ok": True, "result": record["result"],
            "evidence": record["evidence"], "error": None}

# ❌ 反例:把原始 Thought/CoT 当作审计字段;它既非必要证据,也不应持久化。
def naive_raw_thought_audit():
    raw_reply = "Thought: 先猜答案,再搜索确认。\nAction: Lookup[Apple Remote]"
    return {"audit": raw_reply}

# ✅ 示例输入只含可公开的目的与结构化动作,没有原始 CoT。
SAFE_TURNS = [
    {"decision_summary": "核对 Apple Remote 的最初用途",
     "tool_call": {"name": "Lookup", "args": {"query": "Apple Remote"}}},
    {"decision_summary": "证据已足够,结束任务",
     "tool_call": {"name": "Finish", "args": {"answer": "最初用于控制 Front Row"}}},
]

def safe_react_audit():
    state, audit = {"step": 0, "facts": {}}, []
    for turn in SAFE_TURNS:
        call = turn["tool_call"]
        if call["name"] == "Finish":
            return call["args"]["answer"], audit
        observation = lookup(call["args"]["query"])
        state["step"] += 1
        if observation["ok"]:
            state["facts"][call["args"]["query"]] = observation["result"]
        audit.append({
            "decision_summary": turn["decision_summary"],
            "tool_call": call,
            "state": {"step": state["step"], "facts": dict(state["facts"])},
            "result": observation["result"],
            "evidence": observation["evidence"],
            "error": observation["error"],
        })
    return "达到最大步数仍未完成", audit

bad = naive_raw_thought_audit()
answer, good = safe_react_audit()
assert "Thought:" in bad["audit"]
assert answer == "最初用于控制 Front Row"
assert set(good[0]) == {"decision_summary", "tool_call", "state", "result", "evidence", "error"}
print(answer, good[0]["evidence"])
```

要点:① Observation 只能由 `lookup` 这类 runtime 填入,不能让模型伪造;② 生产审计对象固定为**安全决策摘要 + 工具调用 + 状态/结果 + 证据/错误**,不含原始 Thought/CoT;③ `max_steps`、参数校验和权限检查仍应在外层 harness 实现。文本协议可用 `stop=["Observation:"]` 截断以防模型续写观测,生产环境宜改用 [[15 Function Calling 工具调用|Function Calling 工具调用]] 的结构化 tool call。

## 对比:ReAct 在 agent 谱系里的位置

| 范式 | 推理与行动关系 | 何时问大模型 | Token / 延迟 | 抗幻觉 | 适合 |
|---|---|---|---|---|---|
| 纯 CoT | 只推理,不行动 | 通常单次生成 | 取决于模型与推理长度 | 无外部事实校验 | 纯逻辑/数学,无需外部信息 |
| Act-only | 只行动,不推理 | 随动作轮数 | 取决于调用与工具延迟 | 取决于动作和校验 | 动作简单、无需分解 |
| **ReAct** | **每步交错** | **每一步** | 历史重发时会增长 | 有外部校验,仍受工具/证据质量约束 | 多跳、需边走边看的探索式任务 |
| [[10 Plan-and-Execute\|Plan-and-Execute]] | 先规划、通常顺序执行 | 初始规划 + 条件/批量 replan | 取决于 replan 频率、执行器与上下文策略 | 中 | 长程、步骤可预先排布 |
| [[11 ReWOO\|ReWOO]] | 规划/取证/合成三段解耦 | Plan+Solve(取决于实现) | 中间观测不回灌时可降低模型输入 | 蓝图错时中途纠偏较弱 | 步骤可静态规划、重视 token |

ReAct 的常见代价是:每轮都把运行历史重新提供给模型。若历史原样累积,**累计输入有二次项**、串行往返也会增加延迟;具体账单和上下文处理仍取决于缓存、摘要、外部状态和 provider。

**膨胀手算**。设每步新增约 200 token(一段安全决策摘要 + 一条 Observation),跑 10 步。为只看历史增长,先忽略固定系统提示。第 $i$ 步的 prompt 含前 $i-1$ 步历史,约 $200(i-1)$ token,送进模型处理的输入 token 累积为:

$$
\sum_{i=1}^{10} 200(i-1) = 200\times\frac{10\times 9}{2} = 200\times 45 = 9000 \text{ token}
$$

若每步只带当前 200 token(理想化的线性基线),只需 $10\times200=2000$ token。**9000 vs 2000,4.5 倍**——历史重发使累计输入成为 $O(N^2)$;固定系统提示和工具定义还会叠加在两边。缓存、压缩和外部状态可改变实际账单,因此这个例子说明的是增长形状,不是任何 provider 的报价。[[10 Plan-and-Execute|Plan-and-Execute]] 和 [[11 ReWOO|ReWOO]] 的目标之一,正是减少每步都需要大模型读取的上下文。

往上走,如果要让 agent **从失败中学习**就接 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]];如果要**探索多条推理/行动路径**而非单线前进,就上 [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]](LATS 本质就是给 ReAct 套上蒙特卡洛树搜索)。

## 何时用 / 坑

**该用 ReAct 的场景**:任务需要**边走边看**——下一步该干嘛严重依赖上一步的真实结果(多跳检索、网页浏览、需反复试探的环境交互、数据库探索)。这类任务无法预先排好完整计划,只能交错推进。

**不该用的场景**:步骤能**预先静态排布**的长流程(可评估 [[10 Plan-and-Execute|Plan-and-Execute]] 是否更合适);或纯内部推理、根本不碰外部信息的题(纯 CoT 可能更直接,无需强行套 ReAct)。

**常见坑**:
- **死循环 / 不收敛**:模型反复查同一个东西或绕圈。必须设 `max_steps`,并在 prompt 里鼓励"信息够了就 Finish"。
- **伪造 Observation**:不加 `stop` 截断,模型会自己脑补出 Observation 内容,彻底废掉抗幻觉的初衷。Observation **必须**由你的代码真实执行后填入。
- **上下文爆炸**:若每轮重发不断增长的历史,累计输入会按二次项增长;对策:压缩历史 Observation、外置状态,或按任务形态换 Plan-based 范式。
- **动作解析脆弱**:文本协议靠正则,模型一手抖格式就崩。生产请用 [[15 Function Calling 工具调用|Function Calling 工具调用]] 的结构化 tool call。
- **错误传播**:某步 Action 选错工具/错参数,后续会顺着错的 Observation 走偏;需要的话叠加 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 做自我修正。

## 关键事实

- ReAct = **Reason + Act**:论文将可见 reasoning trace 与工具动作交错;生产可改用短决策摘要,而不记录原始 CoT([Yao et al., ReAct, ICLR 2023](https://openreview.net/forum?id=WE_vluYUL-X);[OpenAI, Hiding the Chains of Thought, 2024](https://openai.com/index/learning-to-reason-with-llms/))。
- 三件套是**论文接口**的 Thought / Action / Observation;生产审计则记录安全决策摘要、脱敏工具调用、状态/结果、证据/错误,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]([Yao et al., 2023](https://openreview.net/forum?id=WE_vluYUL-X);[OpenAI gpt-oss 安全说明, 2025](https://openai.com/index/introducing-gpt-oss/))。
- 真实 Observation 提供外部校验,但不能保证正确;工具、参数、来源与结果解释仍需评估([Yao et al., 2023](https://openreview.net/forum?id=WE_vluYUL-X))。
- 循环由**外层 harness 控制**,模型每轮生成动作或收尾;`max_steps` 与结构化动作校验是必要护栏([Yao et al., 2023](https://openreview.net/forum?id=WE_vluYUL-X))。
- 若每步重发全上下文,输入量会按本文手算随步数二次增长;具体账单须实测([Yao et al., 2023](https://openreview.net/forum?id=WE_vluYUL-X))。
- 结构化 [[15 Function Calling 工具调用|Function Calling 工具调用]] 可替代文本正则解析;具体 API 随 provider 和版本变化([OpenAI, Responses API 新工具, 2025](https://openai.com/index/new-tools-and-features-in-the-responses-api/))。

## 框架落地

不要把 ReAct 绑定到某个框架函数名。选择框架时核对其当前官方文档是否支持:结构化工具调用、最大步数/递归限制、持久状态、超时与重试、脱敏 trace。框架 API 和弃用策略变化很快,应在集成时锁定版本并以该版本文档为准。

## 工业界实践

许多 tool-calling agent 可以用 ReAct 的控制环来理解:模型提议 `tool_calls` → runtime 执行 → 把结果回灌 → 模型继续或收尾。这是架构同构,不表示所有产品都采用论文的 prompt 或显式 Thought 字段。

**规模化与成本/延迟**:若每轮重发完整历史,成本随 trace 增长;缓存的折扣、是否可并发和模型路由都由 provider、模型、提示和工具延迟决定,没有可脱离工作负载的统一比例。可采取历史 Observation 压缩/卸载([[21 上下文压缩与卸载|上下文压缩与卸载]])、缓存稳定前缀、限制单次工具结果大小。只有调用之间确实无依赖时,才可考虑 provider 的并行工具调用或 [[12 LLMCompiler|LLMCompiler]] 的 DAG 调度。

**可观测与运维**:不要把原始 Thought/CoT 当作 trace 的必需字段。每个 span 至少记录**安全决策摘要、脱敏工具调用、状态/结果、证据/错误**(另可加 token/延迟);排障时据此定位「哪一跳的 Observation 不足、哪一步选错工具」,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**踩坑与最佳实践**:
- **必设 `max_steps` / `recursion_limit`**:防死循环烧钱,这是上线红线。
- **Observation 必须由代码真实执行后填**,绝不让模型自己续写(否则抗幻觉初衷尽失)。
- **工具结果要瘦身**:别把 50KB 的 API 原始 JSON 整个回灌,先抽字段/截断,否则上下文几步就爆。
- **结构化错误回灌**:工具报错时把错误信息当 Observation 喂回去让模型自纠,而不是直接抛异常终止。
- **超时/重试/限流**包在 runtime 层,模型不该感知这些基础设施细节。

上面的 ✅ 代码给出了可持久化审计事件的最小 schema。接入任一模型/工具 SDK 时,把原始响应限定在临时运行上下文;写入审计库前只投影为该 schema,并在 runtime 层追加 `max_steps`、超时、重试、权限和参数校验。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q1:ReAct 到底解决了什么问题?为什么不直接用 CoT?**
标准答:CoT 不会获得外部新事实,而 Act-only 缺少显式决策来引导动作。ReAct 让决策引导 Action、Observation 提供真实事实;这能降低凭空补全外部事实的风险,但工具、参数和来源仍要评估。
- 追问"那为什么 Observation 有帮助":因为它是外部环境的返回而非模型续写,把下一步决策锚定在可检查的结果上。
- 陷阱:面试官可能问"ReAct 是不是就一定比 CoT 准"——不是,纯逻辑/数学等不碰外部信息的题,ReAct 反而多花 token,论文最强配置是 **ReAct + CoT-SC 互为补充**(能查就查、查不到回退内部推理)。

**Q2:ReAct 一轮的三段分别是什么?谁来控制循环?**
标准答:论文写作 Thought / Action / Observation;生产可把 Thought 换成不含原始 CoT 的短决策摘要。循环由**外层 harness(程序)控制**,不是模型——程序执行 Action、回灌 Observation、设 `max_steps`;模型每轮只生成动作或收尾。
- 追问"`stop` 截断有什么用":让模型只生成到 Observation 之前,防止它自己伪造观测内容。
- 陷阱:很多人误以为"模型自己在跑循环"——错,模型每次只前进一步,是 harness 在驱动。

**Q3:ReAct 最大的代价是什么?怎么优化?**
标准答:**每步重发全上下文 → token 与延迟随步数膨胀**,长 trace 还易迷失目标。优化:Prompt Caching 缓存历史前缀、压缩/卸载老 Observation、小模型跑简单步;若步骤可预排就换 [[10 Plan-and-Execute|Plan-and-Execute]] / [[11 ReWOO|ReWOO]](减少模型介入次数),若有并行结构就换 [[12 LLMCompiler|LLMCompiler]]。
- 追问"为什么不能像 ReWOO 那样只调两次模型":因为 ReAct 面向**边走边看、下一步依赖上一步真实结果**的探索式任务,无法预先排好全程。

**Q4:原始 ReAct 用文本协议,今天怎么实现?**
标准答:可用 [[15 Function Calling 工具调用|Function Calling 工具调用]] 让模型输出结构化 `tool_calls`,由 runtime 执行并回灌结果;这保留控制环,同时避免文本正则解析的脆弱性。具体字段以所选 provider 的版本文档为准。

**Q5:ReAct 会死循环吗?怎么防?**
标准答:会——模型反复查同一东西或绕圈。防:必设 `max_steps` 护栏 + 在 prompt 里鼓励"信息够了就 Finish" + 检测重复动作。这是上线必做项。

## 知识拓展

**谱系定位**:ReAct、[[10 Plan-and-Execute|Plan-and-Execute]]、[[11 ReWOO|ReWOO]] 与 [[12 LLMCompiler|LLMCompiler]]是在「何时让模型介入、是否回灌观测、是否表达任务依赖」上的不同取舍;不能把它们简化为固定的成本/质量排序。

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
