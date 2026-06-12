[[04 Prompt Chaining|Prompt Chaining]] 是把一个任务**拆成固定的串行子步**,前一步的输出当后一步的输入,步与步之间还能插**程序化检查点(gate)**——这是 Anthropic《Building Effective Agents》(2024-12)归纳的最基础工作流模式。

## 本质:用「多步简单」换「一步复杂」
模型在单次调用里同时完成多件事,容易顾此失彼、出错率高。Prompt Chaining 把一个复杂任务**预先分解**成一条固定的步骤流水线:每一步只让 LLM 做一件简单、边界清晰的事,前一步的输出直接喂给下一步。

代价是**延迟变高**(多了几次 LLM 调用),换来的是**每步更准、更可控、更易调试**——你能精确知道是哪一步出了问题。这是典型的「准确度 ↔ 延迟」取舍。

它属于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里 **workflow 一侧**:步骤是开发者**预先写死**的编排,不是模型运行时自己决定的。与全自主、自己决定下一步的 [[09 ReAct|ReAct]] 形成鲜明对照。

## 机制:链 + gate,分步讲透
![[Prompt Chaining.png]]

1. **分解(设计期完成)**:开发者把任务切成 N 个有固定顺序的子步。切分的标准是——每个子步能用一个聚焦的 prompt 干净地完成,且后一步**真的依赖**前一步的产物。
2. **串行执行**:`out1 = LLM(prompt1, 输入)` → `out2 = LLM(prompt2, out1)` → … 前一步输出是后一步输入。
3. **插 gate(可选但关键)**:在某些步之间放一个**程序化检查**(不是 LLM,而是确定性代码):
   - 校验格式 / 字段是否齐全(如 JSON 能否解析、字数是否达标);
   - 不通过则**中止**整条链,或**回退**到上一步重试(避免错误顺流而下,污染后续所有步)。
   gate 是把「LLM 概率性输出」夹进「确定性护栏」的手段,也是这个模式比裸调用更可靠的核心原因。
4. **末步输出**:最后一步的产物即最终结果。

## 来源
出自 Anthropic 工程博客《Building Effective Agents》(2024-12)。文中给的两个典型例子:① 先生成营销文案 → gate 校验 → 再翻译成另一种语言;② 先写文档大纲 → gate 检查大纲是否符合某些标准 → 再据大纲写全文。

## 可跑最小代码(伪代码)
```python
def prompt_chain(topic):
    # Step 1: 生成大纲
    outline = llm(f"为主题《{topic}》生成 5 点文章大纲,每点一行")

    # --- Gate: 程序化检查,不过则中止/回退 ---
    points = [l for l in outline.splitlines() if l.strip()]
    if len(points) < 3:
        # 回退重试一次;真实系统里可循环或抛错
        outline = llm(f"为《{topic}》生成至少 5 点大纲,务必每点一行")
        points = [l for l in outline.splitlines() if l.strip()]
        if len(points) < 3:
            raise ValueError("Gate 未通过:大纲点数不足,中止链")

    # Step 2: 据大纲写正文(吃 Step 1 的输出)
    draft = llm(f"严格按以下大纲扩写成文章:\n{outline}")

    # Step 3: 润色输出
    final = llm(f"润色下文,使语气专业、去口水话:\n{draft}")
    return final
```
要点:gate 是**普通代码**,不是又一次 LLM 调用;它在链中间「卡关」,把错误挡在传播之前。

## 对比表
| 维度   | Prompt Chaining | [[05 Routing\|Routing]] | 裸单次调用  |
| ---- | --------------- | ----------------------- | ------ |
| 结构   | 固定串行多步          | 先分类再分流到一条支路             | 一步     |
| 步骤来源 | 设计期写死           | 设计期写死,运行时选支路            | 无      |
| 适用   | 任务能干净拆成可预定子步    | 输入有清晰类别                 | 任务足够简单 |
| 错误控制 | gate 逐步卡关       | router 选错则全错            | 难定位    |
| 延迟   | 高(N 次调用)        | 中                       | 低      |

## 何时用 / 坑
- **何时用**:任务能被**干净地拆成固定、可预定的子步**,且拆分后每步明显更简单。例如「生成→校验→翻译」「大纲→成文→润色」。
- **坑一**:别为了拆而拆。若子步之间没有真依赖、或拆完每步并没变简单,只是徒增延迟和故障点。
- **坑二**:gate 要放对地方——放在「错误一旦顺流而下就难救」的步之前最值。
- **坑三**:链越长,累积延迟和累积失败率越高;能合并的相邻步该合并。
- **坑四**:它是**静态**编排;如果子任务的数量/种类要到运行时才知道,那已经不是 Prompt Chaining,该看 [[07 Orchestrator-Workers|Orchestrator-Workers]]。

## 工业界实践

Prompt Chaining 是生产里**用得最多、最该先试**的模式——大量「看似需要 agent」的需求,其实一条带 gate 的链就稳稳搞定,成本、延迟、可调试性全甩 agent 几条街。

**框架里怎么落地:**

- **LangChain LCEL**:用管道符 `|` 把步骤串成链——`chain = prompt1 | llm | parser | prompt2 | llm`,天然表达「前一步输出喂后一步」,是 chaining 的教科书实现。
- **LangGraph**:把链建成一串顺序节点 + 条件边,gate 就是节点间的**条件边**(检查不过则路由到「重试」或「中止」节点),还自带持久化,某一步崩了能从该步续跑。
- **OpenAI Agents SDK 的 handoff / DSPy 的 Module 串联**:也都能表达固定串行流水线。结构化输出(JSON mode / structured outputs)让步与步之间的契约更稳——gate 直接校验 schema 而非靠正则。

**典型生产链路(对应 Anthropic 的两个官方例子):**
- 内容生产:`生成营销文案 → gate 校验(长度/敏感词/品牌词)→ 翻译成目标语言`。
- 文档生成:`先写大纲 → gate 检查大纲是否覆盖必需章节 → 据大纲扩写全文 → 润色`。

```python
# LCEL 风格:链 + 结构化输出做 gate(契约校验而非正则)
from pydantic import BaseModel
class Outline(BaseModel):
    points: list[str]

outline_chain = prompt_outline | llm.with_structured_output(Outline)  # Step1 输出受 schema 约束
outline = outline_chain.invoke({"topic": topic})

if len(outline.points) < 3:                 # ← Gate:确定性代码,不过则中止/回退
    raise GateFailed("大纲点数不足")

draft = (prompt_draft | llm).invoke({"outline": outline.points})       # Step2 吃 Step1 输出
final = (prompt_polish | llm).invoke({"draft": draft})                 # Step3
```

**规模化与成本/延迟权衡:**
- 链越长,**累积延迟** = Σ 各步延迟,**累积失败率** ≈ 1 − Π(各步成功率)——5 步每步 95% 成功,整链只剩约 77%。所以「能合并的相邻步该合并」「能并行的无依赖步拆去 [[06 Parallelization|Parallelization]]」。

**累积可靠性手算**。设每步独立成功率 $p=0.95$,整链可靠性 $P_N=p^N$,逐步算:

$$
P_3 = 0.95^3 = 0.857,\quad P_5 = 0.95^5 = 0.774,\quad P_8 = 0.95^8 = 0.663
$$

3 步还有 86%,5 步掉到 77%,**8 步只剩 66%**——三分之一的请求会在某一步翻车。这就是「链越长越要砍步数」的量化依据:每多挂一步,可靠性按 $\times 0.95$ 衰减,且单点提升步成功率($p$ 往 0.99 拉)比无脑加步划算得多。
- 成本可**精确预估**(步数固定),这是相对 agent 的核心优势——批量、预算敏感场景优先 chaining。
- **prompt 缓存**:链里靠前步骤的 system prompt / 长上下文若复用,开缓存能砍掉大量重复输入 token,见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。

**可观测与运维:** chaining 的可观测天然清晰——每一步是一个独立 span,失败时**精确知道是哪一步、什么输入**;这正是它比单次复杂 prompt 更易调试的工程价值。LangSmith / LangFuse 默认按链的步骤拆 trace。

**真实踩坑与最佳实践:**
- **为拆而拆**:子步之间没真依赖、或拆完每步并没变简单,只是徒增延迟和故障点。先问「拆完每步是否明显更易写好 prompt」。
- **gate 放错位置**:gate 要放在「错误一旦顺流而下就难救」的步**之前**最值;放末尾才查等于白拆。
- **gate 用 LLM 而非代码**:gate 的灵魂是**确定性**护栏;若用 LLM 当 gate,它本身就有概率性,失去了「把概率输出夹进确定性检查」的意义(那种「LLM 评估再改」的闭环是 [[08 Evaluator-Optimizer|Evaluator-Optimizer]],不是 gate)。

## 面试高频

**Q1:什么是 Prompt Chaining?它和「把几个 prompt 接起来」有什么区别?**
标准答:把任务拆成**固定串行子步**、前一步输出喂后一步,步间可插**程序化 gate(确定性代码检查)**。区别就在 gate——**把确定性护栏嵌进概率性流水线**,在错误传播前卡关;没有 gate 就只是裸接。
- 追问:**gate 为什么不能是 LLM?** 因为 gate 要确定性;LLM 当 gate 引入新的概率性,且变成评估闭环就是另一个模式了。

**Q2:Prompt Chaining 属于 workflow 还是 agent?为什么?**
标准答:**workflow**。步骤是开发者**设计期写死**的,不是模型运行时决定的——判据是控制权归代码,见 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]]。
- 陷阱:它是多步、调多次 LLM,但「多步 ≠ agent」,别被表象骗。

**Q3:Prompt Chaining 的核心取舍是什么?**
标准答:用**延迟变高**(多了几次 LLM 调用)换**每步更准、更可控、更易调试**(精确定位是哪步出错)。典型「准确度 ↔ 延迟」权衡。

**Q4:什么时候该用 Chaining,什么时候不该?**
标准答:任务能**干净拆成固定、可预定的子步**且拆后每步明显更简单 → 用;子步数量/种类要到运行时才知道 → 不该用(那是 [[07 Orchestrator-Workers|Orchestrator-Workers]]);要依据评估反复改同一产物 → 那是 [[08 Evaluator-Optimizer|Evaluator-Optimizer]]。

**Q5(进阶):一条 5 步的链,每步成功率 95%,整链可靠性大约多少?怎么提升?**
标准答:≈ 0.95^5 ≈ 77%。提升手段:合并能合并的相邻步(减少步数)、在关键步前插 gate 做重试/回退、把无依赖步拆去并行降低耦合。这考的是**累积失败率**意识。

## 知识拓展

- **官方出处与五件套**:出自 **Anthropic《Building Effective Agents》(2024-12)**,是它归纳的**最基础**工作流模式。五件套:本篇(串行)、[[05 Routing|Routing]](先分类再分流)、[[06 Parallelization|Parallelization]](并行无依赖)、[[07 Orchestrator-Workers|Orchestrator-Workers]](运行时动态分发子任务)、[[08 Evaluator-Optimizer|Evaluator-Optimizer]](生成-评估闭环)。
- **与各模式的边界(高频混淆点)**:
  - 与 [[06 Parallelization|Parallelization]] **正交**——Chaining 是串行依赖,Parallelization 是并行无依赖,真实系统常组合(链里某一步内部并行扇出)。
  - 与 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的分界是**静态 vs 动态**:子步数量设计期定死 = Chaining;运行时才由 LLM 决定派几个、派什么 = Orchestrator-Workers。
  - 与 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 的分界是**单向 vs 闭环**:Chaining 是单向流水线,gate 只卡关不迭代;Evaluator-Optimizer 是「生成→评估→改→再评估」的循环。
- **gate 的本质拔高**:gate = 把「LLM 概率性输出」夹进「确定性代码护栏」。这个思想贯穿整个 agent 工程——核心循环里 harness 的工具校验([[03 Agent 核心循环|Agent 核心循环]])、agent 的人工审核点、结构化输出 schema 校验,都是同一思想的变体。
- **反模式**:① 为拆而拆(无真依赖);② gate 放错位置(放末尾);③ gate 用 LLM(失去确定性);④ 链过长(累积延迟与失败率)。
- **深层联系**:Prompt Chaining 是从 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 走向自主 agent 之前**最该先穷尽**的一档——Anthropic 反复强调很多场景一条链就够,不必上 [[09 ReAct|ReAct]] 式自主回路。

## 关键事实
- 五件套里**最简单、最该先尝试**的模式;很多场景一条链就够,不必上 agent。
- gate 是它区别于「把几个 prompt 随便接起来」的本质:**确定性检查嵌入概率性流水线**。
- 与 [[06 Parallelization|Parallelization]] 正交:Chaining 是串行依赖,Parallelization 是并行无依赖,二者常组合使用。
- 想要的若是「依据评估反复改同一产物」,那是闭环的 [[08 Evaluator-Optimizer|Evaluator-Optimizer]],不是单向的 Chaining。
- 累积失败率意识:N 步每步 p 成功 → 整链 ≈ p^N;链越长越要合并/插 gate/并行化。
