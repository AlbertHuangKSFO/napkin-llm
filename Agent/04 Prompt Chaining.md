[[04 Prompt Chaining|Prompt Chaining]] 是把一个任务**拆成固定的串行子步**,前一步的输出当后一步的输入,步与步之间还能插**程序化检查点(gate)**——这是 Anthropic《Building Effective Agents》(2024-12)归纳的最基础工作流模式。

## 本质:用「多步简单」换「一步复杂」
模型在单次调用里同时完成多件事,容易顾此失彼、出错率高。Prompt Chaining 把一个复杂任务**预先分解**成一条固定的步骤流水线:每一步只让 LLM 做一件简单、边界清晰的事,前一步的输出直接喂给下一步。

代价是**延迟变高**(多了几次 LLM 调用),换来的是**每步更准、更可控、更易调试**——你能精确知道是哪一步出了问题。这是典型的「准确度 ↔ 延迟」取舍。

它属于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里 **workflow 一侧**:步骤是开发者**预先写死**的编排,不是模型运行时自己决定的。与全自主、自己决定下一步的 [[09 ReAct|ReAct]] 形成鲜明对照。

**生活类比:** 像餐厅后厨固定做「洗菜 → 切菜 → 烹饪 → 出餐」。切好的菜没洗净就不能进锅，这个质检点就是 gate；厨师不会边洗边临时决定把流程改成修空调。

## 机制:链 + gate,分步讲透
![[Prompt Chaining.png]]

1. **分解(设计期完成)**:开发者把任务切成 N 个有固定顺序的子步。切分的标准是——每个子步能用一个聚焦的 prompt 干净地完成,且后一步**真的依赖**前一步的产物。
2. **串行执行**:`out1 = LLM(prompt1, 输入)` → `out2 = LLM(prompt2, out1)` → … 前一步输出是后一步输入。
3. **插 gate(可选但关键)**:在某些步之间放一个**检查与决策点**:
   - **确定性 gate**:校验 JSON schema、必填字段、权限、测试结果等；适合能由代码精确判真的条件。
   - **语义 gate**:当条件本身是「是否答全、是否忠实原文、是否含有风险语义」时，可用有 rubric 的 LLM judge 输出 `pass / fail / escalate`。它是**概率性**分类器，必须在带人工标签的验证集上校准阈值，并把低置信或高风险样本升级；不能单独承担不可逆的安全控制。
   - 不通过可**中止**、**升级人工**，或回退到上一步重试，避免错误顺流而下。

   两者都能当链中的 gate；区别是确定性 gate 给强保证，语义 gate 给可度量但非绝对的证据。
4. **末步输出**:最后一步的产物即最终结果。

## 来源
出自 Anthropic 工程博客《Building Effective Agents》(2024-12)。文中给的两个典型例子:① 先生成营销文案 → gate 校验 → 再翻译成另一种语言;② 先写文档大纲 → gate 检查大纲是否符合某些标准 → 再据大纲写全文。

## 可跑最小代码
```python
# 这是可直接运行的本地模型替身；接入真实 LLM 时只替换 model()。
def model(stage, text):
    if stage == "one_shot":
        return f"《{text}》：一段未经检查的成文"
    if stage == "outline":
        return "1. 定义\n2. 例子\n3. 风险"
    if stage == "draft":
        return f"按大纲写《{text}》"
    if stage == "polish":
        return f"润色：{text}"
    raise ValueError(stage)

# ❌ 朴素写法：一次调用，无法在中间拦住不合格产物。
def one_shot(topic):
    return model("one_shot", topic)

# ✅ 改进写法：固定链 + 确定性 gate；后一步只吃已通过的前一步输出。
def chain_with_gate(topic):
    outline = model("outline", topic)
    points = [line for line in outline.splitlines() if line.strip()]
    if len(points) < 3:
        raise ValueError("gate: 大纲点数不足")
    draft = model("draft", outline)
    return model("polish", draft)

plain = one_shot("Prompt Chaining")
checked = chain_with_gate("Prompt Chaining")
assert "未经检查" in plain and checked.startswith("润色：")
print(plain)
print(checked)
```
要点:gate 的职责是**决定放行、拦截或升级**。它通常由普通代码实现；当规则无法表达语义时，也可调用经校准的 LLM judge，但要把它当成有误差的分类器，而不是事实裁判。

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

Prompt Chaining 常是值得先验证的基线：Anthropic 建议先采用简单、可组合的模式，只有在任务需要时才增加复杂度。它并不必然胜过 agent；是否更好仍要在目标质量、延迟、成本与风险约束下实测。

**框架里怎么落地:**

- **LangChain LCEL**:用管道符 `|` 把步骤串成链——`chain = prompt1 | llm | parser | prompt2 | llm`,天然表达「前一步输出喂后一步」,是 chaining 的教科书实现。
- **LangGraph**:把链建成一串顺序节点 + 条件边,gate 就是节点间的**条件边**(检查不过则路由到「重试」或「中止」节点),还自带持久化,某一步崩了能从该步续跑。
- 无论使用何种 SDK，固定链都可用普通顺序控制流表达；结构化输出能让步间契约更明确，使 gate 可以直接校验 schema 而非依赖脆弱的文本约定。

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

**语义 gate 的最小契约。** 让 judge 只负责决策，不让它悄悄改稿；这里的 `score` 和阈值须先用已标注样本校准，`uncertain` 一律不自动放行。

```python
decision, score = semantic_judge(
    draft, rubric="是否覆盖全部必需章节？", labels=["pass", "fail", "uncertain"]
)
if decision == "uncertain" or score < CALIBRATED_THRESHOLD:
    return escalate_to_human(draft)
if decision == "fail":
    return retry_outline()
```

这仍是 Chaining，因为 gate 只控制**是否向前走/回退/升级**。若 judge 给出批评、生成器依批评改稿、再由 judge 重评直到停止，才是 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 的反馈闭环。

**规模化与成本/延迟权衡:**
- 链越长,无重试且各步结果近似独立时，**累积延迟**为 $\sum_i t_i$，整链成功率为 $\prod_i p_i$。因此能合并的相邻步应合并；无依赖步可拆去 [[06 Parallelization|Parallelization]]。真实 LLM 步骤往往相关，下面的独立性手算只能作量级直觉，不能当线上 SLA。

**累积可靠性手算**。设每步独立成功率 $p=0.95$,整链可靠性 $P_N=p^N$,逐步算:

$$
P_3 = 0.95^3 = 0.857,\quad P_5 = 0.95^5 = 0.774,\quad P_8 = 0.95^8 = 0.663
$$

在上述独立、无恢复的假设下，3 步约 86%、5 步约 77%、8 步约 66%。它说明链条变长会放大单步风险；实际设计还应比较重试、gate、相关性和任务收益，不能只凭这个玩具数字决定架构。
- 成本可**精确预估**(步数固定),这是相对 agent 的核心优势——批量、预算敏感场景优先 chaining。
- **prompt 缓存**:链里靠前步骤的 system prompt / 长上下文若复用,开缓存能砍掉大量重复输入 token,见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。

**可观测与运维:** chaining 的可观测天然清晰——每一步是一个独立 span,失败时**精确知道是哪一步、什么输入**;这正是它比单次复杂 prompt 更易调试的工程价值。LangSmith / LangFuse 默认按链的步骤拆 trace。

**真实踩坑与最佳实践:**
- **为拆而拆**:子步之间没真依赖、或拆完每步并没变简单,只是徒增延迟和故障点。先问「拆完每步是否明显更易写好 prompt」。
- **gate 放错位置**:gate 要放在「错误一旦顺流而下就难救」的步**之前**最值;放末尾才查等于白拆。
- **把所有 gate 都交给 LLM**:schema、权限、测试等可精确判断的约束应仍由代码执行。只有语义条件难以规则化时才用 LLM judge，并用标注集校准、低置信升级。LLM gate 不会自动变成 [[08 Evaluator-Optimizer|Evaluator-Optimizer]]；只有「反馈→改写→再评估」的闭环才会。

## 面试高频

**Q1:什么是 Prompt Chaining?它和「把几个 prompt 接起来」有什么区别?**
标准答:把任务拆成**固定串行子步**、前一步输出喂后一步，步间可插决定放行/拦截/升级的 gate。确定性约束用代码；语义约束可用经过校准的 LLM judge，但要承认其误差并设置升级路径。区别不在「必须有 gate」，而在每一步的输入输出契约和可观测的串行依赖。
- 追问:**LLM gate 会不会就是 Evaluator-Optimizer?** 不会。LLM gate 若只输出放行/拦截/升级，仍是链中的一个决策点；只有把可执行反馈喂回生成器并反复重评，才进入闭环。

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
  - 与 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 的分界是**单向控制流 vs 反馈闭环**:Chaining 的 gate 可由代码或 LLM 实现，但只决定放行、回退或升级；Evaluator-Optimizer 会把评价反馈用于改写，并重新评价同一产物。
- **gate 的本质拔高**:gate 是把输出变成显式决策点。可验证约束应使用确定性代码；语义判断可使用经校准的 LLM judge；高风险或低置信结果要保留人工复核。核心循环里的 harness 工具校验([[03 Agent 核心循环|Agent 核心循环]])、结构化输出 schema 校验同属这一思想。
- **反模式**:① 为拆而拆(无真依赖);② gate 放错位置(放末尾);③ 用未校准 LLM gate 代替安全控制；④ 链过长却不量测延迟与失败模式。
- **深层联系**:对于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 中可预先拆分的任务，Prompt Chaining 是比 [[09 ReAct|ReAct]] 式自主回路更易控制的候选基线；复杂度是否足够应由评测决定。

## 关键事实
- Anthropic 建议先选择能解决问题的最简单方案；对可预先拆分任务，链式工作流常是可评测的起点，而非必然最优解。
- Anthropic 的原始模式允许在中间步骤加入 programmatic checks；工程上应优先用确定性 gate，语义 gate 则需要 rubric、人工标注集校准和升级路径。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)；[Anthropic Agent Evals, 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)。
- 与 [[06 Parallelization|Parallelization]] 正交:Chaining 是串行依赖,Parallelization 是并行无依赖,二者常组合使用。
- 想要的若是「依据评估反复改同一产物」,那是闭环的 [[08 Evaluator-Optimizer|Evaluator-Optimizer]],不是单向的 Chaining。
- 累积失败率意识:N 步每步 p 成功 → 整链 ≈ p^N;链越长越要合并/插 gate/并行化。
