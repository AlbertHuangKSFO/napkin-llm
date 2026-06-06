[[08 Evaluator-Optimizer|Evaluator-Optimizer]] 是**一个生成器产出结果、一个评估器给反馈、生成器据反馈再改,循环到达标**——是 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 在 workflow 层的对应。这是 Anthropic《Building Effective Agents》(2024-12)五件套之一。

## 本质:把「改稿」做成闭环
人写东西很少一遍过:先写草稿,有人挑毛病,再据意见改,反复几轮才好。Evaluator-Optimizer 就是把这个**生成—批改—再生成**的循环搬给 LLM:

- **Generator(生成器/优化者)**:产出草稿;拿到反馈后**据反馈改写**。
- **Evaluator(评估器)**:按**明确标准**给草稿打分,并给出**具体、可执行的改进反馈**(不是只说「不够好」,而是说「哪里不好、怎么改」)。

两者往复**循环**,直到评估器判定达标(或到迭代上限)。这与单向、一步到底的 [[04 Prompt Chaining|Prompt Chaining]] 根本不同——那是流水线,这是**闭环**。

它属于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里的 **workflow 一侧**:循环结构是预设的,生成与评估的角色固定,与全自主的 [[09 ReAct|ReAct]] 对照。但它已带「自我改进」的味道,是 workflow 里最像 agent 的形态之一。

## 机制:生成 ↔ 评估闭环
![[Evaluator-Optimizer.svg]]

1. **生成草稿**:Generator 据输入产出第一版。
2. **评估**:Evaluator 按既定标准审草稿——打分 + 指出**具体问题**(措辞生硬?漏了要点?事实错误?)。反馈越具体,下一轮改得越准。
3. **判定**:
   - **达标** → 输出,结束;
   - **不达标** → 把反馈**连同上一版**喂回 Generator,回到第 1 步改写。
4. **循环**:重复直到达标,或触发**迭代上限/无进展**的停止条件(防死循环)。

灵不灵的两个前提:① 评价标准**明确**(评估器能稳定判好坏);② 反馈**确实能让下一轮更好**(迭代真有边际收益)。两者缺一,循环就只是空转烧钱。

## 来源
出自 Anthropic《Building Effective Agents》(2024-12)。文中给的典型场景:**文学翻译**——译完由评估器指出哪些地方语气/细微含义不对,生成器据此重译,迭代提质;**复杂搜索**——一轮搜索后评估「信息够不够、要不要换查询/补来源」,据反馈再搜。文中强调此模式适合「有明确评价标准 + 迭代确能带来可观提升」的任务。

## 可跑最小代码(伪代码)
```python
def evaluator_optimizer(task, max_iter=4):
    draft = llm(f"完成任务:\n{task}")          # 首版
    for _ in range(max_iter):
        # 评估器:打分 + 给具体反馈
        verdict = llm(
            f"按标准评估下面结果,先输出 PASS 或 FAIL,"
            f"若 FAIL 再列出具体、可执行的改进点:\n任务:{task}\n结果:{draft}"
        )
        if verdict.startswith("PASS"):
            return draft                        # 达标,出结果
        # 生成器:带着上一版 + 反馈改写
        draft = llm(
            f"根据反馈改进结果。\n任务:{task}\n上一版:{draft}\n反馈:{verdict}"
        )
    return draft                                # 到上限也返回当前最好版
```
要点:① Generator 改写时**要同时拿到上一版和反馈**,否则丢失上下文;② 必须有**迭代上限**兜底,否则评估器永远不满意会死循环。

## 对比表
| 维度 | Evaluator-Optimizer | [[04 Prompt Chaining|Prompt Chaining]] | [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] |
|---|---|---|---|
| 结构 | 生成↔评估**闭环** | 单向串行 | 自主推理中的自省循环 |
| 评估者 | **独立**的 evaluator LLM | gate(确定性代码) | agent 自己反思 |
| 反馈用途 | 喂回改写,迭代提质 | 通过/中止,不改写 | 修正下一步行动 |
| 所属 | workflow(预设循环) | workflow(预设流水线) | agent(自主架构) |
| 停止 | 达标 / 迭代上限 | 走完所有步 | 任务完成 / 上限 |

## 何时用 / 坑
- **何时用**:① 有**明确的评价标准**(评估器能稳定判好坏);② **迭代确能提升**(改了真的更好)。典型:翻译润色、文案打磨、复杂搜索、代码按测试反馈迭代。
- **坑一**:评价标准模糊时别用。评估器若判得不稳,反馈忽左忽右,生成器越改越乱。
- **坑二**:必须设**迭代上限 / 无进展早停**,否则评估器「永远挑得出毛病」会死循环烧钱。
- **坑三**:反馈要**具体可执行**。评估器只说「不够好」等于没说;prompt 里强制它列「哪里、为什么、怎么改」。
- **坑四**:评估器和生成器最好是**两个独立角色/调用**(甚至不同模型),避免「自己夸自己」的同质化偏差。
- **坑五**:迭代有边际递减;前两轮提升最大,后面常趋平,别盲目堆轮数。

## 关键事实
- 这是 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 在 **workflow 层的对应**:同样是「产出→自评→改进」,但这里角色与循环是**预设编排**,而 Reflexion 是 agent 在自主推理中自发反思。两者一脉相承,层级不同。
- 与 [[04 Prompt Chaining|Prompt Chaining]] 的关键区别:Chaining 的 gate 只**放行/拦截**,不喂反馈改写;这里评估器**给反馈、驱动重做**,是闭环不是流水线。
- 评估器本身就是一个微型 [[38 Agent 评估与可观测性|评估]] 系统——「能自动判好坏」是它的前提,也是把它工程化的难点。
- 实务里常与 [[07 Orchestrator-Workers|Orchestrator-Workers]]、[[04 Prompt Chaining|Prompt Chaining]] 组合:某一步的产物用 Evaluator-Optimizer 打磨到达标,再进入下一步。

## 工业界实践
这个模式的工业化核心是一句话:**evaluator 就是 LLM-as-judge**。整个模式能不能用,取决于你能不能造出一个稳定的「自动判好坏」的评估器。

**主流框架与服务(具体名 + 定位):**
- **LLM-as-judge 评估器**:**OpenAI Evals**、**LangSmith Evaluators**、**Ragas**(RAG 专用)、**DeepEval**、**Promptfoo**、**Evidently** 都提供「用 LLM 给输出打分 + 给反馈」的现成评估器,是 evaluator 那一半的事实标准工具。
- **闭环编排**:**LangGraph** 用带条件边的图天然表达「评估不通过就回到生成节点」的循环;Spring AI 等也有「Recursive Advisor」式的自动重评-重试(2025)。
- **可验证场景的硬 evaluator**:代码 Agent 里 evaluator 常常**不是 LLM 而是测试套件 / 编译器 / linter**——跑测试,失败就把报错喂回生成器重写。这是 SWE-bench 类系统的标准闭环,比 LLM judge 更可靠(对错是客观的),见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]。

**典型架构:** Generator(强模型)→ Evaluator(LLM judge 或测试/规则)→ 条件分支:PASS 出结果,FAIL 把【反馈 + 上一版】喂回 Generator → 循环,到迭代上限或无进展早停。

**规模化与成本/延迟:** 每轮是 2 次调用(生成 + 评估),轮数 × 2 是成本上界,所以**迭代上限**不只是防死循环,更是成本闸门。实务经验:**前两轮提升最大,之后边际递减**,大多数任务设 `max_iter=2~3` 就够;再多基本是烧钱。延迟也随轮数线性涨,对实时场景要么限轮数、要么离线/异步跑(如夜间批量润色)。

**可观测与运维:** 必须记录**每一轮的草稿、评估分数、反馈、是否 PASS**,这样能看「质量曲线是否真在涨」——若曲线早早走平甚至抖动,说明评估不稳或迭代无收益,该停。要监控**平均迭代轮数**(突然变多=评估器变挑剔或任务变难)和**「永不达标」率**。

**踩坑与最佳实践:**
- **evaluator 和 generator 用不同角色/调用(最好不同模型)**,避免「自己夸自己」的同质化偏差(self-enhancement bias)——这是 LLM-as-judge 的已知系统性偏差之一。
- 强制评估器输出**结构化、可执行反馈**(哪里、为什么、怎么改),只说「不够好」等于没说。
- 给评估器**明确 rubric + 打分锚点**(1-4 分各代表什么),否则判得忽左忽右。
- 防 LLM judge 的已知偏差:位置偏差(偏好靠前选项)、长度偏差(偏好更长)、近因偏差(偏好「新」回答)——用固定顺序、控制长度、双向比对来缓解。
- 生成器改写时**必须同时拿到上一版 + 反馈**,否则丢上下文从头乱改。

```python
# 生产化:judge 与 generator 分离 + rubric + 上限 + 无进展早停
def evaluator_optimizer(task, max_iter=3):
    draft, best, best_score = llm(task, model=GEN), None, -1
    for _ in range(max_iter):
        score, feedback = judge(task, draft, rubric=RUBRIC, model=JUDGE)  # 不同模型
        log_iter(draft, score, feedback)                                  # 记质量曲线
        if score >= PASS or score <= best_score:        # 达标 或 无进展 → 早停
            return best if score <= best_score else draft
        best, best_score = draft, score
        draft = llm(f"按反馈改进。\n任务:{task}\n上版:{draft}\n反馈:{feedback}",
                    model=GEN)                            # 改写带上一版+反馈
    return draft
```

## 面试高频
**Q1:Evaluator-Optimizer 和 Reflexion 是什么关系?**
标准答:Evaluator-Optimizer 是 [[13 Reflection 与 Reflexion|Reflexion]] 在 **workflow 层的对应**——都是「产出→自评→改进」,但前者角色(生成器/评估器)和循环是**预设编排**(workflow),后者是 agent 在**自主推理中自发反思**(agent 架构)。一脉相承,层级不同。
- 追问区别细节:Evaluator-Optimizer 的评估器是**独立角色/调用**;Reflexion 是 agent **自己**反思自己的轨迹。

**Q2:它和 Prompt Chaining 都有多步,本质区别?**
标准答:Chaining 是**单向流水线**,中间的 gate 只**放行/拦截**不喂反馈、不回头改写;Evaluator-Optimizer 是**闭环**,评估器给反馈、驱动生成器**重做同一产物**,直到达标。一个是流水线(往前走),一个是改稿循环(原地打磨)。

**Q3:这个模式成立的两个前提是什么?缺了会怎样?**
标准答:① 评价标准**明确**(评估器能稳定判好坏);② 迭代**确有边际收益**(改了真的更好)。缺① → 评估忽左忽右,生成器越改越乱;缺② → 循环空转烧钱。两者缺一就别用这个模式。

**Q4:为什么 evaluator 和 generator 最好分开,甚至用不同模型?**
标准答:避免 **self-enhancement bias**——同一个模型/调用既当运动员又当裁判,倾向于给自己的输出打高分,反馈失真。分开(尤其异构模型)能引入外部视角,评估更客观。

**Q5:怎么防死循环?**
标准答:两道闸——**迭代上限**(`max_iter`)兜底,加**无进展早停**(分数不再上升就停,返回历史最佳版)。否则评估器「永远挑得出毛病」会无限烧钱。

**陷阱题:「让 LLM 写完自己检查一遍再改」是不是 Evaluator-Optimizer?** → 看角色是否分离。若生成和评估是**两个独立调用/角色**且有**明确循环结构**,是;若只是同一次推理里顺带自查,那更接近 self-refine / Reflexion 的味道。边界确实模糊,关键看「评估是否被独立编排出来」。

## 知识拓展
- **同族方法谱系:** **Self-Refine**(NeurIPS 2023/2024)是单模型版的生成-自反馈-改进;**Reflexion**(2023)把反思写进记忆指导下一次尝试;Evaluator-Optimizer 把评估器**独立编排**出来。三者共享「产出→评→改」内核,差别在评估者是谁、循环是预设还是自主,见 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]]。
- **前沿(2024-2025):** LLM-as-judge 已成独立研究领域,有多篇综述(《A Survey on LLM-as-a-Judge》2025;《From Generation to Judgment》2025)。重点议题是 judge 的**偏差与鲁棒性**——位置/长度/近因偏差、来源等级偏差(EXPERT>HUMAN>LLM>UNKNOWN)、对抗扰动(《The Silent Judge》arXiv 2509.26072)。生产部署 judge 前务必先评估 judge 本身的可靠性(与人工标注对齐度)。
- **从「LLM judge」到「verifier」:** 在可验证任务(代码、数学)里,把 evaluator 换成**客观验证器**(测试套件、形式化检查、reward model)更可靠,且这种「生成-验证」闭环正是 [[32 Agentic RL 与训练|Agentic RL]] 的数据引擎——验证信号可直接当训练奖励。
- **边界与反模式:** ① 评价标准模糊还硬用 → 反馈抖动,越改越乱;② 不设上限/早停 → 死循环烧钱;③ 同模型自评自改 → self-enhancement bias;④ 盲目堆轮数 → 边际递减,前两轮之后多半白花;⑤ 反馈只说「不够好」→ 等于没反馈,必须强制具体可执行。
- **与兄弟模式:** 常作为「打磨器」嵌进其他模式——[[07 Orchestrator-Workers|Orchestrator-Workers]] 综合后用它把交付物打磨到达标;[[04 Prompt Chaining|Prompt Chaining]] 某一步产物用它迭代提质再进下一步;[[10 Plan-and-Execute|Plan-and-Execute]] 里也常在每步后插一道评估闸。评估器的工程化(怎么自动、稳定判好坏)本身就是 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 的核心课题。
