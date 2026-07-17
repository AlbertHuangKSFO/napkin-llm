[[08 Evaluator-Optimizer|Evaluator-Optimizer]] 是**一个生成器产出结果、一个评估器给反馈、生成器据反馈再改,循环到达标**——是 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 在 workflow 层的对应。这是 Anthropic《Building Effective Agents》(2024-12)五件套之一。

## 本质:把「改稿」做成闭环
人写东西很少一遍过:先写草稿,有人挑毛病,再据意见改,反复几轮才好。Evaluator-Optimizer 就是把这个**生成—批改—再生成**的循环搬给 LLM:

- **Generator(生成器/优化者)**:产出草稿;拿到反馈后**据反馈改写**。
- **Evaluator(评估器)**:按**明确标准**给草稿打分,并给出**具体、可执行的改进反馈**(不是只说「不够好」,而是说「哪里不好、怎么改」)。

两者往复**循环**,直到评估器判定达标(或到迭代上限)。这与单向、一步到底的 [[04 Prompt Chaining|Prompt Chaining]] 根本不同——那是流水线,这是**闭环**。

它属于 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 里的 **workflow 一侧**:循环结构是预设的,生成与评估的角色固定,与全自主的 [[09 ReAct|ReAct]] 对照。但它已带「自我改进」的味道,是 workflow 里最像 agent 的形态之一。

## 机制:生成 ↔ 评估闭环
![[Evaluator-Optimizer.png]]

1. **生成草稿**:Generator 据输入产出第一版。
2. **评估**:Evaluator 按既定标准审草稿——打分 + 指出**具体问题**(措辞生硬?漏了要点?事实错误?)。反馈越具体,下一轮改得越准。
3. **判定**:
   - **达标** → 输出,结束;
   - **不达标** → 把反馈**连同上一版**喂回 Generator,回到第 1 步改写。
4. **循环**:重复直到达标,或触发**迭代上限/无进展**的停止条件(防死循环)。

灵不灵的两个前提:① 评价标准**明确**(评估器能稳定判好坏);② 反馈**确实能让下一轮更好**(迭代真有边际收益)。两者缺一,循环就只是空转烧钱。

## 来源
出自 Anthropic《Building Effective Agents》(2024-12)。文中给的典型场景:**文学翻译**——译完由评估器指出哪些地方语气/细微含义不对,生成器据此重译,迭代提质;**复杂搜索**——一轮搜索后评估「信息够不够、要不要换查询/补来源」,据反馈再搜。文中强调此模式适合「有明确评价标准 + 迭代确能带来可观提升」的任务。

## 可跑最小代码
```python
def generate(task, previous_draft="", feedback=""):
    """首稿没有上版；改写必须同时消费上版与评估反馈。"""
    if not previous_draft:
        return f"{task}：只有概述"
    revisions = {
        (f"{task}：只有概述", "补上事实"): f"{task}：概述 + 事实",
        (f"{task}：概述 + 事实", "删去重复"): f"{task}：事实（但漏了结论）",
    }
    return revisions[(previous_draft, feedback)]

def judge(draft):
    scores = {"只有概述": 6.2, "概述 + 事实": 7.4, "漏了结论": 7.1}
    score = next(value for key, value in scores.items() if key in draft)
    feedback = {6.2: "补上事实", 7.4: "删去重复", 7.1: "补上结论"}[score]
    return score, feedback

# ❌ 朴素写法：只产出首稿，不知道它有没有达到 rubric。
def one_shot(task):
    return generate(task)

# ✅ 改进写法：评估反馈驱动改写，并保留历史最佳而非盲信末轮。
def evaluator_optimizer(task, max_iter=3, pass_score=8.0):
    draft = generate(task)
    best_draft, best_score = draft, float("-inf")
    previous_score, terminal_status = None, "max_iter"
    for _ in range(max_iter):
        score, feedback = judge(draft)
        if score > best_score:
            best_draft, best_score = draft, score
        if score >= pass_score:
            terminal_status = "passed"
            break
        if previous_score is not None and score <= previous_score:
            terminal_status = "no_progress"
            break
        previous_score = score
        draft = generate(task, draft, feedback)
    return {"draft": best_draft, "score": best_score,
            "terminal_status": terminal_status}

assert one_shot("报告").endswith("只有概述")
result = evaluator_optimizer("报告")
assert result == {"draft": "报告：概述 + 事实", "score": 7.4,
                  "terminal_status": "no_progress"}
print(result)
```
要点:① Generator 改写时**要同时拿到上一版和反馈**；② 必须有**迭代上限**与停止状态；③ 返回的是历史最佳 `best_draft`，**末轮 draft 不保证最好**。

## 对比表
| 维度 | Evaluator-Optimizer | [[04 Prompt Chaining\|Prompt Chaining]] | [[13 Reflection 与 Reflexion\|Reflection 与 Reflexion]] |
|---|---|---|---|
| 结构 | 生成↔评估**闭环** | 单向串行 | 自主推理中的自省循环 |
| 评估者 | 独立调用的 LLM judge、规则或 verifier | gate（确定性或经校准的语义判断） | agent 自己反思 |
| 反馈用途 | 喂回改写,迭代提质 | 放行/拦截/升级，不形成改稿闭环 | 修正下一步行动 |
| 所属 | workflow(预设循环) | workflow(预设流水线) | agent(自主架构) |
| 停止 | 达标 / 迭代上限 | 走完所有步 | 任务完成 / 上限 |

## 何时用 / 坑
- **何时用**:① 有**明确的评价标准**(评估器能稳定判好坏);② **迭代确能提升**(改了真的更好)。典型:翻译润色、文案打磨、复杂搜索、代码按测试反馈迭代。
- **坑一**:评价标准模糊时别用。评估器若判得不稳,反馈忽左忽右,生成器越改越乱。
- **坑二**:必须设**迭代上限 / 无进展早停**,否则评估器「永远挑得出毛病」会死循环烧钱。
- **坑三**:反馈要**具体可执行**。评估器只说「不够好」等于没说;prompt 里强制它列「哪里、为什么、怎么改」。
- **坑四**:评估器和生成器应至少是**独立调用与独立 rubric**；是否使用不同模型是可验证的设计选择，不是自动保证。要对比它们与人工标注或客观 verifier 的一致性。
- **坑五**:别假设每轮都会提高。记录每轮分数并在无进展、预算耗尽或达到门槛时停止，返回历史最佳，而不是盲信最后一轮。

## 关键事实
- 这是 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 在 **workflow 层的对应**：生成、评估、停止条件预设编排；Reflexion 更强调 agent 在自主轨迹中的反思。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- 与 [[04 Prompt Chaining|Prompt Chaining]] 的关键区别：Chaining 的 gate 只决定放行、拦截或升级；这里评估反馈会驱动重做同一产物并重新评价。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- evaluator 是一个微型 [[38 Agent 评估与可观测性|评估]] 系统；能稳定判好坏是前提，LLM judge 需与人工或客观 verifier 校准。[Anthropic, 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- 可把 Evaluator-Optimizer 嵌入 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的综合后步骤，或 [[04 Prompt Chaining|Prompt Chaining]] 的某一产物步骤；组合效果应在任务集上验证。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)

## 小数字手算与公式推导

第 $k$ 轮草稿为 $d_k$，评估分数为 $s_k$。为了不把「最后一轮」误当成「最好一轮」，应显式维护

$$
best\_score_k=\max(s_0,s_1,\ldots,s_k),
\qquad
best\_draft_k=d_{\arg\max_{0\le i\le k}s_i}.
$$

例如通过线为 8 分，三轮的分数为 $s_0=6.2$、$s_1=7.4$、$s_2=7.1$。逐轮更新：

$$
(best\_score,best\_draft):
(-\infty,\varnothing)\to(6.2,d_0)\to(7.4,d_1)\to(7.4,d_1).
$$

第三轮分数下降且没达标，所以可设 `terminal_status="no_progress"`；交付 $d_1$ 而非末轮 $d_2$。分数来自 LLM judge 时仍有噪声，这个计算保证的是**按既定 judge 保留最优**，不等于客观真质量必然最高。

## 工业界实践
这个模式的工业化核心是**可用的反馈信号**，不等同于“一定用 LLM-as-judge”。有客观判据时优先用测试、编译器、schema 或业务规则；自由文本等难以程序化判定的任务可使用 LLM judge，但须用 rubric 和人工/专家标签校准。

**落地选择:**
- **可验证任务:** 代码、数学、结构化提取优先接测试套件、编译器、linter、schema 或业务规则；失败信息可成为下一轮反馈，见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]。
- **语义任务:** 翻译、检索完整性、文案质量可用 LLM judge，但应把 rubric 拆成清晰维度，给 judge “未知/升级”出口，并以人工或专家标签反复校准。Anthropic 也建议在可行时选确定性 grader、在必要时使用 LLM grader，并与人工评审校准。[Anthropic, 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- **编排:** 任意能表示“评估→条件跳转→改写”的代码或工作流图都可实现；框架名称不是模式成立的条件。

**典型架构:** Generator(强模型)→ Evaluator(LLM judge 或测试/规则)→ 条件分支:PASS 出结果,FAIL 把【反馈 + 上一版】喂回 Generator → 循环,到迭代上限或无进展早停。

**规模化与成本/延迟:** 若每轮包含一次生成与一次评估，最多 $m$ 轮的调用上界约为 $2m$（具体还取决于首稿、重试和 verifier）。因此迭代上限既是成本闸门也是故障保护。轮数、停止阈值与是否异步应在任务集上试验，不存在普适的 “2～3 轮” 最优值。

**可观测与运维:** 必须记录**每一轮的草稿、评估分数、反馈、是否 PASS**,这样能看「质量曲线是否真在涨」——若曲线早早走平甚至抖动,说明评估不稳或迭代无收益,该停。要监控**平均迭代轮数**(突然变多=评估器变挑剔或任务变难)和**「永不达标」率**。

**踩坑与最佳实践:**
- **把 evaluator 与 generator 分成独立调用/角色**，并用独立样本检查 judge 与人工或 verifier 的一致性；使用不同模型可能带来视角差异，但不能替代校准。
- 强制评估器输出**结构化、可执行反馈**(哪里、为什么、怎么改),只说「不够好」等于没说。
- 给评估器**明确 rubric + 打分锚点**(1-4 分各代表什么),否则判得忽左忽右。
- 检查 LLM judge 对顺序、长度、格式和提示措辞的敏感性；固定候选呈现方式，必要时交换顺序或加入人工复核。不要把单一 judge 分数当作不可质疑的真值。
- 生成器改写时**必须同时拿到上一版 + 反馈**,否则丢上下文从头乱改。

```python
# 生产化:保留历史最优 + rubric + 上限 + 无进展早停
def evaluator_optimizer(task, max_iter=3):
    draft = llm(task, model=GEN)
    best_draft, best_score = draft, float("-inf")
    previous_score, terminal_status = None, "max_iter"
    for _ in range(max_iter):
        score, feedback = judge(task, draft, rubric=RUBRIC, model=JUDGE)  # 不同模型
        log_iter(draft, score, feedback)                                  # 记质量曲线
        if score > best_score:
            best_draft, best_score = draft, score
        if score >= PASS:
            terminal_status = "passed"
            break
        if previous_score is not None and score <= previous_score:
            terminal_status = "no_progress"
            break
        previous_score = score
        draft = llm(f"按反馈改进。\n任务:{task}\n上版:{draft}\n反馈:{feedback}",
                    model=GEN)                            # 改写带上一版+反馈
    return {"draft": best_draft, "score": best_score,
            "terminal_status": terminal_status}
```

## 面试高频
**Q1:Evaluator-Optimizer 和 Reflexion 是什么关系?**
标准答:Evaluator-Optimizer 是 [[13 Reflection 与 Reflexion|Reflexion]] 在 workflow 层的对应——都是「产出→评→改」，但前者把生成、评估、停止条件预设编排。评估可由独立 LLM、规则或 verifier 实现；Reflexion 则更强调 agent 在自主轨迹中的反思。

**Q2:它和 Prompt Chaining 都有多步,本质区别?**
标准答:Chaining 是**单向控制流**，gate（确定性或经校准的语义 gate）只决定放行、拦截或升级；Evaluator-Optimizer 是**闭环**，评估器给可执行反馈，驱动生成器重做同一产物并重新评价。一个是往前走，一个是原地打磨。

**Q3:这个模式成立的两个前提是什么?缺了会怎样?**
标准答:① 评价标准**明确**(评估器能稳定判好坏);② 迭代**确有边际收益**(改了真的更好)。缺① → 评估忽左忽右,生成器越改越乱;缺② → 循环空转烧钱。两者缺一就别用这个模式。

**Q4:为什么 evaluator 和 generator 最好分开,甚至用不同模型?**
标准答:独立角色/调用有利于减少同一上下文的自洽偏差，并让评估可以单独校准；异构模型只是候选手段。真正的证据是它与人工标签或客观 verifier 的一致性，而非“不同模型”这一个配置项。

**Q5:怎么防死循环?**
标准答:至少三件事：**迭代上限**、无进展/预算停止条件、`terminal_status`。每轮更新 `best_draft` 和 `best_score`，停止后返回历史最佳版；最后一轮不保证最好。

**陷阱题:「让 LLM 写完自己检查一遍再改」是不是 Evaluator-Optimizer?** → 看角色是否分离。若生成和评估是**两个独立调用/角色**且有**明确循环结构**,是;若只是同一次推理里顺带自查,那更接近 self-refine / Reflexion 的味道。边界确实模糊,关键看「评估是否被独立编排出来」。

## 知识拓展
- **同族方法谱系:** **Self-Refine**(NeurIPS 2023/2024)是单模型版的生成-自反馈-改进;**Reflexion**(2023)把反思写进记忆指导下一次尝试;Evaluator-Optimizer 把评估器**独立编排**出来。三者共享「产出→评→改」内核,差别在评估者是谁、循环是预设还是自主,见 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]]。
- **证据边界:** Anthropic 指出 LLM-as-judge 需要与人类专家校准，并建议在可行时优先确定性 grader；分数、阈值与最大轮数必须随任务数据验证。[Anthropic, 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- **从「LLM judge」到「verifier」:** 在可验证任务（代码、数学）里，把 evaluator 换成测试套件、形式化检查或业务规则通常更可审计；生成—验证闭环还可连接到 [[32 Agentic RL 与训练|Agentic RL]] 的奖励信号。
- **边界与反模式:** ① 评价标准模糊还硬用 → 反馈抖动；② 不设上限/早停/终止状态 → 死循环烧钱；③ 未校准 judge 就当真值；④ 不保留历史最佳而直接返回末轮；⑤ 反馈只说「不够好」→ 无法执行。
- **与兄弟模式:** 常作为「打磨器」嵌进其他模式——[[07 Orchestrator-Workers|Orchestrator-Workers]] 综合后用它把交付物打磨到达标;[[04 Prompt Chaining|Prompt Chaining]] 某一步产物用它迭代提质再进下一步;[[10 Plan-and-Execute|Plan-and-Execute]] 里也常在每步后插一道评估闸。评估器的工程化(怎么自动、稳定判好坏)本身就是 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 的核心课题。
