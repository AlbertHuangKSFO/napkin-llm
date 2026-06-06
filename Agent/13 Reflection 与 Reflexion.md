**Reflection** 与 **Reflexion** 是两层不同的「让 agent 自我纠错」机制:**Reflection** 是模型对自己刚产出的结果做自评并就地修订(一个单纯的自我批判循环);**Reflexion** 则更进一步,把任务**失败的言语反馈**写进 episodic memory,在**下一次尝试**时回灌——靠记忆改行为,不改权重,这叫「言语强化学习(verbal reinforcement learning)」。

它俩都是 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 这一编排模式在自主 agent 里的体现:有人生成、有人评判、据评判迭代。区别在于反馈停留在**同一次产出内**(Reflection)还是**跨尝试沉淀进记忆**(Reflexion)。Reflexion 因此天然要接 [[19 Agent 记忆系统|Agent 记忆系统]],也常作为 [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]] 里 LATS 的「反思」组件被复用。

## 本质:两层要分清

| | Reflection(自反思) | Reflexion(言语强化) |
|---|---|---|
| 作用域 | **单次产出内**:生成→自评→改 | **跨尝试**:失败→反思→存记忆→下次回灌 |
| 记忆 | 不需要(改完即弃) | 需要 episodic memory 持久存反思 |
| 改的是 | 当前这版答案 | 下一次尝试的行为策略 |
| 类比 | 交卷前自己检查一遍再改 | 这次考砸,写下错因,下次考前再看 |

**一句话**:Reflection 是「就地重写」,Reflexion 是「把教训记下来,下次别再犯」。Reflexion 包含 Reflection,但多了**记忆 + 多次 trial**这条主线。

## Reflection 的机制(基础循环)

最朴素的自我批判循环,三步:
1. **Generate**:LLM 产出初版结果(草稿、代码、答案)。
2. **Reflect / Critique**:同一个或另一个 LLM 角色扮演「审稿人」,挑毛病——指出哪里错、哪里可改进(可借工具,如跑测试、查事实)。
3. **Refine**:把批评意见喂回 generator,产出改进版。可循环 N 轮或到「无更多意见」为止。

这本质就是 [[08 Evaluator-Optimizer|Evaluator-Optimizer]]:generator=optimizer,critic=evaluator。它便宜、好实现,但**反馈不沉淀**——同一类错误下次还会犯。

## Reflexion 的机制(分步讲透)

Reflexion 把上面的循环装进一个**带记忆的多尝试框架**,四个组件:

- **Actor**:执行任务的 LLM(通常本身就是个 [[09 ReAct|ReAct]] agent),据「任务 + 记忆里的历史反思」产出行动轨迹(trajectory)。
- **Evaluator**:给 Actor 这次的轨迹打分——可以是环境给的奖励(代码是否通过测试)、规则、或一个 LLM 评判器。产出**标量/二元反馈**(成功 vs 失败、得几分)。
- **Self-Reflection**:**Reflexion 的灵魂**。它读「轨迹 + Evaluator 的稀疏反馈」,生成一段**自然语言的失败归因**——「我哪一步做错了、为什么、下次该怎么改」。注意:它把「测试没过」这种稀疏信号,翻译成了**信息量丰富的言语指导**。
- **Memory**:把这段反思文本存进 **episodic memory**(一个滑动窗口的反思缓冲)。下一次 trial 时,这些反思被拼进 Actor 的 prompt 回灌进去。

完整流程:**Actor 尝试 → Evaluator 评判 → 失败则 Self-Reflection 写反思 → 存 Memory → 下一轮 Actor 带着反思重试**,直到成功或到最大尝试数。

![[Reflection 与 Reflexion.svg]]

关键洞察:**不更新任何模型权重**。强化学习的「试错改进」效果,这里完全靠「把教训写成自然语言、塞回上下文」来实现——所以叫 verbal RL。代价小(无需训练)、迭代快(改 prompt 即生效)、可解释(反思是人能读的文字)。

![[Reflection 与 Reflexion-记忆回灌.svg]]

## 原论文

- **Reflexion:Shinn, Cassano, Gopinath, Narasimhan, Yao,_Reflexion: Language Agents with Verbal Reinforcement Learning_**(NeurIPS 2023,arXiv 2023-03)。在 HumanEval 代码任务上 pass@1 达 91%(超过当时 GPT-4 的 80%),在 ALFWorld、HotpotQA 等也显著提升。核心贡献就是「verbal reinforcement learning」这一范式 + Actor/Evaluator/Self-Reflection/Memory 架构。
- **Reflection(自我批判)** 不是单篇论文专属,是一类思想,代表工作有 **Self-Refine(Madaan et al., 2023)**(迭代自我反馈精修)、**CRITIC(Gou et al., 2023)**(借外部工具批判修正)。它们都属于「生成-评判-改进」的 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 家族。

## 可跑最小代码(带 memory 的 Reflexion 循环)

```python
def reflexion(task, actor, evaluator, reflect, max_trials=4):
    memory = []                                    # episodic memory:存历次反思文本
    for trial in range(max_trials):
        # 1) Actor 带着历史反思尝试(memory 回灌进 prompt)
        trajectory = actor(task, reflections=memory)
        # 2) Evaluator 评判(测试/规则/LLM 评)
        passed, score, signal = evaluator(task, trajectory)
        if passed:
            return trajectory                      # 成功,收尾
        # 3) Self-Reflection:把稀疏失败信号翻译成言语指导
        reflection = reflect(task, trajectory, signal)
        # 4) 写入记忆,下一轮回灌
        memory.append(reflection)
        memory = memory[-3:]                        # 滑动窗口,只留最近几条
    return trajectory                              # 用尽尝试,返回最后一版

# 对照:朴素 Reflection —— 无 memory、不跨 trial,就地改一版
def reflection(task, gen, critic, rounds=2):
    draft = gen(task)
    for _ in range(rounds):
        critique = critic(task, draft)             # 自评
        if "无改进意见" in critique: break
        draft = gen(task, feedback=critique)        # 据评修订,丢弃旧批评
    return draft
```

两段代码的差别一眼可见:Reflexion 多了 `memory` 这条跨尝试的线,Reflection 的 `critique` 用完即弃。

## 对比:Reflection vs Reflexion vs 其它

| 维度 | Reflection | Reflexion | 普通 [[09 ReAct|ReAct]] |
|---|---|---|---|
| 自我评判 | 有 | 有(Evaluator) | 无 |
| 跨尝试记忆 | 无 | **有(episodic)** | 无 |
| 是否多次 trial | 否(同次精修) | **是** | 否 |
| 改权重 | 否 | 否(verbal RL) | 否 |
| 适用 | 单次产出要更稳更准 | 可重试、有明确成败信号的任务 | 单趟探索式任务 |

## 何时用 / 坑

✅ **用 Reflection**:单次产出质量要拔高、且能定义「好坏标准」(写作润色、代码自查、答案校验)。便宜、即插即用。

✅ **用 Reflexion**:任务**允许多次尝试**、且有**相对明确的成败/评分信号**(代码过不过测试、游戏赢没赢、答案对不对)。它在这类「可验证、可重试」场景收益最大。

❌ **别用**:
- 任务**没有可靠评判信号**——Evaluator 不准,反思就是在错误方向上自我强化。
- 任务**不可重试或重试代价极高**(每次都真改了生产数据),Reflexion 的多 trial 假设不成立。
- 想靠它「修复模型能力本身的缺陷」——它只调行为策略,模型根本不会的东西反思也补不出。

坑:
- **过度自信的反思**:LLM 可能反思出错误的归因,把对的步骤改坏(自我批判不等于自我正确)。
- **记忆爆炸 / 偏题**:反思无限堆积会撑爆上下文、互相矛盾;要用滑动窗口或摘要,见 [[19 Agent 记忆系统|Agent 记忆系统]]。
- **Evaluator 是上限**:整套机制的天花板是评判信号的质量;评判错,迭代越多越偏。
- **死循环**:反复失败却一直反思而不收敛,必须设 `max_trials`。

## 关键事实速记

- Reflection = **单次产出内**的生成-评判-改进;Reflexion = **跨尝试**把言语反馈存 [[19 Agent 记忆系统|Agent 记忆系统]] 再回灌。
- Reflexion 四件套:**Actor、Evaluator、Self-Reflection、Memory**;灵魂是 Self-Reflection 把稀疏信号译成丰富的自然语言指导。
- **不改权重**:用「自然语言反馈」实现强化学习式的试错改进 → verbal RL(Shinn et al., NeurIPS 2023)。
- 两者都属 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 家族;Reflexion 的反思组件被 [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]] 的 LATS 直接复用进树搜索。
- 收益前提:**有可靠评判信号 + 可重试**;否则会在错误方向自我强化。

## 主流开源实现 / Python 库

- **`noahshinn/reflexion`** —— **Reflexion 原论文官方实现**(NeurIPS 2023,Shinn et al.),含 HumanEval / ALFWorld / HotpotQA 三套实验的 Actor/Evaluator/Self-Reflection/Memory 完整代码与日志。要复现论文(如 HumanEval pass@1 91%)或读架构细节看它。
- **`langchain-ai/langgraph-reflection`** —— 官方 prebuilt 的 **reflection** 图:一个 main agent 解题 + 一个 critique agent 审稿,据批评决定是否重做。即插即用,是工程里上 Reflection 的**首选**。
- **`langchain-ai/langgraph` reflexion 教程** —— 官方 tutorial(`examples/reflexion/reflexion.ipynb`)用 LangGraph 把带 episodic memory 的多 trial **Reflexion** 回路搭成图,要完整 Reflexion(跨尝试记忆)用它。

首选:只要"生成-评判-改一版"用 `langgraph-reflection`;要"跨尝试记教训再重试"用 langgraph reflexion 教程;复现论文用 `noahshinn/reflexion`。

## 工业界实践

### 落地形态:Reflection 几乎无处不在,Reflexion 只在「可重试」处出现

- **代码 Agent 的内建反思**:Cursor、GitHub Copilot Workspace、Devin、Claude Code、OpenAI Codex 这类 SWE Agent,普遍把「跑测试 → 读失败 → 改代码 → 再跑」做成默认回路——这正是 Reflexion 的工业形态:**测试套件就是 Evaluator,编译/测试报错就是稀疏信号,模型读 traceback 写出修复就是 Self-Reflection**。详见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]。SWE-bench 上的高分方案(如 SWE-agent、Agentless)几乎都靠「执行反馈回灌 + 多次迭代」,这是 verbal RL 的最强生产证据。
- **Deep Research / 长文生成**:[[29 Deep Research Agent|Deep Research Agent]]、OpenAI Deep Research、Google 的 Gemini Deep Research 普遍用「先草稿 → critic 审 → 补检索/改写」的 Reflection 回路提升事实性与完整度。
- **结构化输出校验**:`instructor`(Pydantic 校验失败 → 把校验错误回灌让模型重填)是最朴素、用量最大的「单步 Reflexion」——校验器是 Evaluator,Pydantic 的 `ValidationError` 是言语反馈。这是工业界跑量最大的 Reflexion 变体,只是没人这么叫它。
- **框架支持**:**LangGraph** 的 reflection / reflexion 图模板;**LlamaIndex** 的 `IntrospectiveAgent`(reflective agent worker);**AutoGen** 用「双 Agent 对话」(一个生成、一个 critic)天然实现 Reflection;**CrewAI** 可配 critic agent 做审稿环。

### 工程化要点(把 demo 变成生产)

| 关注点 | 朴素做法 | 生产做法 |
|---|---|---|
| Evaluator 质量 | 让同一个 LLM 自评 | 优先用**可验证信号**(单测、编译、schema 校验、检索 ground truth);LLM-judge 仅兜底,且独立 prompt/模型避免自我偏袒 |
| 收敛控制 | 固定轮数 | `max_trials` + **早停**(连续两轮无改进、或分数不升即停)+ 每轮**预算熔断** |
| 反思记忆 | 无限堆 | 滑动窗口 / 摘要压缩(见 [[21 上下文压缩与卸载|上下文压缩与卸载]]);只留对当前任务相关的反思 |
| 成本 | 反思也用最强模型 | 反思/评判可用**更小更便宜**的模型;生成用强模型(见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]) |
| 可观测 | 黑盒 | 把每轮 trajectory / 反思文本 / 分数全部 trace 落库(LangSmith、Langfuse、Phoenix),离线复盘哪类反思真有效 |

### 踩坑(生产实录)

- **自评偏高(self-bias)**:LLM 给自己产出打分普遍偏高,导致「明明没改好却判过」。对策:Evaluator 与 Actor 用不同模型 / 不同 prompt,关键场景上**外部可验证信号**。
- **反思「正确但无用」**:模型写出一段听着对的反思,却没改变下一轮行为(反思与行动脱节)。对策:反思要**可执行化**——要求输出「下一步具体怎么改」,而非泛泛归因。
- **延迟翻倍**:N 轮反思 = N 倍 LLM 往返,面向用户的同步场景慎用;多放到**异步/离线**(如夜间批量精修)。
- **错误方向自我强化**:Evaluator 不准时,越反思越偏——这是 Reflexion 最危险的失效模式,务必先确保评判可靠再开多轮。

## 面试高频

**Q1:Reflection 和 Reflexion 有什么区别?**
标准答:Reflection 是**单次产出内**的「生成-自评-修订」就地循环,反馈用完即弃;Reflexion 是**跨尝试**框架,把失败的**言语反馈写进 episodic memory**,下一次 trial 回灌进 prompt,靠记忆改行为。Reflexion 包含 Reflection,但多了「记忆 + 多次 trial」这条主线。一句话:Reflection 是交卷前自查重写,Reflexion 是这次考砸记下错因下次别再犯。
- **追问:Reflexion 改模型权重吗?** 不改。它用「自然语言反馈塞回上下文」实现强化学习式的试错改进,所以叫 **verbal reinforcement learning(言语强化学习)**,Shinn et al. NeurIPS 2023。代价小、迭代快(改 prompt 即生效)、可解释。

**Q2:Reflexion 的四个组件是什么?灵魂是哪个?**
标准答:**Actor**(执行任务、产出 trajectory,通常是 ReAct agent)、**Evaluator**(给 trajectory 打标量/二元分,可来自环境奖励/规则/LLM-judge)、**Self-Reflection**(读「轨迹+稀疏反馈」生成自然语言失败归因)、**Memory**(episodic memory 存反思、下轮回灌)。**灵魂是 Self-Reflection**——它把「测试没过」这种稀疏信号,翻译成信息量丰富的言语指导。

**Q3:什么场景适合 Reflexion?什么场景不适合?**
标准答:适合**可多次尝试 + 有相对明确成败/评分信号**的任务(代码过不过测试、游戏赢没赢、答案对不对)。不适合:① 没可靠评判信号(Evaluator 不准会在错误方向自我强化);② 不可重试或重试代价极高(每次真改生产数据);③ 想修复模型能力本身的缺陷(它只调行为策略,模型根本不会的反思也补不出)。

**Q4(陷阱):多轮反思一定越来越好吗?** 不一定。天花板是 **Evaluator 的质量**;评判错则迭代越多越偏。且 LLM 自评有偏高倾向、反思可能把对的步骤改坏。必须设 `max_trials` + 早停防死循环。

**Q5:Reflexion 和 RLHF/RL 微调的关系?** 都做「试错改进」,但 RL 微调**改权重**(梯度更新)、训练慢、不可解释;Reflexion **不改权重**、靠上下文、即时生效、反思人类可读。Reflexion 是「推理时(inference-time)」的轻量替代,适合无法微调或要快速迭代的场景;两者可叠加(用反思数据反过来做 SFT/RL,见 [[32 Agentic RL 与训练|Agentic RL 与训练]])。

**Q6:Reflexion 和 ToT/LATS 是什么关系?** Reflexion 是**单链多尝试**(一条轨迹失败→反思→重来);[[14 树搜索：ToT 与 LATS|LATS]] 把反思组件**直接复用进树搜索**——失败分支生成反思注入后续扩展,是「搜索 + 行动 + 反思」三合一。

## 知识拓展

- **谱系定位**:Reflection/Reflexion 同属 [[08 Evaluator-Optimizer|Evaluator-Optimizer]] 家族(生成者+评判者);[[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]] 的 LATS 把它升级进树;[[15 Function Calling 工具调用|Function Calling]] 的「错误回灌让模型自纠」是同一思想在工具层的最小化身;记忆侧接 [[19 Agent 记忆系统|Agent 记忆系统]](episodic memory)与 [[21 上下文压缩与卸载|上下文压缩与卸载]](反思摘要)。
- **相关工作与前沿(带年份)**:
  - **Self-Refine**(Madaan et al., 2023):同一模型迭代自我反馈精修,无外部信号,是最纯的 Reflection。
  - **CRITIC**(Gou et al., 2023):用**外部工具**(搜索、计算器、解释器)验证并批判模型输出,而非只靠自评——降低「自我幻觉式反思」。
  - **Self-Consistency**(Wang et al., 2022):多次采样投票,是 Reflection 的「无显式批判」廉价替代,常作对照基线。
  - **Reflexion**(Shinn et al., NeurIPS 2023):HumanEval pass@1 91%(超当时 GPT-4 的 80%),ALFWorld/HotpotQA 显著提升。
  - **STaR / Quiet-STaR**(Zelikman et al., 2022/2024):把「自我生成的好推理」反过来训练模型——把反思从推理时搬进训练时,通往自我改进(见 [[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进]])。
  - 与 OpenAI o1 / DeepSeek-R1(2024–2025)的**推理时反思**对照:大模型已把「反思-回溯」内化进 CoT,部分外置 Reflexion 的需求被模型自身吸收——但**外部可验证信号 + 跨尝试记忆**仍是模型内化不了的部分,Reflexion 思路不过时。
- **边界 / 反模式**:① 在无可靠评判的开放任务上硬上多轮反思 = 自我强化偏差;② 把反思当「能力补丁」修模型根本不会的东西;③ 反思无限堆爆上下文且互相矛盾;④ 同步面向用户场景上多轮反思导致延迟不可接受。
