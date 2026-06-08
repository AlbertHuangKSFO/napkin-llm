[[10 Plan-and-Execute|Plan-and-Execute]] 把 agent 拆成两个角色:先让 **Planner 一次性想出完整的多步计划**,再让 **Executor 逐步照单执行**;只在执行偏离预期时才回头 **replan**——用"规划与执行解耦"换取省 token、更快、长程更稳。

它是对 [[09 ReAct|ReAct]] "每一步都重新问大模型下一步该干嘛"这一昂贵习惯的直接回应,是 [[03 Agent 核心循环|Agent 核心循环]] 在长程任务上的优化变体。

## 本质:把"想"批量化,别每步都惊动大模型

[[09 ReAct|ReAct]] 的循环里,**每走一步都要把全部历史发回大模型、让它现想下一步**。这在多跳探索里很合理,但代价是:

- **token 随步数膨胀**:第 N 步要重发前 N-1 步的全部 Thought/Action/Observation。
- **慢**:每步一次大模型往返,串行累加延迟。
- **长程易迷失**:上下文越滚越长,模型容易忘记最初目标、在局部打转。

Plan-and-Execute 的洞见:**很多任务的步骤是可以预先排布的**。与其每步都问"现在干嘛",不如**一次性把整张路线图想清楚**,然后执行阶段只做"取下一步、干、记结果"这种轻量动作——执行本身可以**不用大模型**(或只用小模型/纯工具调用)。

于是:

- **规划的智力开销集中在一次**(Planner 调用),而非摊到每一步。
- **执行变便宜**:Executor 照计划办事,不需要每步都做完整推理。
- **目标更稳**:完整计划是一份显式的"contract",执行全程对照它,不容易跑偏。
- **可纠偏**:执行结果若与预期不符,触发 **Replan** 重新规划剩余步骤——保留了应对意外的弹性。

代价是**前期规划的盲目性**:Planner 在还没看到任何中间结果时就要排好全程,对**高度不确定、必须边走边看**的任务,它的初始计划往往不准,得频繁 replan。

## 机制:Plan → Execute(逐步)→ Replan 回路

![[Plan-and-Execute.png]]

标准三段:

1. **Plan(规划)**:Planner(一个会"分解任务"的 LLM 调用)接到目标,**一次性产出有序步骤清单**,如 `["查 X 的最新数据", "据结果计算 Y", "汇总成报告"]`。这是整个流程里最耗智力的一次调用。
2. **Execute(执行)**:Executor 从计划**取队首步骤**,实际去做——调工具 / 调子 agent / 跑代码,拿到该步结果。Executor 本身可以是一个小型 [[09 ReAct|ReAct]] 单步、也可以就是一次工具调用,取决于单步复杂度。
3. **Replan(重规划)**:每完成一步(或若干步),把"已完成步骤 + 其结果"交给 Replanner。它判断:
   - 若**计划已走完 / 信息已足够** → 输出最终答案,结束。
   - 若**还有剩余步骤且一切顺利** → 直接继续执行下一步(无需真正重规划)。
   - 若**结果偏离预期 / 出现新信息** → **更新剩余计划**(增删改步骤),再继续。

这个 replan 回路是关键弹性来源:计划不是死的,执行中可以根据现实修正。控制循环同样由**外层 harness** 驱动:维护计划队列、调度 Executor、决定是否 replan、设最大轮数防失控。

> 与 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的关系:Plan-and-Execute 可看成"时间维度"的 orchestrator——Planner 像 orchestrator 拆解任务,Executor 像 worker 干活。区别在 Orchestrator-Workers 更强调**动态分派 / 并行**子任务,而经典 Plan-and-Execute 是**顺序执行**一张预排好的线性计划。

## 原论文 / 来源

Plan-and-Execute 不是单篇会议论文,而是一条**工程模式谱系**,主要来源:

- **BabyAGI**(Yohei Nakajima,2023 年 3 月):最早把"任务规划 + 任务执行 + 任务重排"做成自治循环的开源项目,引爆了"autonomous agent"热潮。它用一个 task list,执行完一个任务后由 LLM 据结果**生成新任务并重排优先级**——正是 plan/execute/replan 的雏形。
- **LangChain `Plan-and-Execute` Agent**(2023):受 BabyAGI 和论文 _Plan-and-Solve Prompting_ 启发的官方实现,明确拆出 **planner**(出步骤)与 **executor**(逐步执行,内部常是一个 ReAct agent)。
- 学术侧的近亲是 **Wang et al., _Plan-and-Solve Prompting_(ACL 2023)**——提出"先让模型规划再求解"的提示策略,思想同源但停留在单次 prompt 层面,没有真正的执行/重规划循环。

可以把 [[11 ReWOO|ReWOO]] 和 [[12 LLMCompiler|LLMCompiler]] 看作这条谱系的**进阶**:ReWOO 把执行阶段的观测彻底从大模型剥离,LLMCompiler 把计划编译成 DAG 来并行执行。

## 可跑的最小实现

```python
def plan(goal, llm):
    """Planner:一次性产出有序步骤清单。"""
    prompt = f"目标:{goal}\n把它拆成最少的有序步骤,每行一步,只列步骤。"
    return [s.strip("-• ") for s in llm(prompt).splitlines() if s.strip()]

def execute_step(step, context, tools_llm):
    """Executor:执行单步,可调工具;context 携带前序结果。"""
    return tools_llm(f"已知:{context}\n现在执行这一步:{step}")

def replan(goal, done, llm):
    """Replanner:据已完成结果决定——收尾 or 更新剩余计划。"""
    prompt = (f"目标:{goal}\n已完成:{done}\n"
              "若信息已足够,输出 'FINISH: <最终答案>';"
              "否则只列出还需要做的剩余步骤(每行一步)。")
    out = llm(prompt)
    if out.startswith("FINISH:"):
        return None, out[len("FINISH:"):].strip()
    return [s.strip("-• ") for s in out.splitlines() if s.strip()], None

def plan_and_execute(goal, planner_llm, exec_llm, max_rounds=10):
    steps = plan(goal, planner_llm)        # ① 一次性规划
    done = []                              # 已完成步骤+结果
    for _ in range(max_rounds):
        if not steps:
            break
        step = steps.pop(0)                # ② 取队首执行
        result = execute_step(step, done, exec_llm)
        done.append((step, result))
        # ③ 重规划:决定收尾 or 更新剩余计划
        steps, answer = replan(goal, done, planner_llm)
        if answer is not None:
            return answer
    return f"按计划执行完成:{done[-1][1] if done else '无结果'}"
```

要点:① **Plan 只调一次大模型**,执行阶段的 `exec_llm` 可以是更小/更便宜的模型甚至纯工具;② `replan` 是省钱关键——顺利时它不真正重排,只在偏差时才更新计划;③ `max_rounds` 防止 replan 无限循环。

## 对比:为什么比 ReAct 省

| 维度 | [[09 ReAct|ReAct]] | **Plan-and-Execute** |
|---|---|---|
| 规划时机 | 隐式,**每步**现想 | 显式,**前期一次**(+偏差时 replan) |
| 调大模型频率 | 每一步 | 规划 1 次 + 每步可用小模型/工具执行 |
| Token / 延迟 | 随步数线性膨胀 | 显著更省、可并行规划与执行 |
| 长程稳定性 | 易迷失目标 | 有显式计划做锚,**更稳** |
| 应对意外 | 天然(每步现想) | 靠 **replan** 补回 |
| 适合 | 高不确定、必须边走边看 | **步骤可预排**的长流程 |

再往省的方向走:[[11 ReWOO|ReWOO]] 把执行阶段的 observation 完全不回灌给大模型(planner 一次出含占位变量的蓝图,worker 静默取证,solver 末尾一次合成);[[12 LLMCompiler|LLMCompiler]] 则把计划做成**任务 DAG**,无依赖的步骤**并行**执行以进一步压延迟。三者是同一条"减少大模型介入次数"主线上的不同点。

**省 token 手算(对同一个 8 步任务)。** 设任务 8 步,每步产生约 $250$ token 的 Thought/Observation,系统提示 + 工具表固定前缀 $1500$ token。**[[09 ReAct|ReAct]] 路线**:每步都把固定前缀 + 前序全历史重发给**大模型**,第 $i$ 步输入约 $1500 + 250i$,八步输入累计

$$
\sum_{i=1}^{8}(1500+250i)=8\times1500+250\times\tfrac{8\times9}{2}=12000+9000=21000 \text{ token}
$$

全部走**大模型**(贵单价)。**Plan-and-Execute 路线**:Planner 大模型只调 1 次(读目标 + 前缀,约 $1500+300\approx1800$ token);执行 8 步走**小模型**,每步只带"前缀 + 当前步 + 精简前序结果",约 $1500+250\times2=2000$ token,八步约 $1.6\times10^4$,但单价按小模型(约大模型的 $\tfrac{1}{10}$)折算只等效 $\sim1600$ 大模型 token;末尾 replan/收尾大模型 1 次约 $2500$。**大模型等效 token $\approx1800+1600+2500\approx5900$,约为 ReAct 的 $28\%$**——省的不只是 token 总量,更是把绝大多数 token 从**大模型单价**挪到了**小模型单价**。这正是"规划的智力开销集中一次、执行廉价化"在账面上的样子;前提是任务步骤**可静态预排**(否则频繁 replan 会把省下的又吐回去)。

## 何时用 / 坑

**该用的场景**:目标明确、**步骤大体可预先排布**的长程任务——多阶段数据处理、研究报告生成、固定 SOP 的自动化流程。这类任务每步都问大模型纯属浪费。

**不该用的场景**:**高度不确定、下一步严重依赖上一步真实结果**的探索式任务(此时初始计划必然不准,频繁 replan 反而比 [[09 ReAct|ReAct]] 还贵)。

**常见坑**:
- **初始计划过乐观**:Planner 没见过中间结果就排全程,常漏掉意外分支。务必保留 replan 弹性,别把计划当死命令。
- **Replan 抖动 / 死循环**:每步都大改计划会反复横跳、不收敛。给 `max_rounds`,并让 replan 倾向"小修而非重排"。
- **计划与执行脱节**:Executor 执行结果没有结构化回灌给 Replanner,导致它"看不见"现实而瞎排。`done` 上下文必须如实带回。
- **单步过重**:若每个 step 本身又是个复杂探索,Executor 内部可能还得套一个 [[09 ReAct|ReAct]],别把所有复杂度都压到"一步"里。
- **错误步骤被跳过不补救**:某步失败若不触发 replan,后续会建在错误结果上。replan 要能识别失败并插入补救步骤。

## 关键事实

- 两角色:**Planner(一次出完整多步计划)+ Executor(逐步执行)**,加 **Replan** 回路应对偏差。
- 核心动机:把 [[09 ReAct|ReAct]] "每步都问大模型"的开销**集中到规划一次**,执行阶段廉价化 → 省 token、更快、长程更稳。
- 执行器可用**小模型 / 纯工具**,智力开销集中在 Planner;这是它比 ReAct 省的根本原因。
- 弹性来自 **replan**:计划不是死的,可据真实结果增删改剩余步骤。
- 来源是工程谱系(**BabyAGI / LangChain Plan-and-Execute**)+ 学术近亲 **Plan-and-Solve Prompting(ACL 2023)**,非单篇会议论文。
- 进阶方向:[[11 ReWOO|ReWOO]](观测不回灌)、[[12 LLMCompiler|LLMCompiler]](DAG 并行);上游近亲是 [[07 Orchestrator-Workers|Orchestrator-Workers]]。
- 软肋:**初始计划在高不确定任务上不准**,得频繁 replan,此时未必比 ReAct 划算。

## 主流开源实现 / Python 库

- **`langchain-ai/langgraph` Plan-and-Execute 教程** —— **当下首选**。官方 tutorial 把模式搭成三个节点:`plan_step`(Planner 出多步计划)→ `execute_step`(每步用一个子 agent / 工具执行,可用更轻的 LLM)→ `replan_step`(据已完成结果决定收尾或更新剩余计划),用 `StateGraph` 连成带 replan 回路的图。这正是"先规划后执行"在 LangGraph 里的标准搭法。
- **`run-llama/llama_index` 结构化规划 / step-wise 执行** —— LlamaIndex 用 `AgentRunner` + `AgentWorker` 把"推进一步"和"完整 run"拆开,配合结构化 planner 可实现先规划、再逐步执行;适合检索密集型流程。
- **`yoheinakajima/babyagi`** —— 历史源头,task list + LLM 重排优先级的自治循环(plan/execute/replan 雏形),适合读思想,不建议直接生产。

首选:用 LangGraph 教程版手搭三节点图最清晰可控。Plan-and-Execute 无独立 pip 包,本质是一种图编排模式。

## 工业界实践

Plan-and-Execute 在生产里很少以"原教旨"形态出现,但它的**核心拆分(规划层 / 执行层 / 重规划层)是绝大多数复杂 agent 产品的骨架**——尤其是 Deep Research、多步数据处理、SOP 自动化这类"目标明确、步骤大体可排"的场景。

**主流落地形态(具体名 + 定位)**:
- **LangGraph 的 plan-execute-replan 图**:官方教程把它落成 `plan_step → execute_step → replan_step` 三节点 `StateGraph`,是工程里最常被照搬的模板。生产里通常给执行节点配**更便宜的模型**(规划用大模型,执行用小模型/工具)。
- **OpenAI Deep Research / Gemini Deep Research / Perplexity**:用户看到的"先列研究计划,再逐条检索,最后汇总"正是 Plan-and-Execute 的产品化,见 [[29 Deep Research Agent|Deep Research Agent]]。
- **Devin / 各类 SWE agent 的"计划面板"**:先生成 task plan 再逐步执行、出偏差就改计划,本质同源,见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]。
- **Anthropic 的 orchestrator-worker 多 agent 研究系统**:lead agent 规划并分派、subagent 执行,是 Plan-and-Execute 在多 agent 维度的展开,见 [[22 多智能体系统|多智能体系统]]、[[07 Orchestrator-Workers|Orchestrator-Workers]]。

**典型架构**:`目标 → Planner(大模型,出有序步骤)→ 计划队列 → Executor(每步:小模型/ReAct 单步/纯工具)→ 结果回灌 → Replanner(收尾 or 更新剩余计划)`。执行层常把每个 step 做成一个**子 agent 或工具调用**;复杂 step 内部再套一个小 [[09 ReAct|ReAct]]。

**规模化与成本/延迟**:相比 ReAct,省在"规划只调一次大模型 + 执行廉价化"。但要警惕 **replan 抖动**——每步都大改计划会反复横跳、不收敛,既慢又贵。生产对策:① replan 倾向"小修而非重排";② 设 `max_rounds` / `recursion_limit`;③ 执行层分层路由(简单步用 Haiku 级,难步才升级)。延迟上,经典 Plan-and-Execute 是**顺序执行**,若步骤间无依赖可并行,应升级到 [[12 LLMCompiler|LLMCompiler]] 的 DAG 调度。

**可观测与运维**:计划本身是一份**显式 contract**,天然适合做可观测——把"原始计划 / 每步执行结果 / 每次 replan 的 diff"都记成 trace(LangSmith / Langfuse)。排障时能直接看"是规划排错了,还是某步执行失败,还是 replan 反复横跳",定位比纯 ReAct 的长 trace 更清晰,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。计划可中断/可恢复(checkpointer),长程任务断点续跑,见 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]。

**踩坑与最佳实践**:
- **别把初始计划当死命令**:Planner 没见过中间结果就排全程,务必保留 replan 弹性。
- **执行结果必须结构化回灌给 Replanner**:否则它"看不见"现实而瞎排(计划与执行脱节)。
- **replan 要能识别失败并插补救步骤**:否则后续步骤建在错误结果上。
- **单步别过重**:若每个 step 本身又是复杂探索,要么拆细,要么 step 内套 ReAct,别把所有复杂度压到"一步"里。
- **给 Planner 工具清单/约束**:让它只排"可执行"的步骤,避免计划里出现没有对应工具的动作。

```python
# LangGraph 风格的三节点骨架(伪代码)
def plan_step(state):                       # 大模型只调一次
    state["plan"] = planner_llm(state["goal"])      # ["查X", "据X算Y", "汇总"]
    return state

def execute_step(state):                    # 执行层可用更便宜的模型/工具
    step = state["plan"][0]
    result = cheap_executor(step, context=state["done"])
    state["done"].append((step, result))
    state["plan"] = state["plan"][1:]
    return state

def replan_step(state):                     # 省钱关键:顺利不重排,只在偏差时改
    decision = replanner_llm(state["goal"], state["done"], state["plan"])
    if decision.finished:
        state["answer"] = decision.answer
    else:
        state["plan"] = decision.updated_remaining_steps   # 倾向小修
    return state
# 图:plan → execute →(replan → execute 循环)→ END;带 max_rounds 护栏
```

## 面试高频

**Q1:Plan-and-Execute 相比 ReAct 省在哪?为什么?**
标准答:ReAct **每步都把全部历史发回大模型现想下一步**,token 随步数膨胀、串行慢、长程易迷失。Plan-and-Execute 把"规划的智力开销集中到一次 Planner 调用",执行阶段廉价化(可用小模型/纯工具),只在偏差时 replan——所以省 token、更快、有显式计划做锚更稳。
- 追问"执行层为什么能用小模型":因为执行只是"照计划取下一步、干、记结果",不需要每步做完整推理。
- 陷阱:面试官问"那它是不是永远比 ReAct 好"——不是,高不确定、下一步严重依赖上一步真实结果的探索式任务,初始计划必然不准、频繁 replan 反而比 ReAct 还贵。

**Q2:三个角色/三段分别干什么?**
标准答:Planner(一次性出完整有序步骤)/ Executor(逐步执行,调工具/子 agent/跑代码)/ Replan 回路(据已完成结果决定:收尾 / 继续 / 更新剩余计划)。控制循环由外层 harness 驱动:维护计划队列、调度执行、决定是否 replan、设最大轮数。
- 追问"replan 一定会重排吗":不一定——顺利时它只是"继续下一步",真正重排只在结果偏离预期时发生,这正是省钱点。

**Q3:Plan-and-Execute 和 Orchestrator-Workers 什么关系?**
标准答:Plan-and-Execute 可看成"时间维度"的 [[07 Orchestrator-Workers|Orchestrator-Workers]]——Planner 像 orchestrator 拆任务、Executor 像 worker 干活。区别:Orchestrator-Workers 更强调**动态分派/并行**子任务;经典 Plan-and-Execute 是**顺序执行**一张预排好的线性计划。

**Q4:它从哪来的?有没有原论文?**
标准答:不是单篇会议论文,而是工程模式谱系——**BabyAGI(2023,task list + LLM 重排优先级的自治循环)+ LangChain Plan-and-Execute Agent**;学术近亲是 **Plan-and-Solve Prompting(ACL 2023)**,但后者停留在单次 prompt 层面,没有真正的执行/重规划循环。
- 陷阱:把 Plan-and-Solve(纯提示策略)和 Plan-and-Execute(带执行/replan 循环的编排)混为一谈是常见错。

**Q5:它最大的软肋是什么?怎么缓解?**
标准答:**初始计划的盲目性**——Planner 在没看到任何中间结果时就排全程,对高不确定任务往往不准。缓解:保留 replan 弹性、让 replan 能识别失败并插补救步骤、给 `max_rounds` 防抖动;若任务本就高度探索式,干脆退回 [[09 ReAct|ReAct]]。

## 知识拓展

**谱系定位**:这是"减少大模型介入次数"主线的**第二站**:[[09 ReAct|ReAct]](每步问)→ **Plan-and-Execute(规划集中一次)** → [[11 ReWOO|ReWOO]](连中间观测都不回灌、固定调 2 次)→ [[12 LLMCompiler|LLMCompiler]](把计划编译成 DAG 并行)。Plan-and-Execute 是这条线的"分水岭"——首次把规划与执行解耦。

**两个进阶方向**:
- **省 token 的极致**:[[11 ReWOO|ReWOO]] 把执行阶段的 observation 完全不回灌给大模型(planner 一次出含占位变量的蓝图,worker 静默取证,solver 末尾一次合成),报告比 ReAct 省约 5× token。
- **降延迟的极致**:[[12 LLMCompiler|LLMCompiler]] 把计划做成**带依赖的任务 DAG**,无依赖步骤并行执行,把 [[06 Parallelization|Parallelization]] 下沉进执行引擎。

**相关论文/前沿**:
- **Wang et al., _Plan-and-Solve Prompting_(ACL 2023)**——学术近亲,单 prompt 的"先规划再求解"。
- **Yao et al., _Tree of Thoughts_(NeurIPS 2023)**——当"计划"不是单条而是多条候选时,规划层本身可以做树搜索,见 [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]]。
- **Anthropic, _How we built our multi-agent research system_(2025)**——把 Plan-and-Execute 放大到多 agent:lead 规划+分派、subagent 并行执行,见 [[22 多智能体系统|多智能体系统]]。
- **长程自我改进**:把"执行结果 → 修正计划"做成持续循环,接近 [[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进(Ralph loop)]]。

**边界与反模式**:
- **反模式 1:对高不确定任务硬上 Plan-and-Execute**——初始计划必然不准,频繁 replan 比 ReAct 还贵,该用 [[09 ReAct|ReAct]]。
- **反模式 2:每步都大改计划**——replan 抖动、不收敛,既慢又烧钱;让 replan 倾向小修。
- **反模式 3:执行结果不结构化回灌**——Replanner 看不见现实而瞎排,计划与执行彻底脱节。
- **边界**:经典版是顺序执行,无法表达"无依赖步骤并行";一旦任务有明显并行结构,应升级到 [[12 LLMCompiler|LLMCompiler]]。
