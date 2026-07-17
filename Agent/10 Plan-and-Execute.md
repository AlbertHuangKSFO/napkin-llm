[[10 Plan-and-Execute|Plan-and-Execute]] 把 agent 拆成两个角色:先让 **Planner 产出初始多步计划**,再让 **Executor 通常按顺序执行**;当确定的失败/状态条件或批量检查触发时,再 **replan** 剩余步骤。它用「规划与执行解耦」换取在步骤可预排任务上的成本和可控性优势,但不是无条件更快或更省。

它是对 [[09 ReAct|ReAct]] "每一步都重新问大模型下一步该干嘛"这一昂贵习惯的直接回应,是 [[03 Agent 核心循环|Agent 核心循环]] 在长程任务上的优化变体。

## 本质:把"想"批量化,别每步都惊动大模型

在原始/简单的 [[09 ReAct|ReAct]] 循环里,每走一步都会把运行历史提供给模型、让它现想下一步。这在多跳探索里很合理,但代价是:

- **token 随步数膨胀**:第 N 步要重发前 N-1 步的全部 Thought/Action/Observation。
- **慢**:每步一次大模型往返,串行累加延迟。
- **长程易迷失**:上下文越滚越长,模型容易忘记最初目标、在局部打转。

Plan-and-Execute 的洞见:**很多任务的步骤可以先排出初稿**。与其每步都问"现在干嘛",不如先给出路线图,执行阶段再做"取下一步、干、记结果"这种轻量动作——Executor 可以是纯工具、确定性程序,也可以是模型。它是否省钱取决于 Executor 的实现、replan 次数和上下文大小。

**生活类比:按搬家清单办事。**先列出「清点箱子 → 预约货车 → 搬运 → 验收」,搬运工默认按清单一项一项做,每搬完 3 箱才例行核对一次;只有货车故障、清单前提失效或发现漏箱才重排。两个互不相关房间能否同时收拾,必须先确认它们没有共享物品和先后依赖——不能因为都叫「搬家」就默认并行。

于是:

- **初始规划集中**:先用一次 Planner 调用生成初始计划;这不等于全程只有一次 LLM 调用。
- **执行可轻量化**:若步骤可由工具或确定性程序完成,Executor 无需在每步重新做全局规划。
- **目标可核对**:计划是一份显式 contract,可把每步输出与前置条件、完成条件对照。
- **可纠偏**:由确定性/批量触发器决定何时调用 Replanner 更新剩余步骤,避免无意义地每步重排。

代价是**前期规划的盲目性**:Planner 在还没看到任何中间结果时就要排好全程,对**高度不确定、必须边走边看**的任务,它的初始计划往往不准,得频繁 replan。

## 机制:Plan → Execute(逐步)→ Replan 回路

![[Plan-and-Execute.png]]

标准三段:

1. **Plan(规划)**:Planner(一个会"分解任务"的 LLM 调用)接到目标,产出初始有序步骤清单,如 `["查 X 的最新数据", "据结果计算 Y", "汇总成报告"]`。这是第一笔规划调用,不是对后续 replan 的豁免。
2. **Execute(执行)**:Executor 从计划**取队首步骤**,实际去做——调工具 / 调子 agent / 跑代码,拿到该步结果。Executor 本身可以是一个小型 [[09 ReAct|ReAct]] 单步、也可以就是一次工具调用,取决于单步复杂度。
3. **Replan(重规划)**:不要无条件在每步调用 LLM。先由 harness 用**确定的触发器**判断是否需要复核,例如:工具失败/超时、结果违反下一步前置条件、出现使计划失效的新事实、执行了固定批次(如每 3 步),或计划队列耗尽。触发后才把"已完成步骤 + 其结果"交给 Replanner,它判断:
   - 若**计划已走完 / 信息已足够** → 输出最终答案,结束。
   - 若**还有剩余步骤且一切顺利** → 直接继续执行下一步(无需真正重规划)。
   - 若**结果偏离预期 / 出现新信息** → **更新剩余计划**(增删改步骤),再继续。

这个 replan 回路是关键弹性来源:计划不是死的,执行中可以根据现实修正。控制循环同样由**外层 harness** 驱动:维护计划队列、顺序调度 Executor、判断确定触发器、再决定是否调用 Replanner,并设 `max_steps` 和 replan 预算防失控。

> 与 [[07 Orchestrator-Workers|Orchestrator-Workers]] 的关系:Plan-and-Execute 可看成"时间维度"的 orchestrator——Planner 像 orchestrator 拆解任务,Executor 像 worker 干活。区别在 Orchestrator-Workers 更强调**动态分派 / 并行**子任务,而经典 Plan-and-Execute 是**顺序执行**一张预排好的线性计划。

## 原论文 / 来源

Plan-and-Execute 不是单篇会议论文,而是一条**工程模式谱系**,主要来源:

- **BabyAGI**(Yohei Nakajima,2023):早期的开源 task-list 循环之一;执行后创建新任务并重排优先级,可作为 plan/execute/replan 的工程先例,但不宜据此宣称唯一源头。
- **LangChain, _Planning for Agents_(2024-07-20)**:讨论长程任务要把一系列动作与不断回灌的结果结合起来,并建议把领域约束写进特定的认知架构;它说明此类规划问题的背景,**不规定**本文的失败/批量 replan gate。[原文](https://www.langchain.com/blog/planning-for-agents)
- 学术侧的近亲是 **Wang et al., _Plan-and-Solve Prompting_(ACL 2023)**——提出"先让模型规划再求解"的提示策略,思想同源但停留在单次 prompt 层面,没有真正的执行/重规划循环。

**⚠️ 常见误区**:别把 **Plan-and-Solve**(纯提示策略,一次 prompt 里先写计划再答,无工具执行)和 **Plan-and-Execute**(带 Executor 调工具 + Replan 回路的编排架构)当成一回事——前者只是怎么 prompt,后者是怎么搭 agent。

可以把 [[11 ReWOO|ReWOO]] 和 [[12 LLMCompiler|LLMCompiler]] 看作这条谱系的**进阶**:ReWOO 把执行阶段的观测彻底从大模型剥离,LLMCompiler 把计划编译成 DAG 来并行执行。

## 可跑的最小实现:❌ 每步重排 vs ✅ 顺序队列 + gate

为使示例可直接运行,下面用确定性函数模拟计划和执行;`planner_calls` 表示真实系统本应发生的 Planner/Replanner 调用数。默认路径只做 `queue.pop(0)`,没有并行 worker。

```python
INITIAL_QUEUE = [
    {"name": "查库存", "depends_on": []},
    {"name": "生成报价", "depends_on": ["查库存"]},
    {"name": "发送摘要", "depends_on": ["生成报价"]},
]

def execute(step):
    return {"ok": True, "invalidates_plan": False,
            "value": f"完成: {step['name']}"}

def should_replan(result, completed, remaining, every=3):
    return (not result["ok"]
            or result["invalidates_plan"]
            or len(completed) % every == 0
            or not remaining)

def replan_or_finish(completed, remaining):
    """门控通过后才进入;本例没有新事实,所以保留队列或结束。"""
    if not remaining:
        return [], "计划完成"
    return list(remaining), None

# ❌ 反例:每完成一步就重排,即使没有失败、没有新事实、也未到批次检查点。
def naive_every_step_replan():
    queue = list(INITIAL_QUEUE)
    completed, planner_calls = [], 1          # 初始 plan
    while queue:
        step = queue.pop(0)
        completed.append(step["name"])
        planner_calls += 1                    # 不必要的 replan
        queue, _ = replan_or_finish(completed, queue)
    return completed, planner_calls

# ✅ 改进:顺序队列默认执行;失败/失效/每 3 步/队列耗尽时才 replan。
def gated_sequential_plan_execute(max_steps=10):
    queue = list(INITIAL_QUEUE)
    completed, planner_calls = [], 1          # 初始 plan
    final = None
    while queue and len(completed) < max_steps:
        step = queue.pop(0)                   # 唯一默认调度:取队首、顺序执行
        assert set(step["depends_on"]) <= set(completed)
        result = execute(step)
        completed.append(step["name"])
        if should_replan(result, completed, queue):
            planner_calls += 1                # 此例只在第 3 步/队列耗尽时触发一次
            queue, final = replan_or_finish(completed, queue)
            if final is not None:
                break
    return completed, planner_calls, final

# 只有显式 DAG 给出且所有依赖边已验证时,才可把同一批 ready 节点交给并行调度器。
def dag_parallel_candidates(tasks, completed):
    names = {task["name"] for task in tasks}
    assert all(set(task["depends_on"]) <= names for task in tasks)
    return [task["name"] for task in tasks
            if set(task["depends_on"]) <= set(completed)]

naive_done, naive_calls = naive_every_step_replan()
good_done, good_calls, final = gated_sequential_plan_execute()
assert naive_done == good_done == ["查库存", "生成报价", "发送摘要"]
assert (naive_calls, good_calls, final) == (4, 2, "计划完成")
assert dag_parallel_candidates(INITIAL_QUEUE, []) == ["查库存"]
print(f"naive={naive_calls}, gated={good_calls}, final={final}")
```

要点:① 调用数是**初始 1 次 + 每次 gate 通过的 replan**,并非「全程只调一次」。② 此例用 `every=3` 表示批量检查;失败和计划失效会立即触发,阈值应由风险和实测设定。③ `queue.pop(0)` 明确了经典模式的默认顺序语义。④ 只有计划被表达并验证为 DAG、且节点间没有未满足依赖时,才把 `dag_parallel_candidates` 的同批节点交给 [[12 LLMCompiler|LLMCompiler]] 一类并行调度;不能由线性清单推断并行。⑤ `max_steps` 防止执行/replan 无限循环。

## 对比:为什么比 ReAct 省

| 维度 | [[09 ReAct\|ReAct]] | **Plan-and-Execute** |
|---|---|---|
| 规划时机 | 隐式,**每步**现想 | 显式,初始规划 + 条件/批量 replan |
| 调大模型频率 | 每一步 | 初始规划 + 每次触发的 replan;Executor 是否调模型取决于实现 |
| Token / 延迟 | 历史重发时累计输入会增长 | replan 稀少且 Executor 轻量时可能更省;经典执行仍是顺序的 |
| 默认调度 | 每步根据当前 Observation 决定 | **线性队列 `pop(0)` 顺序执行**;不默认启动并行 worker |
| 并行条件 | 取决于具体工具调用依赖 | 仅在**已验证 DAG**证明节点无未满足依赖时,交给 LLMCompiler 类调度器 |
| 长程稳定性 | 依赖每步实时决策 | 有显式计划做锚,但计划过期时必须 replan |
| 应对意外 | 天然(每步现想) | 靠 **replan** 补回 |
| 适合 | 高不确定、必须边走边看 | **步骤可预排**的长流程 |

[[11 ReWOO|ReWOO]] 将规划、取证、合成进一步解耦;[[12 LLMCompiler|LLMCompiler]] 则将计划表示为任务 DAG。经典 Plan-and-Execute 本身仍是顺序执行;**只有依赖允许**时,才应把并行需求交给 LLMCompiler 的 DAG 调度。

**执行编排模式选型卡**(按任务形态查):

| 任务形态 | 选什么 | 为什么 |
|---|---|---|
| 高不确定、下一步严重依赖上一步真实结果 | **[[09 ReAct\|ReAct]]** | 每步现想最灵活、天然纠错;代价是 token 随步数膨胀、长程易迷失 |
| 步骤大体可预排的长流程(多阶段处理、报告生成) | **Plan-and-Execute** | 初始计划可复用,执行通常顺序进行;用条件/批量 replan 应对偏差 |
| 步骤可预排 + 中间观测无需回灌 | **[[11 ReWOO\|ReWOO]]** | 用占位变量蓝图减少中间模型介入,但中途纠偏能力较弱 |
| 步骤间有明显并行结构、要压延迟 | **[[12 LLMCompiler\|LLMCompiler]]** | 计划为 DAG 且依赖满足时,无依赖步骤可并行执行 |
| 解可枚举、要回溯/择优(博弈、证明、规划) | **[[14 树搜索：ToT 与 LATS\|ToT/LATS]]** | 把循环展开成搜索树多路探索 + 回溯,代价随分支数与评估次数增加 |

> 主线一句话:**ReAct(每步决策)→ Plan-and-Execute(初始计划 + 触发式 replan)→ ReWOO(观测不回灌)/ LLMCompiler(DAG 调度)**。它们不是单向的优劣排序;成本、延迟和纠错能力都取决于任务依赖与 replan 频率。

**输入 token 手算(8 步的假设性例子)。** 设固定前缀 $P=1500$,每步新增的决策摘要/Observation 为 $h=250$ token。**[[09 ReAct|ReAct]] 路线**:第 $i$ 步把前 $i-1$ 步历史重发给大模型,输入约 $P+h(i-1)$,八步累计

$$
\sum_{i=1}^{8}[1500+250(i-1)]=8\times1500+250\times\frac{8\times7}{2}=12000+7000=19000 \text{ token}
$$

**Plan-and-Execute 路线**:初始规划输入设为 $P+300=1800$ token。若 Executor 是纯工具,且上面的确定门控在第 3、6、8 步触发 3 次 replan,每次只带紧凑状态 $P+700=2200$ token,则规划类输入为

$$
1800+3\times2200=8400\text{ token},\qquad \frac{8400}{19000}\approx44\%.
$$

这是**特定假设下的输入量**:没有把工具返回、输出 token、缓存命中或价格折算混进来。若改成每步 replan,则为 $1800+8\times2200=19400$ token,已经高于此例的 ReAct 输入。结论不是固定折扣,而是:只有步骤确实可预排、Executor 足够轻量、且 replan 被可靠门控时,这种拆分才可能省。

## 何时用 / 坑

**该用的场景**:目标明确、**步骤大体可预先排布**的长程任务——多阶段数据处理、研究报告生成、固定 SOP 的自动化流程。先比较实际 replan 率、工具成本和质量,再决定是否采用。

**不该用的场景**:**高度不确定、下一步严重依赖上一步真实结果**的探索式任务;初始计划可能很快过期,频繁 replan 可能抵消节省,应与 [[09 ReAct|ReAct]] 在相同任务集上比较。

**常见坑**:
- **初始计划过乐观**:Planner 没见过中间结果就排全程,常漏掉意外分支。务必保留 replan 弹性,别把计划当死命令。
- **Replan 抖动 / 死循环**:每步都大改计划会反复横跳、不收敛。给 `max_steps` 与 replan 预算,并让 replan 倾向"小修而非重排"。
- **计划与执行脱节**:Executor 执行结果没有结构化回灌给 Replanner,导致它"看不见"现实而瞎排。`done` 上下文必须如实带回。
- **单步过重**:若每个 step 本身又是个复杂探索,Executor 内部可能还得套一个 [[09 ReAct|ReAct]],别把所有复杂度都压到"一步"里。
- **错误步骤被跳过不补救**:某步失败若不触发 replan,后续会建在错误结果上。replan 要能识别失败并插入补救步骤。

## 关键事实

- 本文把长程任务落为**Planner(初始多步计划)+ Executor(通常顺序执行)+ Replan**回路;长程规划须在动作结果不断回灌时处理上下文增长([LangChain, Planning for Agents, 2024-07-20](https://www.langchain.com/blog/planning-for-agents))。
- 在本文的队列/gate 实现中,Planner 调用数 = **初始 1 次 + gate 触发次数**(+可选最终合成);若每步都 replan,就不能宣称只规划一次。这是本文为减少无条件重排设定的工程策略([LangChain, 2024-07-20](https://www.langchain.com/blog/planning-for-agents))。
- Executor 可以是**纯工具、确定性程序或模型**;是否更省取决于 replan 率、上下文、模型/工具价格和缓存,必须实测。将领域约束置于应用架构而非假定通用最优,与 [LangChain, 2024-07-20](https://www.langchain.com/blog/planning-for-agents) 的建议一致。
- 本文的失败/状态失效/批量检查 gate 是可审计的**工程扩展**,不是 LangChain 2024-07-20 原文规定的协议;它负责决定何时把结构化状态交回 Replanner([LangChain, Planning for Agents, 2024-07-20](https://www.langchain.com/blog/planning-for-agents))。
- 它是工程谱系(**BabyAGI**)+ 学术近亲 **Plan-and-Solve Prompting(ACL 2023)**,并以 LangChain 对长程规划的工程讨论作背景,非单篇会议论文([BabyAGI 原仓库, 2023](https://github.com/yoheinakajima/babyagi);[Wang et al., 2023](https://aclanthology.org/2023.acl-long.147/);[LangChain, 2024-07-20](https://www.langchain.com/blog/planning-for-agents))。
- 进阶方向:[[11 ReWOO|ReWOO]](观测不回灌)、[[12 LLMCompiler|LLMCompiler]](仅依赖图证明可并行时用 DAG 调度);后者见 [Kim et al., LLMCompiler, ICML 2024](https://arxiv.org/abs/2312.04511)。

## 框架落地

实现不依赖某个框架名称。核对所选版本能否表达:显式计划状态、**顺序** step 队列、确定性 replan gate、重试/超时、预算上限和可恢复的审计 trace。[LangChain 的长程规划讨论(2024-07-20)](https://www.langchain.com/blog/planning-for-agents) 可作问题背景参考,但本文 gate 是工程设计,集成时应锁版本并回看官方文档。

## 工业界实践

生产实现可使用同样的分层:`目标 → Planner(初始计划) → 顺序计划队列 → Executor → 结构化结果 → 确定触发器 → Replanner(收尾/小修/重排)`。复杂 step 可以内部调用 [[09 ReAct|ReAct]],但那会改变模型调用数,应单独计量。

**规模化与成本/延迟**:不要用「初始计划只调一次」推导成本结论。应在 trace 中分别统计初始 planner、每次 replan、Executor、工具和最终合成的 token/价格/延迟;然后比较 replan 率、完成率和 p95。经典 Plan-and-Execute 的执行延迟通常顺序累加;有明确 DAG 依赖且无依赖节点存在时,才交给 [[12 LLMCompiler|LLMCompiler]] 并行调度。

**可观测与运维**:把原始计划、每步执行结果、触发原因、每次 replan diff、预算和证据来源记成 trace。这样可区分「初始计划错、工具失败、前置条件失效、还是 replan 抖动」,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**踩坑与最佳实践**:
- **别把初始计划当死命令**:Planner 没见过中间结果就排全程,务必保留 replan 弹性。
- **执行结果必须结构化回灌给 Replanner**:否则它"看不见"现实而瞎排(计划与执行脱节)。
- **replan 要能识别失败并插补救步骤**:否则后续步骤建在错误结果上。
- **单步别过重**:若每个 step 本身又是复杂探索,要么拆细,要么 step 内套 ReAct,别把所有复杂度压到"一步"里。
- **给 Planner 工具清单/约束**:让它只排"可执行"的步骤,避免计划里出现没有对应工具的动作。

控制流可读成:`plan → queue.pop(0) → execute → gate ──否→ queue.pop(0); gate ──是→ replan_or_finish → queue.pop(0) / END`。默认没有并行分支;并行是已验证 DAG 的另一路调度决策。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q1:Plan-and-Execute 相比 ReAct 省在哪?为什么?**
标准答:ReAct 会频繁让模型根据历史决定下一步。Plan-and-Execute 先生成初始计划,让 Executor 顺序推进,仅在确定触发器或批量边界调用 Replanner;当 Executor 轻量且 replan 稀少时,它**可能**减少大模型输入。要以同一任务集的成本、完成率、replan 率验证,不能承诺固定节省。
- 追问"执行层为什么能轻量":如果 step 已足够明确,Executor 可以直接调工具或确定性程序;复杂/不确定 step 仍可能需要模型或 ReAct。
- 陷阱:高不确定任务可能频繁 replan,消掉节省;此时与 [[09 ReAct|ReAct]] 做实测对比。

**Q2:三个角色/三段分别干什么?**
标准答:Planner(初始步骤)/ Executor(通常顺序执行,调工具/子 agent/跑代码)/ gate(失败、状态失效、批量、队列耗尽等确定条件)/ Replanner(收尾、保留或更新剩余步骤)。外层 harness 维护计划队列、调度执行、记录触发原因并设最大轮数。
- 追问"replan 一定会重排吗":不一定;它可收尾、原样保留或小修。关键是先由 gate 避免每步无条件调用它。

**Q3:Plan-and-Execute 和 Orchestrator-Workers 什么关系?**
标准答:Plan-and-Execute 可看成"时间维度"的 [[07 Orchestrator-Workers|Orchestrator-Workers]]——Planner 像 orchestrator 拆任务、Executor 像 worker 干活。区别:Orchestrator-Workers 更强调**动态分派/并行**子任务;经典 Plan-and-Execute 是**顺序执行**一张预排好的线性计划。

**Q4:它从哪来的?有没有原论文?**
标准答:不是单篇会议论文,而是工程模式谱系——**BabyAGI(2023,task list + LLM 重排优先级的自治循环)** 与 LangChain/LangGraph 的 plan-execute 示例;学术近亲是 **Plan-and-Solve Prompting(ACL 2023)**,但后者停留在单次 prompt 层面,没有真正的执行/重规划循环。
- 陷阱:把 Plan-and-Solve(纯提示策略)和 Plan-and-Execute(带执行/replan 循环的编排)混为一谈是常见错。

**Q5:它最大的软肋是什么?怎么缓解?**
标准答:**初始计划可能过期**——Planner 看不到中间结果,高不确定任务会偏离。缓解:结构化结果、确定 replan gate、失败时插补救步骤、`max_steps`/预算护栏;若任务本就高度探索式,与 [[09 ReAct|ReAct]] 做离线评估后选择。

## 知识拓展

**谱系定位**:可把它看作 [[09 ReAct|ReAct]] 的一种解耦变体:初始规划与执行分开,再按触发条件重规划。[[11 ReWOO|ReWOO]] 进一步减少中间模型介入;[[12 LLMCompiler|LLMCompiler]] 在**依赖允许**时把计划组织为 DAG 并行。

**两个进阶方向**:
- **减少中间模型介入**:[[11 ReWOO|ReWOO]] 用含占位变量的蓝图将取证与合成解耦;具体节省依模型、trace 和基准而变。
- **降低可并行任务的墙钟时间**:[[12 LLMCompiler|LLMCompiler]] 把计划做成**带依赖的任务 DAG**,仅对无依赖步骤并行执行。

**相关论文/前沿**:
- **Wang et al., _Plan-and-Solve Prompting_(ACL 2023)**——学术近亲,单 prompt 的"先规划再求解"。
- **Yao et al., _Tree of Thoughts_(NeurIPS 2023)**——当"计划"不是单条而是多条候选时,规划层本身可以做树搜索,见 [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]]。
- **多 agent 编排**:lead 规划、worker 执行是相关但不同的扩展,见 [[22 多智能体系统|多智能体系统]] 与 [[07 Orchestrator-Workers|Orchestrator-Workers]]。
- **长程自我改进**:把"执行结果 → 修正计划"做成持续循环,接近 [[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进(Ralph loop)]]。

**边界与反模式**:
- **反模式 1:对高不确定任务硬上 Plan-and-Execute**——初始计划可能快速过期,频繁 replan 可能比 [[09 ReAct|ReAct]] 更贵;先在代表性任务集上验证。
- **反模式 2:每步都大改计划**——replan 抖动、不收敛,既慢又烧钱;让 replan 倾向小修。
- **反模式 3:执行结果不结构化回灌**——Replanner 看不见现实而瞎排,计划与执行彻底脱节。
- **边界**:经典版是顺序执行,无法表达"无依赖步骤并行";一旦任务有明显并行结构,应升级到 [[12 LLMCompiler|LLMCompiler]]。
