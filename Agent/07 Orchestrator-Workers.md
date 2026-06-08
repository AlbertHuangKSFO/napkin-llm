[[07 Orchestrator-Workers|Orchestrator-Workers]] 是**一个编排者 LLM 在运行时动态把任务拆成子任务、派给若干 worker LLM、再综合它们的结果**——关键在「动态」:子任务不是预先写死的。这是 Anthropic《Building Effective Agents》(2024-12)五件套里**最接近 agent** 的一个。

## 本质:运行时才知道要拆成什么
[[06 Parallelization|Parallelization]] 的子任务是开发者**设计期就定死**的;但很多真实任务,**要拆成几块、每块做什么,得看具体输入才知道**。比如「改这个 feature」涉及哪几个文件、改几处,事先无法枚举。

Orchestrator-Workers 把「拆分」这件事本身**交给一个 LLM 在运行时做**:

- **Orchestrator(编排者 LLM)**:理解任务 → **动态分解**成子任务 → 决定派几个 worker、每个干什么 → 收集结果后**综合**成最终答案。
- **Worker(执行者 LLM)**:各自领一个子任务,独立完成,交回结果。

正因为「拆分由模型运行时决定」,它在 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 上**把 workflow 往 agent 推了一步**:仍有固定的「拆—派—合」骨架(像 workflow),但骨架里填什么由模型自主(像 agent)。

## 机制:拆 → 派 → 合
![[Orchestrator-Workers.png]]

1. **接任务**:编排者拿到一个边界不预知的任务。
2. **动态分解**:编排者 LLM **运行时**决定把它拆成哪些子任务、拆几个——子任务的**数量和内容都是模型生成的**,不是代码里枚举的。
3. **派活(扇出)**:每个子任务交给一个 worker LLM(常并发执行)。worker 可带自己的工具、上下文。
4. **回收(扇入)**:各 worker 交回结果。
5. **综合**:编排者 LLM 把所有 worker 的输出**整合**成连贯的最终产物(不是简单拼接,常需消解冲突、补缺、统一风格)。

与 Parallelization 并排看,骨架很像(扇出—扇入),**唯一但根本的差别在第 2 步**:这里的拆分是动态的、由编排者运行时决定的。

## 来源
出自 Anthropic《Building Effective Agents》(2024-12)。文中典型例子:**编码任务**——要改的文件、每个文件改什么,事先不知道,由编排者读懂需求后动态决定派给哪些 worker;**搜索任务**——需要从哪些来源、用哪些查询去搜,运行时才定。文中明确指出此模式与 Parallelization 的区别正是「子任务的动态性」。

## 可跑最小代码(伪代码)
```python
import json, concurrent.futures as cf

def orchestrator_workers(task):
    # 1) 编排者 LLM 运行时动态拆分(子任务数量/内容由模型生成)
    plan = llm(
        f"把下面任务拆成若干独立子任务,输出 JSON 数组,每项是一句子任务描述:\n{task}"
    )
    subtasks = json.loads(plan)            # 例:["改 auth.py 加超时", "改 db.py 加索引", ...]

    # 2) 派活:每个子任务一个 worker,并发执行
    def run_worker(st):
        return llm(f"完成这个子任务并返回结果:\n{st}")
    with cf.ThreadPoolExecutor() as ex:
        results = list(ex.map(run_worker, subtasks))

    # 3) 编排者综合(非简单拼接:消解冲突、统一风格)
    final = llm(
        "把以下各子任务的结果整合成一份连贯的最终交付:\n"
        + "\n---\n".join(f"[{s}]\n{r}" for s, r in zip(subtasks, results))
    )
    return final
```
要点:`subtasks` 的**长度和内容是 LLM 在运行时产出的**——这正是它区别于 Parallelization(那里子任务在代码里写死)的代码层面体现。

## 对比表
| 维度 | Orchestrator-Workers | [[06 Parallelization|Parallelization]] | 真正的[[22 多智能体系统|多智能体系统]] |
|---|---|---|---|
| 子任务来源 | 编排者**运行时动态**生成 | 设计期**预先写死** | 各 agent 自主、可协商 |
| 控制结构 | 中心编排,单向派/收 | 固定扇出扇入 | 去中心 / 对等 / 可循环交互 |
| worker 自主度 | 低,只执行被派的子任务 | 无(就是并行调用) | 高,自己规划、用工具、可再派 |
| 离 agent 多远 | 半步:骨架固定+内容自主 | 远:纯静态编排 | 已是 agent |

## 与真正多体的关系(别混淆)
Orchestrator-Workers **不等于** [[22 多智能体系统|多智能体系统]]。它仍是**中心化、单层、单向**的:编排者派、worker 干、编排者收,没有 worker 之间的对话、协商、循环。

真正的多体里,各 [[26 Sub-agents 与 Agent Teams|sub-agent]] 是更自主的实体:能自己规划、自己用工具、甚至再往下派活、彼此交互。Orchestrator-Workers 是通往那里的**最小一步**——把「拆分」交给模型,但还没把「自主循环」交出去。把它理解成**带动态规划的 workflow**,而非完整 agent,最准确。

## 何时用 / 坑
- **何时用**:任务的**子结构无法预先枚举**,得看具体输入才知道拆成什么。典型:多文件代码修改、开放式研究/搜索、复杂报告撰写。
- **坑一**:编排者拆得不好,后面全崩——拆分质量是命门,给编排者足够的上下文和清晰的拆分约束。
- **坑二**:综合环节最容易塌。worker 各干各的,结果可能冲突/重复/风格不一,综合那步要专门处理,别简单字符串拼接。
- **坑三**:成本与延迟随 worker 数膨胀;给「最多拆几个」设上限。
- **坑四**:别过度工程化。若子任务其实**能预先枚举**,直接用 [[06 Parallelization|Parallelization]],不必请一个编排者 LLM。
- **坑五**:worker 之间如需协作/协商,这个模式不够,该上 [[22 多智能体系统|多智能体系统]]。

## 关键事实
- 五件套里**最接近 agent** 的模式:固定骨架 + 模型动态填内容。
- 与 [[06 Parallelization|Parallelization]] 的**唯一本质差别**:子任务是**运行时动态拆**还是**设计期写死**。这是判断该用哪个的最简问题。
- 它是理解 [[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]] 与 [[22 多智能体系统|多智能体系统]] 的台阶:先掌握「中心派活」,再理解「自主协作」。
- 现实里 Claude Code、Deep Research 类系统的「主 agent 派 sub-agent」本质上是这个模式的工程化加强版。

## 工业界实践
这是五件套里**生产价值最高**的模式之一——所有 Deep Research、复杂代码 Agent 的骨架都是它。

**标杆案例:Anthropic 多智能体研究系统(2025-06 工程博客)。** lead agent(编排者)拿到 query 后**制定策略、动态派 3–5 个 subagent 并行探索**不同方向,每个 subagent **有自己独立的上下文窗口**,最后由 lead 综合 + 单独一遍引用(citation)校对。关键数据:
- 在内部研究评测上**比单 agent Claude Opus 4 强 90.2%**;
- 代价是**约 15 倍于普通对话的 token**;
- 性能归因里,**token 用量本身解释了 80% 的方差**,工具调用数和模型选择是另两个因子。

**15× token 怎么来的(粗算)**。设单 agent 普通对话一轮约耗 $M$ token。Orchestrator-Workers 把成本摊在三处:① lead 拆分 + 综合两端,各读写全上下文,约 $2\sim3\,M$;② 派 4 个 subagent,每个**各带独立上下文**、各跑多轮工具交互,单个就约 $2\sim3\,M$,4 个即 $\sim 10\,M$;③ 末尾单独一遍引用校对约 $1\,M$。合计:

$$
\underbrace{3M}_{\text{lead 拆+合}} + \underbrace{4\times 2.5M}_{\text{4 subagent}} + \underbrace{M}_{\text{引用校对}} \approx 14M
$$

正好落在 Anthropic 报告的 **~15×** 量级。直觉:贵不在 lead,而在「**N 个 subagent 各自重跑一遍完整上下文 + 工具循环**」——这也解释了为何 token 用量解释了 80% 的性能方差:本质是拿 token 买并行推理的广度。

这印证了核心逻辑:**把活分给各有独立上下文的 subagent,等于为并行推理扩容**——单 agent 的上下文塞不下的「广度优先」研究任务,靠 orchestrator-workers 拆开就能做。

**主流框架与服务(具体名 + 定位):**
- **LangGraph**:用图 + supervisor 节点表达编排者,subgraph 当 worker;LangChain 官方有 supervisor / hierarchical 多 agent 模板。
- **OpenAI Agents SDK**(前 Swarm):handoff + agent-as-tool,主 agent 把子任务当工具调用派给子 agent。
- **CrewAI**:以「角色 + 任务 + 流程」建模,hierarchical 流程就是 orchestrator-workers。
- **AutoGen / AG2**:GroupChat + manager 做编排。
- **Claude Code 的 sub-agent / Task 工具**:主 agent 用工具派子任务给隔离上下文的 sub-agent,是这个模式的代码 Agent 化身,见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]、[[29 Deep Research Agent|Deep Research Agent]]。

**典型架构:** 编排者(强模型,负责拆分 + 综合)→ 动态派 N 个 worker(可用便宜模型,各带工具/子上下文,并发执行)→ 回收 → 编排者综合(消解冲突 + 统一风格)。worker 隔离上下文是关键——既扩容又防止互相污染。

**规模化与成本/延迟:** 成本和延迟随 worker 数**线性甚至超线性**膨胀(15x token 不是夸张)。必须给「**最多拆几个**」设硬上限,并对 worker 设超时 + 部分结果兜底。延迟靠并发压(总时间≈最慢 worker + 编排两端),但编排者的「拆分」和「综合」两端是串行瓶颈,综合阶段尤其慢(要读全部 worker 输出)。

**可观测与运维:** 这是最难调试的模式——必须有**全链路 tracing**(LangSmith、Langfuse、OpenTelemetry GenAI):记录编排者拆出的子任务、每个 worker 的输入输出/耗时/成败、综合的最终裁决。要监控**拆分质量**(拆出来的子任务是否真独立、是否覆盖全)和**综合质量**(有无丢信息/冲突没消解)。错误会沿「拆→派→合」放大,定位要能逐层下钻,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**踩坑与最佳实践:**
- 给编排者**充足上下文 + 明确拆分约束**(独立性、粒度、数量上限),拆不好后面全崩。
- 综合**绝不能简单字符串拼接**:要专门 prompt 消解 worker 间的冲突、重复、风格不一。
- worker prompt 里写清「你只负责这一小块,别越界」,防 worker 各自扩张任务。
- 引用/事实核对单独成一遍(像 Anthropic 那样),别混进综合那步。

```python
# 生产化:编排者动态拆 + 数量上限 + worker 并发 + 独立综合
def orchestrator_workers(task, max_workers=5):
    plan = llm(f"把任务拆成至多 {max_workers} 个【相互独立】的子任务,"
               f"输出 JSON 数组:\n{task}", model=STRONG)   # 编排用强模型
    subtasks = json.loads(plan)[:max_workers]               # 硬上限兜底
    with cf.ThreadPoolExecutor() as ex:                     # worker 并发
        results = list(ex.map(lambda s: llm(s, model=CHEAP, tools=TOOLS), subtasks))
    return llm("整合为连贯交付,消解冲突、统一风格,勿简单拼接:\n"
               + format(subtasks, results), model=STRONG)   # 综合用强模型
```

## 面试高频
**Q1:Orchestrator-Workers 和 Parallelization 唯一的本质区别是什么?**
标准答:**子任务的来源**——Parallelization 子任务**设计期写死**(数量内容固定);Orchestrator-Workers 子任务**编排者 LLM 运行时动态生成**(看具体输入才知道拆成什么)。骨架(扇出扇入)几乎一样,差别全在「拆分这步是代码枚举还是模型生成」。这是判断该用哪个的最简一问。

**Q2:它算 agent 吗?在 workflow-agent 谱系的什么位置?**
标准答:**半步 agent**——有固定的「拆—派—合」骨架(像 workflow),但骨架里填什么(拆成几个、各干啥)由模型自主(像 agent)。把它理解成「**带动态规划的 workflow**」最准确。它没把「自主循环」交出去(worker 只执行不再规划),所以还不是完整 agent。

**Q3:它和真正的多智能体系统差在哪?**
标准答:Orchestrator-Workers 是**中心化、单层、单向**——编排者派、worker 干、编排者收,worker 之间**没有对话/协商/循环**。真正的多体里各 sub-agent 更自主:能自己规划、用工具、再往下派、彼此交互。前者是通往后者的最小一步,见 [[22 多智能体系统|多智能体系统]]、[[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]]。
- 追问「worker 之间要协作怎么办?」→ 这个模式不够,该上多智能体。

**Q4:为什么多 agent 系统这么贵(15x token),还值得用?**
标准答:Anthropic 数据显示 token 用量解释了 80% 的性能方差——多 agent 本质是**用 token/算力换并行推理容量**。值得用的场景是「**广度优先**、信息量超单一上下文窗口」的研究/搜索类任务;不值得的是窄而深、串行依赖强的任务(那时多 agent 反而因协调开销变差)。

**陷阱题:「让一个 LLM 把任务拆成 JSON 子任务再分别执行」是什么模式?** → Orchestrator-Workers,不是 Parallelization——因为子任务是模型运行时生成的。

## 知识拓展
- **前沿(2025):** Anthropic《How we built our multi-agent research system》(2025-06)是这个模式工业化的权威读物,给出了 90.2% 提升 / 15x token / token 解释 80% 方差等硬数据。**Plan-Execute** 类架构(见 [[10 Plan-and-Execute|Plan-and-Execute]])是它的近亲——先规划再分步执行,区别是 plan-execute 更强调「计划-执行-重规划」的循环。
- **拆分的两种风格:** ① **静态计划**:编排者一次拆完全部子任务再派(适合可一眼看清结构的任务);② **增量拆分**:边做边拆,根据已完成结果决定下一批(更鲁棒,但更慢、更难控)。增量拆分已逼近 agent 自主循环。
- **边界与反模式:** ① 子任务其实**能预先枚举**却请一个编排者 LLM → 过度工程,直接用 [[06 Parallelization|Parallelization]];② 拆分粒度过细 → worker 太多、综合塌方、成本爆炸;③ worker 间有强依赖却用单向派活 → 拿到过期输入,应改串行或上多体;④ 综合环节图省事字符串拼接 → 冲突没消解,最常见的塌方点。
- **与兄弟模式:** 现实系统里它常作为骨架,内部嵌 [[04 Prompt Chaining|Prompt Chaining]](每个 worker 内部一条链)或 [[08 Evaluator-Optimizer|Evaluator-Optimizer]](综合后再评估打磨)。成本/延迟系统化优化见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]];部署与持久化(长任务、断点续跑)见 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]。
