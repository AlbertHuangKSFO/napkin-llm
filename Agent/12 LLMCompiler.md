**LLMCompiler** 把一组工具调用先**编译成一张 DAG(有向无环图)**,识别出彼此无依赖的任务**并行**发出执行,而不是像 [[09 ReAct|ReAct]] 那样一次只调一个工具、串行地等一个回来再调下一个。

它和 [[11 ReWOO|ReWOO]]、[[10 Plan-and-Execute|Plan-and-Execute]] 同属「先规划、再执行」一脉,但更进一步:[[11 ReWOO|ReWOO]] 把整条计划写成线性蓝图,LLMCompiler 把计划写成**带依赖关系的图**,从而把 [[06 Parallelization|Parallelization]] 这种并发能力直接编进 agent 的执行引擎里。本质是借了编译器的思路——**先把任务编译成依赖图,再让调度器最大化并行**。

## 本质:为什么要编译成 DAG

ReAct 式回路的致命低效在于**串行**:每调一个工具,都要等 LLM 生成 → 工具返回 → 再喂回 LLM 生成下一步。如果一个任务需要 5 次工具调用,哪怕这 5 次彼此无关,也得排成 5 个 round-trip,延迟和 token 成本线性叠加。

LLMCompiler 的观察是:**很多工具调用之间根本没有数据依赖**。「查 A 的身高」和「查 B 的身高」可以同时发;只有「比较谁更高」才必须等前两个回来。把这种依赖关系显式画成 DAG,无依赖的节点就能塞进同一批并发执行。

- **节点** = 一次工具调用(带参数,参数里可引用其他节点的输出,如 `$1`、`$2`)。
- **边** = 数据依赖(`$3 = compare($1, $2)` 意味着 $3 依赖 $1 和 $2)。
- **并行** = 所有入度已满足(依赖都就绪)的节点,可在同一轮一起发出。

![[LLMCompiler.png]]

## 机制:四个组件,分步讲透

### 1. Planner —— 一次性产出整张 DAG
LLM 读用户任务,**一次**生成一个流式的任务计划(不是一步步生成)。计划里每个任务形如 `$k = tool(args...)`,参数中用 `$j` 占位引用尚未执行完的上游任务输出。Planner 不等任何工具返回,直接把整张依赖图吐出来——这是它和 [[09 ReAct|ReAct]] 最本质的区别:**规划与执行解耦**。原论文还做了「计划流式化」优化:Planner 一边生成任务一边就交给下游,不必等整张图生成完。

### 2. Task Fetching Unit —— 解析依赖、就绪即发
这是调度核心。它维护每个任务的依赖状态:
- 监听 Planner 的输出流,把任务入队;
- 解析每个任务的 `$k` 占位符,确定它依赖哪些上游;
- 一旦某任务的所有依赖都已拿到结果,就**把占位符替换成真实值**,标记为「就绪」,立即推给 Executor;
- 收到 Executor 回填的结果后,更新下游任务的就绪状态,形成流水线。

### 3. Executor —— 并行执行就绪任务
用一个**异步/线程池**执行器,把所有「就绪」任务真正并发跑起来。无依赖的任务在同一时刻一起发出,各自完成后把结果回写给 Task Fetching Unit,解锁下游。这一步把 [[06 Parallelization|Parallelization]] 从「应用层手动 fan-out」下沉成了「引擎自动调度」。

### 4. Joiner(可选)—— 汇总或决定重规划
所有任务跑完后,Joiner(一个 LLM 调用)看着结果做两件事之一:
- **收尾**:信息够了,直接合成最终答案返回用户;
- **重规划(replan)**:发现还缺信息(比如中途某工具失败、或需要基于初步结果展开新一轮查询),就把当前结果回灌给 Planner,生成新的 DAG,再跑一轮。

有了 Joiner,LLMCompiler 也能处理**动态、多轮**的任务,而不只是一次性静态图。

![[LLMCompiler-流水线.png]]

## 原论文

- **Kim, Moon, Hooper, Gholami et al., _An LLM Compiler for Parallel Function Calling_**(UC Berkeley,arXiv 2023-12,ICML 2024)。
- 论文核心主张:在需要多个工具调用的任务上,相比 ReAct 可达到 **~3.7× 延迟加速、~6.7× 成本降低、~9% 准确率提升**(数据随基准而异),并支持开源模型做并行 function calling(不依赖 OpenAI 的原生 parallel function calling)。
- 思想直接类比传统编译器:Planner≈前端生成 IR(中间表示=DAG),Task Fetching Unit≈指令调度器,Executor≈乱序执行单元。

**延迟加速从哪来(关键路径手算)。** 加速的本质是:串行延迟 $\approx$ **所有步骤之和**,并行延迟 $\approx$ **DAG 关键路径(最长依赖链)**。举「查 6 个实体身高、再两两比较、最后汇总」:设每次工具往返(含 LLM 一轮)$\approx0.8\text{s}$。**[[09 ReAct|ReAct]] 串行**:6 次查 + 3 次比较 + 1 次汇总 = 10 个往返顺序排,$10\times0.8=8.0\text{s}$。**LLMCompiler 并行**:6 次查互无依赖,**一批同发**($1$ 层);3 次比较各依赖 2 个查结果,这 3 个比较也互不依赖,**再一批同发**($1$ 层);最后汇总 $1$ 层——关键路径深度只有 $3$ 层,$3\times0.8=2.4\text{s}$。**$8.0/2.4\approx3.3\times$ 加速**,与论文 $\sim3.7\times$ 同量级。注意上限被关键路径锁死:若任务退化成纯链(每步依赖上一步),关键路径 $=$ 全部步数,DAG 与串行**无差别**——这就是为什么强串行/强探索任务用 LLMCompiler 没有收益,该回 [[09 ReAct|ReAct]]。

## 可跑最小代码(伪代码)

```python
# 1) Planner 产出的 DAG —— 每个任务声明它依赖的上游
plan = [
    Task(id=1, tool="search", args=["A 的身高"], deps=[]),        # 根,无依赖
    Task(id=2, tool="search", args=["B 的身高"], deps=[]),        # 根,无依赖 → 可与 1 并行
    Task(id=3, tool="compare", args=["$1", "$2"], deps=[1, 2]),   # 依赖 1、2
]

# 2) Task Fetching Unit + Executor:就绪即并发发出
import concurrent.futures as cf
results = {}
pending = {t.id: t for t in plan}
with cf.ThreadPoolExecutor() as pool:
    while pending:
        ready = [t for t in pending.values()
                 if all(d in results for d in t.deps)]   # 依赖全部就绪
        futures = {}
        for t in ready:
            args = [results[int(a[1:])] if str(a).startswith("$") else a
                    for a in t.args]                      # 替换 $k 占位符
            futures[pool.submit(TOOLS[t.tool], *args)] = t  # 同批并发提交
            del pending[t.id]
        for fut in cf.as_completed(futures):              # 回填结果,解锁下游
            t = futures[fut]
            results[t.id] = fut.result()

# 3) Joiner:汇总 or 触发重规划
answer = joiner_llm(task, results)        # 够了→收尾;不够→ replan 生成新 DAG 再跑
```

要点:`while pending` 这层循环每轮把当前所有就绪任务**一次性并发提交**,这就是「并行 function calling」的引擎实现。

## 对比:ReAct / ReWOO / LLMCompiler

| 维度 | [[09 ReAct\|ReAct]] | [[11 ReWOO\|ReWOO]] | LLMCompiler |
|---|---|---|---|
| 规划时机 | 边走边想(无全局规划) | 一次性出**线性**蓝图 | 一次性出**DAG**(带依赖) |
| 执行结构 | 串行,逐个调 | 蓝图按序执行(变量回填) | **就绪即并发** |
| 并行能力 | 无 | 有限(线性蓝图难表达分叉) | **原生并行**(无依赖任务同发) |
| 中途纠错 | 强(每步看反馈) | 弱(蓝图定死) | 中(Joiner 可重规划) |
| token 成本 | 高(反复读历史) | 低(规划与执行分离) | 低 + 快(并行省 round-trip) |
| 适用 | 强探索、步骤不可预定 | 步骤可预定、省 token | 多工具、有并行结构、要低延迟 |

## 何时用 / 坑

✅ **该上 LLMCompiler**:任务需要**多个工具调用**且其中存在**可并行的无依赖子任务**(多源信息聚合、批量查询、扇出检索),且你在乎延迟和成本。典型如「对比 N 个实体」「同时查多个 API 再综合」。

❌ **别上**:任务本质是**强串行、强探索**的(每一步都依赖上一步的观测来决定下一步该干嘛),这种用 [[09 ReAct|ReAct]] 更自然——硬编译成 DAG 反而因为缺乏中途反馈而更脆。

坑:
- **Planner 出的 DAG 错了就全错**:依赖关系判断错(本该并行的标成串行,或本该串行的标成并行)会直接拖垮效果或导致用到未就绪的 `$k`。Planner 的 prompt/few-shot 质量是上限。
- **并行不等于免费**:同时打多个外部 API 可能撞限流(rate limit)、放大失败面;Executor 需要做并发上限与重试。
- **依赖解析的健壮性**:`$k` 占位符替换、循环依赖检测要做对,否则死锁。
- **Joiner 的重规划可能震荡**:replan 没有收敛条件会反复重规划烧预算,需设最大轮数。

## 关键事实速记

- LLMCompiler = **Planner(出 DAG)+ Task Fetching Unit(解析依赖、就绪即发)+ Executor(并行)+ Joiner(汇总/重规划)**。
- 核心收益是**并行 function calling**:把无依赖的工具调用塞进同一批并发,省 round-trip → 更快、更省。
- 与 [[11 ReWOO|ReWOO]] 的差别:ReWOO 是线性蓝图,LLMCompiler 是**带依赖的图**,因此能表达并行分叉。
- 与 [[09 ReAct|ReAct]] 的差别:用「中途反馈能力」换「并行与速度」;探索性任务别硬套。
- 它把 [[06 Parallelization|Parallelization]] 从应用层 fan-out 下沉成了 agent 执行引擎的内建调度。

## 主流开源实现 / Python 库

- **`SqueezeAILab/LLMCompiler`** —— **原论文官方实现**(UC Berkeley,ICML 2024),完整含 Planner / Task Fetching Unit / Executor / Joiner 四组件与流式规划,`run_llm_compiler.py` 可直接复现论文的并行 function calling;不依赖 OpenAI 原生 parallel function calling,开源模型也能跑。要对齐论文数值用它。
- **`langchain-ai/langgraph` LLMCompiler 教程** —— 官方 tutorial(`examples/llm-compiler/LLMCompiler.ipynb`),用 LangGraph 把 DAG 调度搭成状态图,**最适合拿来改造进自己工程**的工程模板。

首选:做工程集成选 LangGraph 教程版(易接现有 LangChain 工具);要严谨复现论文用 `SqueezeAILab/LLMCompiler`。无独立通用 pip 包,一般以上述两种形态落地。

## 工业界实践

LLMCompiler 的核心思想——**把无依赖的工具调用并行发出以压延迟**——在工业界已经部分被**平台原生能力**吸收(provider 的 parallel tool calls),但完整的"编译成 DAG + 调度器"形态仍用在那些**多工具、有明显并行结构、且延迟敏感**的生产场景。

**主流落地形态(具体名 + 定位)**:
- **OpenAI / Anthropic / Gemini 的 parallel tool calls**:模型一轮里直接吐多个 `tool_calls`,runtime 并发执行。这是 LLMCompiler"无依赖并行"思想的**平台内建版**,无需自己搭 Task Fetching Unit——但它只能并行"同一轮模型已决定的调用",**跨轮、带复杂依赖的 DAG 调度**仍需 LLMCompiler 这种显式编排。
- **LangGraph LLMCompiler 教程图**:把 Planner / Task Fetching Unit / Executor / Joiner 搭成状态图,适合接现有 LangChain 工具、需要显式 DAG 调度的工程。
- **`SqueezeAILab/LLMCompiler`(官方实现)**:含流式规划,不依赖 OpenAI 原生 parallel function calling,**开源模型也能跑并行 function calling**——这点在自托管场景很关键。
- **多源信息聚合服务**:典型如"同时查 N 个 API / 检索 N 个源再综合"的后端(行情聚合、多平台比价、扇出检索),正是 LLMCompiler 的甜点区,见 [[06 Parallelization|Parallelization]]、[[29 Deep Research Agent|Deep Research Agent]]。

**典型架构**:`Planner(流式吐 DAG,$k 占位引用上游)→ Task Fetching Unit(解析依赖、就绪即替换占位符并发提交)→ Executor(线程池/异步并发执行就绪任务)→ Joiner(汇总 or replan)`。类比编译器:Planner≈前端生成 IR(DAG),Task Fetching Unit≈指令调度器,Executor≈乱序执行单元。

**规模化与成本/延迟**:论文报告相比 ReAct 可达 **~3.7× 延迟加速、~6.7× 成本降低、~9% 准确率提升**(随基准而异)。生产里收益主要来自**省 round-trip**:把 5 个无依赖工具调用从 5 个串行往返压成 1 批并发。但"并行不等于免费"——同时打多个外部 API 容易撞**限流(rate limit)**、放大失败面,Executor 必须做**并发上限 + 重试 + 退避**。延迟收益的上限由 DAG 的**关键路径(最长依赖链)**决定:若任务本质强串行(每步依赖上一步),DAG 退化成一条链,LLMCompiler 没优势。

**可观测与运维**:DAG 本身是一张可视化的执行图,适合做可观测——把每个节点的工具名、参数、依赖、并发批次、耗时记成 span,排障时直接看"哪个节点拖慢了关键路径 / 哪批并发撞了限流 / Joiner 是否在反复 replan"。生产监控重点:**并发实际命中率**(有多少任务真的并行了)、**关键路径耗时**、**replan 轮数**,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。

**踩坑与最佳实践**:
- **Planner 的 DAG 质量是上限**:依赖判断错(该并行标成串行、或该串行标成并行 → 用到未就绪的 `$k`)会直接拖垮效果。靠高质量 prompt / few-shot,且执行前做**循环依赖检测 + 拓扑校验**。
- **Executor 必设并发上限与重试**:防限流和级联失败。
- **占位符替换要稳健**:`$k` 替换、依赖回填出错会死锁。
- **Joiner 的 replan 要设最大轮数**:否则没有收敛条件会反复重规划烧预算(震荡)。
- **流式规划权衡**:Planner 边生成边交付能更早启动执行,但实现更复杂;简单任务不必上流式。

```python
# Task Fetching Unit + Executor:就绪即并发(核心引擎)
import concurrent.futures as cf
results, pending = {}, {t.id: t for t in plan}
with cf.ThreadPoolExecutor(max_workers=N) as pool:   # 并发上限防限流
    while pending:
        ready = [t for t in pending.values()
                 if all(d in results for d in t.deps)]    # 依赖全就绪
        futures = {}
        for t in ready:
            args = [results[int(a[1:])] if str(a).startswith("$") else a
                    for a in t.args]                       # 替换 $k 占位符
            futures[pool.submit(TOOLS[t.tool], *args)] = t # 同批并发提交
            del pending[t.id]
        for fut in cf.as_completed(futures):               # 回填,解锁下游
            results[futures[fut].id] = fut.result()
answer = joiner_llm(task, results)        # 够了收尾;不够 replan 生成新 DAG(设 max_rounds)
```

## 面试高频

**Q1:LLMCompiler 解决 ReAct 的什么问题?核心机制?**
标准答:ReAct 串行——每调一个工具都要等"LLM 生成 → 工具返回 → 再喂回 LLM",N 次调用就是 N 个 round-trip,延迟/成本线性叠加,哪怕这些调用彼此无关。LLMCompiler 把工具调用**编译成 DAG**,识别无依赖任务**并行**发出,省掉 round-trip,论文报告 ~3.7× 加速、~6.7× 降本。
- 追问"为什么类比编译器":Planner≈前端生成 IR(DAG),Task Fetching Unit≈指令调度器,Executor≈乱序执行单元——和 CPU 把无依赖指令乱序并行执行同构。

**Q2:四个组件分别干什么?**
标准答:Planner(一次/流式产出整张 DAG,任务形如 `$k = tool(args)`,参数用 `$j` 占位引用上游)/ Task Fetching Unit(解析依赖、就绪就把占位符换成真实值并推给 Executor)/ Executor(异步/线程池并发执行就绪任务,结果回写解锁下游)/ Joiner(可选,汇总收尾 or 触发 replan 生成新 DAG)。
- 追问"Joiner 有什么用":让 LLMCompiler 能处理**动态、多轮**任务而不只是一次性静态图。

**Q3:LLMCompiler 和 ReWOO 的区别?**
标准答:两者都用"带依赖的变量计划"、都是先规划后执行。区别:[[11 ReWOO|ReWOO]] 是**线性蓝图**(难表达并行分叉),主打**省 token**;LLMCompiler 是**带依赖的 DAG**,能表达并行分叉,主打**降延迟**(原生并行 + Joiner 可重规划)。
- 陷阱:别说"LLMCompiler 一定更省 token"——它的主轴是延迟;省 token 是规划/执行分离的副产物。

**Q4:什么时候不该用 LLMCompiler?**
标准答:任务**强串行、强探索**(每步都依赖上一步观测决定下一步)时别用——DAG 退化成一条链,没有并行收益,且缺中途反馈更脆,该用 [[09 ReAct|ReAct]]。LLMCompiler 的甜点是"多工具 + 存在无依赖可并行子任务 + 延迟敏感"。

**Q5:并行带来哪些工程风险?**
标准答:① 撞**限流**、放大失败面 → Executor 要并发上限 + 重试退避;② Planner 把依赖判错 → 用到未就绪 `$k` 或丢掉本可并行的机会,需拓扑/循环依赖校验;③ Joiner 无收敛条件 → replan 震荡烧预算,设 max_rounds。

## 知识拓展

**谱系定位**:这是"减少大模型介入次数 / 降本提速"主线的**终点**:[[09 ReAct|ReAct]](每步串行问)→ [[10 Plan-and-Execute|Plan-and-Execute]](规划集中一次)→ [[11 ReWOO|ReWOO]](观测不回灌、省 token)→ **LLMCompiler(编译成 DAG、并行降延迟)**。它把 [[06 Parallelization|Parallelization]] 从"应用层手动 fan-out"**下沉成 agent 执行引擎的内建调度**——这是它最独特的贡献。

**延伸/进阶**:
- **平台内建并行**:provider 的 parallel tool calls 已把"单轮无依赖并行"做进 API,但**跨轮、复杂依赖的 DAG 调度**仍是 LLMCompiler 的领地;理解二者边界很重要。
- **与多 agent 的关系**:LLMCompiler 在"单 agent 内并行工具调用",而 [[22 多智能体系统|多智能体系统]] / [[07 Orchestrator-Workers|Orchestrator-Workers]] 在"多个 agent 并行干子任务"——前者是引擎级并发,后者是组织级并发,可叠加。
- **更动态的图**:Joiner 的 replan 让静态 DAG 能演化成多轮;再往前是把规划本身做成搜索(见 [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]])。

**相关论文/前沿**:
- **Kim, Moon, Hooper, Gholami et al., _An LLM Compiler for Parallel Function Calling_(UC Berkeley,arXiv 2023-12,ICML 2024)**——本范式原点,首次把编译器调度思路引入 agent。
- **OpenAI parallel function calling(2023-11 起)/ Anthropic parallel tool use(2024)**——平台把"单轮并行"内建,改变了手搓 LLMCompiler 的性价比。
- **DAG 调度 + Agentic RAG**:多源并行检索再合成,见 [[36 Agentic RAG|Agentic RAG]]、[[29 Deep Research Agent|Deep Research Agent]]。

**边界与反模式**:
- **反模式 1:对强串行/强探索任务硬编译成 DAG**——退化成链且缺中途反馈,比 ReAct 更脆。
- **反模式 2:无并发上限地狂打外部 API**——撞限流、放大失败,反而更慢更不稳。
- **反模式 3:Joiner replan 不设收敛条件**——震荡烧预算。
- **边界**:延迟收益上限由 DAG **关键路径**决定;若关键路径就是整个任务长度(纯串行),LLMCompiler 无加速空间。当 provider 原生 parallel tool calls 已够用时,优先用它而非自搭调度器。
