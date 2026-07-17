[[07 Orchestrator-Workers|Orchestrator-Workers]] 是一种**中心编排拓扑**：编排者在运行时动态拆解任务、委派 worker、再综合结果。Anthropic 将这一工作流的关键差异归为「子任务由编排者按具体输入决定，而非预先写死」；它不是对 worker 是否具备 agent 能力的判定。

## 本质:运行时才知道要拆成什么
[[06 Parallelization|Parallelization]] 的子任务是开发者**设计期就定死**的;但很多真实任务,**要拆成几块、每块做什么,得看具体输入才知道**。比如「改这个 feature」涉及哪几个文件、改几处,事先无法枚举。

Orchestrator-Workers 把「拆分」这件事本身**交给一个 LLM 在运行时做**:

- **Orchestrator(编排者 LLM)**:理解任务 → **动态分解**成子任务 → 决定派几个 worker、每个干什么 → 收集结果后**综合**成最终答案。
- **Worker(执行者)**:各自领一个子任务并交回结果。它可以是一次 LLM 调用、带工具的 agent、或更受限的函数；「worker」描述的是在拓扑中的位置，不是智力或自治程度。

因此应分开问两件事：**拓扑**是否中心化（谁拥有派工与综合权），以及**节点自治**有多强（worker 是否会规划、调用工具、迭代或再委派）。固定「拆—派—合」骨架可以是 workflow；若编排者或 worker 以环境反馈驱动自己的多轮工具循环，则那些节点也可具备 agentic 行为，不能只凭名称下结论。

**生活类比:** 像项目经理接到“改线上登录并提速”的需求后，先读需求再临时决定要找认证、数据库、测试三位同事；任务数随问题而变。若每天无论什么需求都固定派“写文档、查日志”两项，那只是静态并行，不是动态编排。

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

## 可跑最小代码
```python
# 本地可运行的“编排者替身”；生产中 plan_subtasks 可由 LLM 产生结构化计划。
def run_worker(subtask):
    return f"完成：{subtask}"

# ❌ 朴素写法：无论输入是什么，都跑设计期写死的任务清单。
def static_orchestrate(task):
    fixed_subtasks = ["检查日志", "写变更摘要"]
    return [run_worker(item) for item in fixed_subtasks]

def plan_subtasks(task):
    planned = []
    if "登录" in task:
        planned.append("检查认证流程")
    if "提速" in task:
        planned.append("分析慢查询")
    return planned or ["澄清需求"]

# ✅ 改进写法：按当前输入动态生成子任务，中心节点再回收结果。
def dynamic_orchestrate(task):
    subtasks = plan_subtasks(task)
    results = [run_worker(item) for item in subtasks]
    return {"subtasks": subtasks, "results": results}

task = "修复登录失败并提速"
static = static_orchestrate(task)
dynamic = dynamic_orchestrate(task)
assert static == ["完成：检查日志", "完成：写变更摘要"]
assert dynamic["subtasks"] == ["检查认证流程", "分析慢查询"]
print(dynamic)
```
要点:`subtasks` 的**长度和内容是 LLM 在运行时产出的**——这正是它区别于 Parallelization(那里子任务在代码里写死)的代码层面体现。

## 对比表
| 维度         | 中心化 Orchestrator-Workers | [[06 Parallelization\|Parallelization]] | 对等型 [[22 多智能体系统\|多智能体系统]] |
| ---------- | -------------------- | --------------------------------------- | ------------------------ |
| 子任务来源      | 编排者可**运行时动态**生成       | 设计期**预先写死**                             | 可由任一 peer 决定/协商           |
| 控制结构       | 中心节点拥有派工与综合权           | 固定扇出扇入                                  | agent 之间可直接 handoff / 协作         |
| worker 自主度 | **可低可高**：函数、LLM 或工具 agent | 通常是预设调用                               | **可低可高**：由具体实现决定           |
| 是否多智能体 | 若多个节点是 agent，即可属于中心化多智能体 | 不由拓扑单独决定 | 若多个 peer 是 agent，即可属于对等多智能体 |

## 中心化与对等：都是可能的多智能体拓扑

不要把「中心化」误叫成“假的多智能体”。如果编排者与多个 worker 都是 agent，中心编排同样是多智能体系统；差异在于**控制权与通信边**。OpenAI 将多智能体概括为两类：

- **中心化 manager（agents as tools）**：manager 决定调用哪个 worker、给什么输入、如何综合；worker 通常把结果返给 manager，用户面与全局状态由中心掌握。
- **对等/去中心化 handoff**：多个 agent 彼此可以转交控制权；没有一个节点必须统一综合或始终面向用户。

中心 worker 也完全可以是 agentic：它可在自己的受限上下文中规划、使用工具、验证结果后再返回；只要协议规定最终控制权仍回到编排者，拓扑依旧是中心化。是否允许 worker 互相对话、再委派或直接接管用户会改变通信协议与治理难度，但不会决定它「算不算真的多智能体」。参见 [[22 多智能体系统|多智能体系统]]、[[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]]。

## 何时用 / 坑
- **何时用**:任务的**子结构无法预先枚举**,得看具体输入才知道拆成什么。典型:多文件代码修改、开放式研究/搜索、复杂报告撰写。
- **坑一**:编排者拆得不好,后面全崩——拆分质量是命门,给编排者足够的上下文和清晰的拆分约束。
- **坑二**:综合环节最容易塌。worker 各干各的,结果可能冲突/重复/风格不一,综合那步要专门处理,别简单字符串拼接。
- **坑三**:成本与延迟随 worker 数膨胀;给「最多拆几个」设上限。
- **坑四**:别过度工程化。若子任务其实**能预先枚举**,直接用 [[06 Parallelization|Parallelization]],不必请一个编排者 LLM。
- **坑五**:worker 之间如需协作/协商，先明确是否真的需要对等 handoff；若只需中心协调下的多轮回收—再派发，可仍保持中心化。扩展通信边前要评估共享状态、权限与可观测性成本。

## 关键事实
- Anthropic 的 Orchestrator-Workers workflow 将「子任务按输入动态生成」作为它相对 [[06 Parallelization|Parallelization]] 的关键特征；但实际系统还要考虑依赖、权限和汇总协议。[Anthropic, 2024](https://www.anthropic.com/engineering/building-effective-agents)
- 「中心化 / 对等」与「worker 是否 agentic」是正交维度。中心 manager 与对等 handoff 都可构成多智能体系统。[OpenAI, 2025](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- 对复杂任务，优先选择能满足控制、可解释性与成本约束的最小拓扑；不能仅因任务用了多个 LLM 就判断架构优劣。

## 工业界实践
这是复杂研究或开放式任务中常见的一种拓扑，但不是所有 Deep Research 或代码 agent 的唯一骨架。若任务共享状态很多、依赖强或缺乏可并行分块，中心多 agent 可能不合适。

**案例边界：Anthropic 多智能体研究系统（2025）。** 该系统采用 lead agent + 并行 subagent 的中心化多智能体架构；subagent 会自行搜索、评估工具结果并把发现交回 lead，lead 还可决定是否继续研究。Anthropic 报告的是**其内部研究评测与其实现**：相对指定的单 agent 基线提升 90.2%，其多 agent 系统通常比聊天交互多用约 15 倍 token，且在其 BrowseComp 分析中 token 用量单独解释 80% 的性能方差。这些是有条件的案例结果，不应改写成任何任务的预期收益或固定成本。[Anthropic, 2025](https://www.anthropic.com/engineering/multi-agent-research-system)

**成本与延迟的可迁移账本。** 若编排、每个 worker、综合的 token 分别为 $c_o,c_1,\ldots,c_n,c_s$，则

$$
C_{total}=c_o+\sum_{i=1}^{n}c_i+c_s.
$$

当 worker 独立且真并发时，墙钟时间近似为

$$
T\approx t_o+\max_i(t_i)+t_s,
$$

还应加上排队、限流与重试。先量测这两个式中的实际值，再设置 worker 数、超时与部分结果策略；不要从某个公开案例反推自己的 15× 成本。

**小数字手算。** 设编排与综合 token 分别为 $c_o=3,c_s=4$，三名 worker 分别为 $c_1=5,c_2=4,c_3=6$，则

$$
C_{total}=3+(5+4+6)+4=22.
$$

若编排、三名 worker、综合的耗时分别为 $0.5,(1.5,1.0,2.0),0.5$ 秒，独立且可并发时

$$
T_{parallel}=0.5+\max(1.5,1.0,2.0)+0.5=3.0\text{s};
$$

串行则为 $0.5+1.5+1.0+2.0+0.5=5.5\text{s}$。所以该例用相同 22 token 单位换来约 $5.5/3.0\approx1.83$ 倍墙钟加速；真实系统还会受排队、重试与综合上下文长度影响。

**实现不绑定框架:** 关键是协议而不是产品名：谁创建 worker、携带什么上下文、worker 能做哪些工具调用、如何回收结果、何时取消/升级。OpenAI 的官方指南把中心 manager（agent-as-tool）与对等 handoff 分别作为多智能体拓扑；LangChain 的 subagent 文档也将 supervisor 描述为中心化控制，并允许调用带自身工具循环的 subagent。[OpenAI, 2025](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)；[LangChain Docs](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents)。代码 agent 的相关工程化见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]、[[29 Deep Research Agent|Deep Research Agent]]。

**典型架构:** 编排者（拆分 + 调度 + 综合）→ 动态派 $N$ 个 worker（可为函数或 agent，带各自工具/上下文）→ 回收 → 编排者综合。是否隔离上下文、是否允许共享记忆，都应按任务依赖和泄露风险设计。

**规模化与成本/延迟:** worker 数与携带上下文会推高成本；并发可压低墙钟时间，但编排与综合仍是串行部分。必须设置数量上限、超时、权限边界和部分结果策略，并以实际 trace 量测而非套用公开案例数字。

**可观测与运维:** 这是最难调试的模式——必须有**全链路 tracing**(LangSmith、Langfuse、OpenTelemetry GenAI):记录编排者拆出的子任务、每个 worker 的输入输出/耗时/成败、综合的最终裁决。要监控**拆分质量**(拆出来的子任务是否真独立、是否覆盖全)和**综合质量**(有无丢信息/冲突没消解)。错误会沿「拆→派→合」放大,定位要能逐层下钻,见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。

**踩坑与最佳实践:**
- 给编排者**充足上下文 + 明确拆分约束**(独立性、粒度、数量上限),拆不好后面全崩。
- 综合不应在未经检查时简单字符串拼接：要处理 worker 间的冲突、重复、覆盖缺口和风格不一。
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
> 面试地图：[[Agent 面试题库]]
**Q1:Orchestrator-Workers 和 Parallelization 唯一的本质区别是什么?**
标准答:**子任务的来源**——Parallelization 子任务**设计期写死**(数量内容固定);Orchestrator-Workers 子任务**编排者 LLM 运行时动态生成**(看具体输入才知道拆成什么)。骨架(扇出扇入)几乎一样,差别全在「拆分这步是代码枚举还是模型生成」。这是判断该用哪个的最简一问。

**Q2:它算 agent 吗?在 workflow-agent 谱系的什么位置?**
标准答:先说清判据，别只看名字。固定「拆—派—合」是 workflow 骨架；若编排者动态决定委派，或 worker 依据环境反馈多轮规划、用工具、重试，那么相应节点具备 agentic 行为。拓扑不决定自治程度，需看实际控制循环。

**Q3:它和真正的多智能体系统差在哪?**
标准答:两者不是「真/假」关系。Orchestrator-Workers 描述中心化控制：编排者派工并综合；对等多体让 agent 直接 handoff 或协商。worker 能否规划、调用工具、再委派是另一维度。中心化多 agent 适合需要统一用户面、权限与综合的场景；需要直接 peer-to-peer 协作时才考虑扩大通信图，见 [[22 多智能体系统|多智能体系统]]、[[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]]。
- 追问「worker 之间要协作怎么办?」→ 先判断中心协调的回收—再派发是否足够；若确需直接 handoff/共享控制权，再设计对等协议与治理。

**Q4:为什么多 agent 系统可能更贵，还值得用?**
标准答:多个 worker 复制了调用、上下文与协调开销，成本需按 $C_{total}$ 实测。Anthropic 的 15× 与 80% 方差是其研究系统的条件性观测，不是通用常数。它在广度优先、独立方向多、信息超单一上下文或工具繁多的高价值任务中可能值得；强依赖或共享上下文很重的任务常被协调开销抵消。

**陷阱题:「让一个 LLM 把任务拆成 JSON 子任务再分别执行」是什么模式?** → Orchestrator-Workers,不是 Parallelization——因为子任务是模型运行时生成的。

## 知识拓展
- **证据边界:** Anthropic 的多 agent 研究系统说明了中心 orchestrator + 具备工具循环的 subagent 可以共存；它也明确指出强依赖、需要共享全部上下文的领域未必适配多 agent。把公开案例当作设计假设，必须用自己的任务集复验。[Anthropic, 2025](https://www.anthropic.com/engineering/multi-agent-research-system)
- **拆分的两种风格:** ① **静态计划**:编排者一次拆完全部子任务再派(适合可一眼看清结构的任务);② **增量拆分**:边做边拆,根据已完成结果决定下一批(更鲁棒,但更慢、更难控)。增量拆分已逼近 agent 自主循环。
- **边界与反模式:** ① 子任务其实**能预先枚举**却请一个编排者 LLM → 过度工程,直接用 [[06 Parallelization|Parallelization]];② 拆分粒度过细 → worker 太多、综合塌方、成本爆炸;③ worker 间有强依赖却用单向派活 → 拿到过期输入,应改串行或上多体;④ 综合环节图省事字符串拼接 → 冲突没消解,最常见的塌方点。
- **与兄弟模式:** 现实系统里它常作为骨架,内部嵌 [[04 Prompt Chaining|Prompt Chaining]](每个 worker 内部一条链)或 [[08 Evaluator-Optimizer|Evaluator-Optimizer]](综合后再评估打磨)。成本/延迟系统化优化见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]];部署与持久化(长任务、断点续跑)见 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]。
