[[11 ReWOO|ReWOO]](Reasoning WithOut Observation)让 **Planner 一次性产出含占位变量(#E1, #E2…)的完整蓝图**,**Worker** 去把每个变量取成真实证据,**Solver** 最后一次性把证据填回蓝图合成答案——把"推理"与"观测"彻底解耦,中间**不把 observation 反复回灌给大模型**,从而大幅省 token。

它是 [[10 Plan-and-Execute|Plan-and-Execute]] 谱系里"压缩大模型介入次数"做到极致的一支,与 [[09 ReAct|ReAct]] 形成最鲜明的对比。

## 本质:观测不必每次都打扰大模型

[[09 ReAct|ReAct]] 的昂贵在于**交错**:Thought→Action→Observation 每一轮,都要把"到目前为止的全部历史(尤其是冗长的 Observation)"重新塞回 prompt 发给大模型。一个 N 步任务,长长的检索结果会被**重复携带 N 次**,token 爆炸。

ReWOO 的洞见一句话:**模型用来"规划"的推理,和工具返回的"观测",可以分开**。

- 规划时,模型其实**不需要看到真实结果**也能把整条解题链想清楚——它只要知道"第 1 步去查人口、第 2 步用第 1 步的结果算增长"。于是让它一次性写出蓝图,**用占位变量 `#E1`、`#E2` 代替还不知道的真实值**。
- 真实证据由 **Worker** 静默去取,**不经过大模型**——纯执行工具、填变量。
- 直到最后,**Solver** 才把"蓝图 + 全部证据"一次性交给大模型做最终合成。

结果:**大模型总共只被调用两次**(Planner 出蓝图 + Solver 合成),而非 ReAct 的每步一次。冗长的中间 Observation **从不重复回灌**,token 大幅下降。论文报告在多个基准上,ReWOO 用约 ReAct **5 倍更少的 token** 达到相当甚至更好的准确率。

代价同样明确:**蓝图是"盲规划"**——Planner 没看到任何真实结果就排好全程,一旦某步的现实结果与假设大相径庭,蓝图无法中途自纠(它本来就不接收观测)。所以 ReWOO 最适合**步骤可静态规划**的任务。

## 机制:Planner / Worker / Solver 三件套

![[ReWOO.svg]]

1. **Planner(规划者,LLM)**:一次性产出**带占位变量的蓝图**。每条形如:

   ```
   Plan: 查中国人口
   #E1 = Search[中国 最新人口]
   Plan: 据人口算某指标
   #E2 = Calculate[#E1 * 0.05]
   ```

   关键是 `#E2` 的参数里**引用了 `#E1`**——变量之间形成依赖关系,这张蓝图本质是一个**计划图**。

2. **Worker(执行者,非 LLM)**:遍历蓝图,**逐个把 `#En` 求成真实值**。遇到 `#E1 = Search[...]` 就真去调检索器,拿到"14 亿"填进 `#E1`;轮到 `#E2 = Calculate[#E1 * 0.05]` 时,**先把 `#E1` 的真实值代入**(变量替换),再调计算器。**互不依赖的变量可以并行取**。整个取证过程**不调用大模型**。

3. **Solver(求解者,LLM)**:拿到**蓝图 + 完整证据表(所有 #En 的真实值)**,**一次性**推理、合成最终答案。这是第二次、也是最后一次大模型调用。

对比 [[09 ReAct|ReAct]]:ReAct 是 `想→做→看→想→做→看…` 把观测**交错**回灌;ReWOO 是 `想完整蓝图(含变量)→ 静默取全部证据 → 一次合成`,观测**只在末尾汇合一次**。

> 变量替换是 ReWOO 的灵魂:`#E1` 在蓝图阶段是符号占位,在 Worker 阶段被真实值原地替换,使得"规划"完全不必等"观测"。这也是它和 [[12 LLMCompiler|LLMCompiler]] 的共同基因——后者把这种带依赖的变量蓝图进一步显式建成 **DAG** 来调度并行。

## 原论文

**Xu et al., _ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models_**(2023 年 5 月 arXiv)。作者来自 University of Pennsylvania 与 Allen Institute for AI / 等(Binfeng Xu 等)。论文标题里的 **WOO = WithOut Observation**,直白点题:把 observation 从推理循环里拿掉。

核心贡献与结论:
- 提出 **Planner–Worker–Solver** 三模块分解,把"参数推理"与"工具观测"解耦。
- 在 ALFWorld、HotpotQA、TriviaQA 等基准上,相比 ReAct **平均省约 5× token**,准确率持平或更高。
- 还展示了**蒸馏**:用 GPT-3.5 当 Planner 教师,把规划能力蒸馏进 7B 小模型(LLaMA),证明这套分解利于把"工具增强推理"下放到小模型。

## 可跑的最小实现

```python
import re

def planner(task, llm):
    """产出带占位变量的蓝图:[(plan文本, var, tool, arg), ...]"""
    prompt = f"""把任务拆成计划。每步两行:
Plan: <为什么>
#E<n> = <Tool>[<参数,可引用前面的 #E>]
可用 Tool:Search、Calculate。任务:{task}"""
    blueprint = []
    for line in llm(prompt).splitlines():
        m = re.match(r"#E(\d+)\s*=\s*(\w+)\[(.*)\]", line.strip())
        if m:
            blueprint.append((f"#E{m.group(1)}", m.group(2), m.group(3)))
    return blueprint

TOOLS = {"Search": lambda q: ..., "Calculate": lambda e: str(eval(e))}
VAR_RE = re.compile(r"#E\d+")

def worker(blueprint):
    """逐变量取证;取前先把已得的 #E 真实值代入(变量替换)。无依赖者可并行。"""
    evidence = {}
    for var, tool, arg in blueprint:
        arg = VAR_RE.sub(lambda m: evidence.get(m.group(0), m.group(0)), arg)  # 替换
        evidence[var] = TOOLS[tool](arg)          # 不调大模型
    return evidence

def solver(task, blueprint, evidence, llm):
    """末尾一次性合成。"""
    plan_str = "\n".join(f"{v} = {t}[{a}] -> {evidence[v]}" for v, t, a in blueprint)
    return llm(f"任务:{task}\n证据:\n{plan_str}\n据此给出最终答案。")

def rewoo(task, llm):
    bp = planner(task, llm)        # 大模型调用 ①
    ev = worker(bp)               # 纯工具,0 次大模型
    return solver(task, bp, ev, llm)  # 大模型调用 ②
```

全程**大模型只调 2 次**(planner + solver);worker 阶段零大模型、且无依赖的变量可并行取。对比 [[09 ReAct|ReAct]] 的 `while` 循环里每轮一次大模型,省的就是这些中间往返。

## 对比:ReWOO vs ReAct vs Plan-and-Execute

| 维度 | [[09 ReAct|ReAct]] | [[10 Plan-and-Execute|Plan-and-Execute]] | **ReWOO** |
|---|---|---|---|
| 观测如何进入推理 | 每步**交错回灌** | 每步结果回灌给 replanner | **只在 Solver 末尾汇合一次** |
| 大模型调用次数 | ~每步一次 | 规划 1 次 + 偏差时 replan | **固定 2 次**(Plan+Solve) |
| 规划方式 | 隐式逐步 | 显式、可 replan | 显式**蓝图 + 占位变量**,盲规划 |
| 取证能否并行 | 否(串行交错) | 视实现 | **无依赖变量可并行** |
| Token 成本 | 最高 | 较省 | **最省**(论文 ~5×↓) |
| 中途自纠 | 强(每步现看) | 有(replan) | **弱**(蓝图不收观测,错了难纠) |
| 适合 | 高不确定、边走边看 | 步骤可预排的长流程 | **步骤可静态规划**、要省 token |

ReWOO 与 [[12 LLMCompiler|LLMCompiler]] 是近亲:都用"带依赖的变量计划"。区别是 LLMCompiler 进一步把蓝图**编译成显式 DAG**,引入流式规划与更激进的并行调度,主打**降延迟**;ReWOO 更朴素,主打**省 token**。这条"减少大模型介入"的主线([[09 ReAct|ReAct]] → [[10 Plan-and-Execute|Plan-and-Execute]] → [[11 ReWOO|ReWOO]] → [[12 LLMCompiler|LLMCompiler]])会在 [[14 树搜索：ToT 与 LATS|树搜索：ToT 与 LATS]] 里被汇成谱系总表。

## 何时用 / 坑

**该用的场景**:解题路径**可以预先静态排布**、且**中间观测冗长**(长检索片段、大段 API 返回)的任务——此时 ReAct 反复回灌观测最浪费,ReWOO 的省 token 优势最大;以及要把工具增强能力**蒸馏进小模型**的场景。

**不该用的场景**:**下一步严重依赖上一步真实结果、充满意外分支**的探索式任务。ReWOO 盲规划的蓝图在这里会频繁失准,而它又**不接收中途观测来自纠**,表现会差——这种任务用 [[09 ReAct|ReAct]] 更合适。

**常见坑**:
- **盲规划失准**:Planner 没见过真实结果就排全程,某步现实与假设不符时整张蓝图连锁失效,且无法中途修。可加一层"Solver 发现证据矛盾就触发重 plan"的兜底,但那就部分回到 [[10 Plan-and-Execute|Plan-and-Execute]] 了。
- **变量替换出错**:`#E2` 引用 `#E1` 时若替换逻辑漏掉、或 `#E1` 取证失败返回空,会污染后续。Worker 要校验每个变量取证成功再往下。
- **依赖环 / 引用未定义变量**:Planner 偶尔生成 `#E2` 引用尚未定义的 `#E3`,或形成环。执行前需做拓扑校验。
- **Worker 工具调用脆弱**:这层不经大模型纠错,参数格式错就直接失败,需要稳健的解析与重试。
- **错把它当万能省钱键**:对高不确定任务硬上 ReWOO,准确率会掉;省 token 的前提是任务**本来就适合静态规划**。

## 关键事实

- **ReWOO = Reasoning WithOut Observation**:把工具观测从推理循环里拿掉,只在末尾汇合。
- 三件套:**Planner(出含占位变量 #En 的蓝图)/ Worker(非 LLM,逐变量取证 + 变量替换)/ Solver(LLM,一次合成)**。
- 大模型**固定只调 2 次**(Planner + Solver),中间观测**从不重复回灌** → 论文报告比 [[09 ReAct|ReAct]] **省约 5× token**,准确率持平或更高。
- 灵魂是**占位变量 + 依赖引用**:`#E2 = Calc[#E1 * r]`,使"规划"完全不必等"观测";无依赖变量**可并行取证**。
- 论文:**Xu et al., _ReWOO: Decoupling Reasoning from Observations…_(2023)**,并展示了把规划能力**蒸馏进 7B 小模型**。
- 软肋:**盲规划、不收中途观测 → 蓝图错了难自纠**;只适合**步骤可静态规划**的任务。
- 谱系定位:[[10 Plan-and-Execute|Plan-and-Execute]] 的省 token 极致版,与 [[12 LLMCompiler|LLMCompiler]](DAG 并行降延迟)同源,与 [[09 ReAct|ReAct]](每步交错观测)正相反。

## 主流开源实现 / Python 库

- **`billxbf/ReWOO`** —— **原论文官方实现**(Binfeng Xu),含 Planner/Worker/Solver 三模块、ALFWorld/HotpotQA 等评测脚本(`run_eval.py`)。要复现论文数值看它;作者后来把"更好的实现"并进了 Gentopia。
- **`langchain-ai/langgraph` ReWOO 教程** —— 官方 tutorial(`docs/docs/tutorials/rewoo/rewoo.ipynb`,JS 版亦有),用 LangGraph 状态图实现"多步 planner + 变量替换",是当下**最适合直接拿来改的工程模板**;旧的 `examples/rewoo/` 目录已归档不再更新。

首选:想跑通/改造选 LangGraph 教程版;要对齐论文 baseline 用 `billxbf/ReWOO`。ReWOO 没有独立 pip 包,一般作为 LangGraph 图模式手搓。

## 工业界实践

ReWOO 在工业界更多是作为一种**成本优化思路**被吸收,而非整套照搬——它的核心洞见("把冗长观测从推理循环里拿掉,只在末尾汇合一次")被用在那些**任务可静态规划、中间观测很大、且追求极致省 token** 的批量场景。

**主流落地形态与适配场景(具体名 + 定位)**:
- **LangGraph ReWOO 教程图**:`planner(出带 #En 变量的蓝图)→ worker(变量替换 + 静默取证)→ solver(末尾一次合成)` 的状态图,是工程里最常被拿来改造的模板。
- **批量数据富化 / ETL pipeline**:对一批实体执行"查多个固定字段 → 算指标 → 汇总"的流水,步骤可静态排布、观测(API 返回)冗长,ReWOO 把"每条都反复回灌观测"砍成"取证只走工具、末尾一次合成",省得最狠。
- **可蒸馏的工具增强小模型**:论文证明用 GPT-3.5 当 Planner 教师可把规划能力蒸馏进 7B 小模型——生产里"用小模型 + ReWOO 分解"来降本是真实路线,见 [[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]]、[[32 Agentic RL 与训练|Agentic RL 与训练]]。
- **RAG 的固定多跳取证**:当多跳问题的检索路径基本可预排时,ReWOO 式蓝图能并行取多个证据再一次合成,见 [[36 Agentic RAG|Agentic RAG]]。

**规模化与成本/延迟**:ReWOO 的成本卖点是**大模型固定只调 2 次**(Planner + Solver),中间观测从不重复回灌——论文报告比 ReAct 省约 **5× token**。延迟上,**无依赖的 `#En` 变量可并行取证**,把串行 round-trip 压成几批并发(这点与 [[12 LLMCompiler|LLMCompiler]] 同源)。但要注意:Solver 末尾要把**全部证据**一次塞进上下文,若证据总量巨大,Solver 那一次调用反而可能很贵——所以 ReWOO 适合"中间观测多、但筛后证据可控"的任务,生产里常在 worker 阶段先抽字段/截断再喂给 solver。

**可观测与运维**:ReWOO 的蓝图(含变量依赖)本身就是一张**可审计的计划图**,worker 的取证表(每个 #En 的真实值)是结构化日志,排障时直接看"哪个变量取证失败/被污染"。但它的弱点也在可观测层暴露:**中途没有观测回灌,模型看不到现实**,所以失败往往要到 Solver 阶段才显现——生产里常加一层"Solver 发现证据矛盾就触发重 plan"的兜底(代价是部分退回 [[10 Plan-and-Execute|Plan-and-Execute]])。

**踩坑与最佳实践**:
- **执行前做拓扑校验**:防 Planner 生成依赖环、或 `#E2` 引用尚未定义的 `#E3`。
- **变量替换要稳**:`#E1` 取证失败返回空时别静默往下污染后续,worker 应校验每个变量取证成功再继续。
- **worker 工具调用要稳健**:这层不经大模型纠错,参数格式错就直接失败,需要稳健解析 + 重试。
- **别把 ReWOO 当万能省钱键**:对高不确定任务硬上,准确率会掉;省 token 的前提是任务**本就适合静态规划**。
- **证据瘦身后再喂 Solver**:避免 Solver 那次调用因证据过长而吃掉省下来的 token。

```python
# ReWOO 三件套(全程大模型只调 2 次)
def rewoo(task, llm):
    blueprint = planner(task, llm)        # 大模型①:出带 #En 变量的蓝图
    evidence = {}
    for var, tool, arg in blueprint:      # worker:0 次大模型
        arg = substitute(arg, evidence)   # 把已得 #E 真实值代入(变量替换)
        if depends_ready(var, evidence):  # 无依赖者可并行取
            evidence[var] = TOOLS[tool](arg)   # 纯工具,失败要校验
    return solver(task, blueprint, evidence, llm)  # 大模型②:末尾一次合成
```

## 面试高频

**Q1:ReWOO 名字什么意思?核心思想一句话?**
标准答:**Reasoning WithOut Observation**——把工具观测从推理循环里拿掉,推理(规划)和观测彻底解耦,中间不把 observation 反复回灌给大模型,只在末尾 Solver 一次性汇合。
- 追问"为什么这样能省":ReAct 一个 N 步任务会把冗长观测重复携带 N 次,token 爆炸;ReWOO 大模型固定只调 2 次,中间观测从不回灌,论文省约 5× token。

**Q2:三件套各是什么?谁调大模型、谁不调?**
标准答:Planner(LLM,出带占位变量 #En 的蓝图)/ Worker(**非 LLM**,逐变量取真实证据 + 变量替换,无依赖变量可并行)/ Solver(LLM,拿蓝图+完整证据一次合成)。大模型只在 Planner 和 Solver 出现,Worker 全程零大模型。
- 追问"占位变量起什么作用":`#E2 = Calc[#E1 * r]` 让"规划"完全不必等"观测"——规划阶段 #E1 是符号,worker 阶段才被真实值原地替换,这是 ReWOO 的灵魂。
- 陷阱:有人以为 worker 也调模型——不,worker 纯执行工具,这正是省钱根源。

**Q3:ReWOO 和 ReAct 是什么关系?和 LLMCompiler 呢?**
标准答:与 [[09 ReAct|ReAct]] **正相反**——ReAct 每步交错回灌观测(最贵最灵活),ReWOO 观测只末尾汇合一次(最省但盲规划)。与 [[12 LLMCompiler|LLMCompiler]] **同源**——都用"带依赖的变量计划";区别是 LLMCompiler 进一步把蓝图编译成显式 DAG、主打降延迟,ReWOO 更朴素、主打省 token。

**Q4:ReWOO 的最大软肋是什么?什么时候不该用?**
标准答:**盲规划**——Planner 没看到真实结果就排全程,且中途不收观测,蓝图一旦错了难自纠。所以**下一步严重依赖上一步真实结果、充满意外分支的探索式任务别用**(该用 ReAct);ReWOO 只适合步骤可静态规划、且中间观测冗长的任务。

**Q5:论文还有什么值得提的贡献?**
标准答:除了 Planner-Worker-Solver 分解和约 5× 省 token,论文还展示了**蒸馏**——用 GPT-3.5 当 Planner 教师把规划能力蒸馏进 7B LLaMA,证明这套分解利于把"工具增强推理"下放到小模型。

## 知识拓展

**谱系定位**:这是"减少大模型介入次数"主线的**省 token 极致点**:[[09 ReAct|ReAct]](每步问)→ [[10 Plan-and-Execute|Plan-and-Execute]](规划集中一次)→ **ReWOO(连观测都不回灌、固定调 2 次)** → [[12 LLMCompiler|LLMCompiler]](同样的变量计划,但编译成 DAG 主打降延迟)。ReWOO 和 LLMCompiler 是这条线上的**双胞胎**——共享"带依赖的变量蓝图"基因,一个优化 token、一个优化延迟。

**进阶/延伸**:
- **混合路线**:生产里常给 ReWOO 加"Solver 发现证据矛盾就触发重 plan"的兜底,这其实是把 ReWOO 往 [[10 Plan-and-Execute|Plan-and-Execute]] 拉,用一点点重规划弹性换"盲规划失准"的鲁棒性。
- **把变量蓝图显式成图**:再往前一步就是 [[12 LLMCompiler|LLMCompiler]] 的 DAG + Task Fetching Unit 调度。
- **规划能力的获取**:ReWOO 靠 prompt/蒸馏拿到规划能力;更前沿的是用 [[31 Agent 提示词优化(DSPy)|DSPy]] 自动优化 Planner 提示,或用 [[32 Agentic RL 与训练|Agentic RL]] 直接训练规划策略。

**相关论文/前沿**:
- **Xu et al., _ReWOO: Decoupling Reasoning from Observations…_(2023)**——本范式原点,提出三模块分解 + 蒸馏。
- **Kim et al., _An LLM Compiler for Parallel Function Calling_(ICML 2024)**——把变量蓝图编译成 DAG 并行,见 [[12 LLMCompiler|LLMCompiler]]。
- **provider 原生 parallel tool calls(2024 起)**:OpenAI/Anthropic 让模型一轮发多个工具调用,某种程度上把 ReWOO/LLMCompiler 的"无依赖并行取证"内建进了 API,削弱了手搓 ReWOO 的部分动机。

**边界与反模式**:
- **反模式 1:对高不确定任务硬上 ReWOO**——盲规划失准且不收观测自纠,准确率掉,该用 [[09 ReAct|ReAct]]。
- **反模式 2:Solver 一次塞入巨量未筛证据**——可能把省下的 token 又吐回去,先在 worker 阶段抽字段/截断。
- **反模式 3:跳过拓扑校验**——依赖环或引用未定义变量会直接死在执行阶段。
- **边界**:ReWOO 假设"规划阶段不需要看真实结果就能想清全程",这个假设在探索式任务上不成立,是它能力的硬边界。
