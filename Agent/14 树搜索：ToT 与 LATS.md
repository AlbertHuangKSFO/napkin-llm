**树搜索** 让 agent 不再只走一条推理链,而是**展开一棵搜索树**、并行探索多条路径、择优而行——失败的分支可以**剪枝/回溯**,而不是一条道走到黑。代表是 **ToT(Tree of Thoughts)** 和 **LATS(Language Agent Tree Search)**。

这是对单链范式的根本升级:[[09 ReAct|ReAct]] 是一条「想-做-看」的链,[[10 Plan-and-Execute|Plan-and-Execute]]、[[11 ReWOO|ReWOO]] 把规划解耦但执行仍近似线性,[[12 LLMCompiler|LLMCompiler]] 把无依赖任务并行成 DAG——但它们**都不回溯**。树搜索把「探索-评估-回溯」这套经典搜索算法搬进 LLM 推理:用**更多算力(token)换更高的正确率**。LATS 更是把 [[09 ReAct|ReAct]] 的行动、[[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的反思、MCTS 的搜索三者统一进一棵树。

## 本质:为什么要搜索树

单链推理(chain)的死穴是**一锤定音**:每一步只生成一个延续,选错了就错到底,无法回头。但很多难题(数学、规划、24 点、代码)需要**试探多种走法、比较优劣、放弃死路**。

树搜索把这件事形式化:
- **节点** = 一个中间状态(ToT 里叫「thought 思维」;LATS 里是「ReAct 风格的状态:思考+行动+观测」)。
- **边** = 从一个状态到下一个状态的扩展(生成一个候选延续 / 执行一个动作)。
- **搜索** = 系统地扩展、用评估器给节点打分、剪掉没希望的、回溯到更优的祖先再展开。

代价是**算力暴涨**:一条链 1 次推理,一棵树要对每个节点反复调 LLM 生成与评估。所以树搜索是「贵但强」,适合**高难、可验证、值得花钱**的任务。

## ToT:Tree of Thoughts

### 机制(分步)
ToT 把「思维」当作搜索单元,四个要素:

1. **思维分解(thought decomposition)**:把问题拆成可一步步推进的「思维」(比如解 24 点的每一步算术、写作的每一段提纲)。
2. **思维生成(generation)**:在当前节点,让 LLM **采样多个候选思维**作为子节点(sample 多次,或一次列举若干)。
3. **状态评估(evaluation)**:用 LLM 给每个候选状态打分——「这条路能不能通到解?」可以是数值打分,或多个状态间投票比较。这是剪枝的依据。
4. **搜索算法**:用 **BFS** 或 **DFS** 遍历这棵树;BFS 每层保留 top-k 最有希望的节点(beam),DFS 一路深入、撞死路就**回溯**。低分节点直接**剪枝**。

![[树搜索：ToT 与 LATS-思维树.svg]]

ToT 是**纯推理**的搜索(原版不强调调外部工具),核心贡献是证明了「让 LLM 显式地探索+评估+回溯」能在需要前瞻/试错的任务上大幅超过 Chain-of-Thought。

### 原论文
**Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan,_Tree of Thoughts: Deliberate Problem Solving with Large Language Models_**(Princeton & Google DeepMind,NeurIPS 2023,arXiv 2023-05)。经典战绩:在「24 点游戏」上把 GPT-4 的成功率从 CoT 的 4% 提到 74%;在创意写作、迷你填字游戏上也显著提升。

## LATS:Language Agent Tree Search

### 机制:MCTS 四步 + 行动 + 反思
LATS 把 **蒙特卡洛树搜索(MCTS)** 引入 LLM agent,并融合 [[09 ReAct|ReAct]] 的行动和 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的反思。每一轮迭代走 MCTS 的四步:

1. **选择(Selection)**:从根用 **UCT(UCB applied to Trees)** 公式往下走,在「利用高价值节点」与「探索访问少的节点」之间平衡,选出一个待扩展节点。
2. **扩展(Expansion)**:在该节点采样若干个 **ReAct 风格的动作**(思考+工具调用),生成多个子节点——和 ToT 的「生成多候选」类似,但这里的候选是**带工具动作**的。
3. **评估 / 模拟(Evaluation)**:**真的执行**这些动作拿到环境观测(observation),再用 LLM(或环境奖励)给得到的状态打**价值分**。这一步让 LATS 能与真实环境交互,而非纯推理。
4. **反向传播(Backpropagation)**:把评估到的价值沿路径**回传**,更新所有祖先节点的统计(访问次数、累计价值),供下一轮 Selection 用。

额外地,当某条路径**失败**时,LATS 借用 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 生成一段**言语反思**,注入到后续的扩展中(纠正方向)——把「搜索 + 行动 + 反思」三件事统一进同一棵树。

![[树搜索：ToT 与 LATS-MCTS.svg]]

### 原论文
**Zhou, Geng, Zhao, Wu, Bai, Kim, ... ,_Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models_**(UIUC,ICML 2024,arXiv 2023-10)。一句话定位:把 ToT 的纯推理树升级为**能调工具、能反思、用 MCTS 搜索**的 agent 树;在 HumanEval、WebShop、HotpotQA 等推理+行动混合任务上达到当时 SOTA。

## 可跑最小代码(伪代码)

### ToT:生成-评估-剪枝(BFS/beam)
```python
def tot_bfs(root, generate, evaluate, beam=3, depth=4, k=5):
    frontier = [root]                                  # 当前层的状态
    for _ in range(depth):
        candidates = []
        for state in frontier:
            for _ in range(k):                          # 每个状态采样 k 个候选思维
                candidates.append(state + generate(state))
        scored = [(evaluate(c), c) for c in candidates] # 状态评估器打分
        scored.sort(reverse=True)
        frontier = [c for _, c in scored[:beam]]        # 剪枝:只留 top-beam(其余丢弃)
        if any(is_solution(c) for c in frontier):       # 找到解就停
            return next(c for c in frontier if is_solution(c))
    return frontier[0]
```
DFS 版把 `frontier` 换成栈、撞到低分/死路就 `pop` 回溯到上一个分叉点。

### LATS:MCTS 四步
```python
def lats(root, value_fn, n_iter=30):
    for _ in range(n_iter):
        node = root
        while node.children:                            # 1) 选择:UCT 往下走
            node = max(node.children, key=uct)          #    平衡利用与探索
        children = node.expand_with_react_actions()     # 2) 扩展:采样多个 ReAct 动作
        for child in children:
            obs = child.execute()                       # 3) 评估:真执行动作拿观测
            child.value = value_fn(child.state, obs)     #    LLM/环境打价值分
            if child.failed:                             #    失败 → Reflexion 反思
                child.reflection = reflect(child.trajectory)
        leaf = max(children, key=lambda c: c.value)
        backpropagate(leaf)                             # 4) 反向传播:价值回传更新祖先
        if leaf.is_terminal_success():
            return leaf.trajectory                      # 找到通过路径,返回
    return best_trajectory(root)

def uct(n):  # 选择公式:利用项 + 探索项
    import math
    return n.value/n.visits + C*math.sqrt(math.log(n.parent.visits)/n.visits)
```

## 对比:链式 vs 树式

| 维度 | 链式([[09 ReAct|ReAct]] / [[10 Plan-and-Execute|Plan-and-Execute]]) | ToT | LATS |
|---|---|---|---|
| 结构 | 单链 | 搜索树 | 搜索树(MCTS) |
| 回溯 | 无 | 有(剪枝/回溯) | 有(反向传播) |
| 评估 | 隐式 / 无 | 显式状态评估器 | 价值函数 + 真实观测 |
| 工具 | 有(ReAct) | 原版偏纯推理 | 有(ReAct 式动作) |
| 反思 | 无(除非加 Reflexion) | 无 | **内建**(复用 Reflexion) |
| token 成本 | 低~中 | 高 | 最高 |
| 适用 | 通用 | 难推理、需前瞻试错 | 高难、可验证、要交互 |

## 何时用 / 坑

✅ **该上树搜索**:任务**难、需要多路探索与回溯**、且**有可靠的状态评估信号**、并且你**愿意为正确率付高 token 成本**。ToT 适合纯推理难题(数学、规划、约束满足);LATS 适合「推理 + 工具行动 + 可验证奖励」的混合任务(代码、网页操作)。

❌ **别上**:
- 任务简单或线性——单链 [[09 ReAct|ReAct]] 就够,树搜索纯属浪费算力。
- **评估器不可靠**——整套搜索的方向盘是 evaluate/value_fn,它不准,搜索就是在错误地形上乱爬,越搜越偏。
- 对**延迟/成本敏感**的生产场景——树搜索动辄几十倍 LLM 调用,贵且慢。

坑:
- **评估器是上限**(同 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 的 Evaluator):打分不准 → 剪错枝、留错路。
- **成本爆炸**:分支数 × 深度 × 每节点多次 LLM 调用,token 指数级膨胀;必须卡 beam 宽度、深度、迭代上限、预算熔断。
- **状态可比性**:评估要能在不同分支间公平比较,否则 beam/UCT 的选择失真。
- **LATS 需可重置/可模拟的环境**:它要「真执行动作拿观测」,若环境有副作用(改了生产数据)或不可回退,MCTS 的多次试探假设就破了。

## ReAct 谱系横向对比(全家福)

把这一簇范式按关键维度排成一行行——这是理解「自主推理架构」演进的主线图:

![[树搜索：ToT 与 LATS-谱系对比.svg]]

| 范式 | 观测时机 | 并行度 | token 成本 | 搜索 / 回溯 | 适用场景 |
|---|---|---|---|---|---|
| [[09 ReAct|ReAct]] | 每步即观测,边走边想 | 串行,无并行 | 中~高(反复读历史) | 单链,无回溯 | 通用探索型任务,步骤不可预定 |
| [[10 Plan-and-Execute|Plan-and-Execute]] | 先规划后执行,步内观测 | 步内可串行 | 中(省去每步重规划) | 线性,可整体重规划 | 多步任务、需全局计划、减少 LLM 调用 |
| [[11 ReWOO|ReWOO]] | **规划期不观测**,蓝图执行后才代入 | 有限并行 | **低(最省)** | 线性,弱回溯 | 步骤可预定、强调省 token |
| [[12 LLMCompiler|LLMCompiler]] | DAG 编排,就绪即执行 | **原生并行**(无依赖任务同发) | 低 + 快 | DAG,Joiner 可重规划 | 多工具且可并行、要低延迟 |
| ToT | 扩展即评估(纯推理为主) | 分支可并行评估 | 高 | **树搜索 + 剪枝/回溯** | 难推理、需前瞻试错(24点/规划) |
| LATS | 行动 + 观测都进树(带工具) | 分支可并行 | **最高** | **MCTS + 回溯 + 反思** | 高难、可验证、愿付高成本的推理+行动 |

读法:从上到下,**探索越深、回溯越强、token 越贵**。工程取舍永远是「正确率 vs 成本/延迟」——多数任务用上半区(链式/规划)就够,只有真·难题才值得下半区的树搜索。

## 关键事实速记

- 树搜索 = 不止一条链,而是**展开搜索树**:扩展 → 评估 → 剪枝/回溯 → 择优;用 token 换正确率。
- **ToT**(Yao et al., NeurIPS 2023):思维为节点,BFS/DFS + 状态评估 + 剪枝回溯;24点从 4%→74%。偏纯推理。
- **LATS**(Zhou et al., ICML 2024):**MCTS 四步**(选择 UCT / 扩展 ReAct 动作 / 评估真执行 / 反向传播)+ [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 反思,统一搜索-行动-反思。
- 和单链([[09 ReAct|ReAct]]/[[10 Plan-and-Execute|Plan-and-Execute]]/[[11 ReWOO|ReWOO]]/[[12 LLMCompiler|LLMCompiler]])的本质区别:**有回溯**。
- 上限是**评估器/价值函数**的质量;成本会指数膨胀,必须卡 beam/深度/迭代/预算。
- 别为难度不匹配的任务硬上:简单任务上树搜索 = 烧钱。

## 主流开源实现 / Python 库

**ToT**
- **`princeton-nlp/tree-of-thought-llm`** —— **原论文官方实现**(Yao et al., NeurIPS 2023),含 24 点 / 创意写作 / 填字三套任务,pip 包名是 **`tree-of-thoughts-llm`**(注意带 `-llm`)。要复现论文用它,**首选**。
- ⚠️ **`kyegomez/tree-of-thoughts`**(pip `tree-of-thoughts`,无 `-llm`)—— 第三方"即插即用"版,star 高但**社区广泛反映代码跑不通、疑似自动生成**,且占用了官方想要的 PyPI 名。别被名字骗到,优先用上面 princeton 官方版。

**LATS**
- **`lapisrocks/LanguageAgentTreeSearch`** —— **LATS 原论文官方实现**(Zhou et al., ICML 2024),含 HumanEval / WebShop / HotpotQA 实验(任务简报里未列,这是核对补上的官方源)。
- **`langchain-ai/langgraph` LATS 教程** —— 官方 tutorial 把 MCTS(UCT 选择 / 扩展 ReAct 动作 / 真执行评估 / 反向传播)+ Reflexion 反思搭成 LangGraph 图,**工程落地首选**,易接现有工具与检索。

首选:复现论文用各自官方仓库;要在工程里跑 LATS 用 LangGraph 教程版。ToT/LATS 都没有"标准生产 pip 库",主流是官方代码或 LangGraph 图模板。

## 工业界实践

### 真相:纯 ToT/LATS 极少直接上生产,但其「搜索思想」无处不在

树搜索的**纯学术形态**(对每个节点反复调 LLM 评估+回溯)在生产里**罕见直接部署**——成本/延迟太高。但它的核心思想被大量工业系统以「轻量化」方式吸收:

- **Best-of-N / 多采样投票**:最常见的「退化版树搜索」——并行采样 N 条完整答案,用打分器(reward model 或 LLM-judge)选最优。它是「深度 1、宽度 N」的树,没有回溯但保留了「多路探索+评估择优」。工业界(代码生成、数学解题)用得极广,因为可**完全并行**、延迟可控。
- **推理时缩放(inference-time / test-time compute scaling)**:OpenAI o1/o3、DeepSeek-R1(2024–2025)把「探索-评估-回溯」**内化进单条长 CoT**——模型在一条链里自己分叉、自我否定、回退。这是树搜索思想的最大规模工业化,但搜索发生在模型「脑内」而非外层 harness。
- **代码 Agent 的搜索**:部分 SWE Agent(见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]])用「生成多个候选 patch → 跑测试评分 → 选过测试的」,本质是以**测试套件为价值函数**的浅层树搜索/Best-of-N。
- **过程奖励模型(PRM)做引导**:数学推理(如 OpenAI 的 PRM、Math-Shepherd)用 PRM 给每一步打分,配 beam search / 树搜索挑高分路径——这是 ToT「状态评估器」的工业化、用训练好的奖励模型替代 LLM 自评,更稳更便宜。
- **框架支持**:**LangGraph** 提供 ToT / LATS 官方教程图(可接现有工具与检索);**RAP / Tree-of-Thoughts** 开源库做研究复现。生产里更常见的是自己写 Best-of-N + 一个打分器,而非搬完整 MCTS。

### 成本工程(树搜索的命门)

| 杠杆 | 作用 | 典型配置 |
|---|---|---|
| beam 宽度 / 分支数 `k` | 控制每层候选数 | 难题 3–5,简单任务 1–2 |
| 最大深度 | 防无限深入 | 按任务步数设硬上限 |
| 迭代上限 `n_iter`(LATS) | 控 MCTS 轮数 | 几十轮即熔断 |
| 预算熔断 | 防 token/费用爆炸 | 设 token/调用次数/美元上限,超了返回当前最优 |
| 评估器选型 | 决定方向盘 | 优先**可验证信号**(单测/编译);LLM-judge 慎用且要稳定 |
| 早停 | 找到合格解即停 | `is_solution` 命中立即返回 |

经验:树搜索 token 成本 ≈ 分支数 × 深度 × 每节点 LLM 调用次数,**指数级膨胀**。必须把上面所有护栏同时卡死,否则一道题烧掉几十上百次调用。

### 踩坑

- **评估器不可靠 = 整套崩**:搜索的方向盘是 evaluate/value_fn,它不准则剪错枝、留错路,越搜越偏。上生产前必须先验证评估信号质量。
- **LATS 需可重置/可模拟环境**:它要「真执行动作拿观测」,若环境有副作用(改了生产数据)或不可回退,MCTS 的多次试探假设破裂。生产里常退化为「只在沙箱/只读环境跑 LATS」。
- **状态可比性失真**:不同分支的状态评估若不在同一尺度,beam/UCT 选择会失真。
- **延迟不可接受**:动辄几十倍 LLM 往返,面向用户同步场景几乎不能用,多放离线/异步。

## 面试高频

**Q1:树搜索相比 ReAct 这类单链范式,本质区别是什么?**
标准答:**有回溯**。单链(ReAct、Plan-and-Execute、ReWOO、LLMCompiler)选错一步就错到底、无法回头;树搜索展开搜索树,系统地「扩展→评估→剪枝/回溯→择优」,用更多 token 换更高正确率。代价是算力指数膨胀。

**Q2:ToT 和 LATS 的区别?**
标准答:**ToT**(Yao et al., NeurIPS 2023)是**纯推理**树——节点是「思维」,用 BFS/DFS + LLM 状态评估 + 剪枝回溯,原版不强调调外部工具;经典战绩 24 点从 CoT 的 4% 到 74%。**LATS**(Zhou et al., ICML 2024)把 **MCTS** 引入,节点是「ReAct 风格的思考+工具动作+观测」,**真执行动作拿环境反馈**,并融合 [[13 Reflection 与 Reflexion|Reflexion]] 反思,统一「搜索+行动+反思」。一句话:ToT 在脑内想,LATS 能动手且会反思。

**Q3:讲一下 LATS 的 MCTS 四步。**
标准答:① **选择(Selection)**:从根用 **UCT** 公式往下,平衡「利用高价值节点」与「探索访问少的节点」;② **扩展(Expansion)**:在选中节点采样多个 ReAct 风格动作生成子节点;③ **评估/模拟(Evaluation)**:真执行动作拿观测,用 LLM 或环境奖励给状态打价值分;④ **反向传播(Backpropagation)**:把价值沿路径回传,更新所有祖先的访问次数与累计价值,供下轮选择用。失败路径额外生成言语反思注入后续扩展。
- **追问:UCT 公式?** $\text{UCT}(n)=\underbrace{\frac{Q(n)}{N(n)}}_{\text{利用}}+\underbrace{C\sqrt{\frac{\ln N(\text{parent})}{N(n)}}}_{\text{探索}}$。第一项是节点平均价值(利用),第二项随访问次数下降、鼓励探索冷门节点;$C$ 调探索强度。

**Q4(陷阱):树搜索一定比单链强吗?** 不是。① 任务简单/线性时,单链就够,树搜索纯属烧钱;② **评估器不可靠时越搜越偏**;③ 对延迟/成本敏感的生产场景往往用不起。树搜索是「贵但强」,只对**高难、可验证、值得花钱**的任务划算。

**Q5:为什么生产里很少直接部署完整 ToT/LATS?用什么替代?** 因为成本/延迟指数膨胀且需要可靠评估器与可重置环境。工业界更常用**轻量化变体**:Best-of-N 多采样投票(可并行、延迟可控)、PRM 引导的 beam search、或干脆用 o1/R1 这类把搜索**内化进单条长 CoT**的推理模型。

**Q6:树搜索的天花板是什么?** **评估器/价值函数的质量**——和 [[13 Reflection 与 Reflexion|Reflexion]] 的 Evaluator 同理。打分不准则剪错枝、留错路,搜索在错误地形上乱爬。

## 知识拓展

- **谱系定位**:树搜索是 ReAct 谱系的「下半区」——[[09 ReAct|ReAct]](单链)→ [[10 Plan-and-Execute|Plan-and-Execute]]/[[11 ReWOO|ReWOO]](规划解耦)→ [[12 LLMCompiler|LLMCompiler]](DAG 并行)→ ToT/LATS(带回溯的搜索树)。LATS 把 [[13 Reflection 与 Reflexion|Reflexion]] 反思组件直接吸收进来。评估器质量这一上限与 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 相关,成本控制接 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
- **相关工作与前沿(带年份)**:
  - **Tree of Thoughts**(Yao et al., NeurIPS 2023):奠基,证明显式探索+评估+回溯能大幅超 CoT。
  - **Graph of Thoughts**(Besta et al., AAAI 2024):把「树」推广为「图」——思维节点间可合并/聚合,比纯树更灵活。
  - **LATS**(Zhou et al., ICML 2024):MCTS + ReAct + Reflexion 统一推理/行动/规划。
  - **RAP: Reasoning via Planning**(Hao et al., 2023):把 LLM 当世界模型,用 MCTS 做规划。
  - **Process Reward Models / Let's Verify Step by Step**(Lightman et al., OpenAI, 2023):逐步打分的 PRM,是工业化「状态评估器」的关键。
  - **AlphaGeometry / AlphaProof**(DeepMind, 2024):神经搜索在形式化数学上的巅峰,思想同源(搜索+验证器)。
  - **推理时缩放**:OpenAI o1(2024)、DeepSeek-R1(2025)把树搜索内化进长 CoT,代表「外部搜索」向「模型内搜索」迁移的趋势;但**外部可验证环境 + 工具行动**仍是模型内化不了的部分,LATS 思路在 Agentic 任务上不过时。
- **边界 / 反模式**:① 为难度不匹配的简单/线性任务硬上树搜索 = 指数级烧钱;② 评估器不可靠时多搜 = 错误地形越爬越远;③ 在有副作用/不可回退的真实环境跑 LATS 的多次试探;④ 不卡 beam/深度/迭代/预算护栏导致成本失控。
