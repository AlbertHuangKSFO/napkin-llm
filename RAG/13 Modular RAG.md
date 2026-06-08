[[13 Modular RAG|Modular RAG]] 是把 RAG 系统**拆成可插拔的乐高积木**的设计范式:不再把它当成「检索→生成」一条写死的直线,而是拆成一组**独立模块(modules)+ 可替换算子(operators)**,再由一个编排层(orchestration)按需把它们拼成不同的流程拓扑。它是 [[01 什么是 RAG|什么是 RAG]] 从 Naive 到 Advanced 再到 Modular 的演进终点,也是工程上把各种 RAG 技巧(查询变换、混合检索、重排序、自评纠错……)统一到一个框架里的「机箱」。

提出这个框架的是 Gao et al. 2024 的综述《Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks》(arXiv:2407.21059)。它的洞见是:RAG 这两年冒出的方法太多、太杂,很多都塞不进经典的「retrieve-then-generate」直线里,需要一个**更高抽象的、可重构的**框架来收编它们。

![[Modular RAG-乐高.png]]

## 机制

### 从 Naive 到 Advanced 到 Modular 的三段演进

理解 Modular RAG，先看它在「演进谱系」里的位置(这套 Naive / Advanced / Modular 三分法源自同一作者团队更早的 RAG 综述):

- **Naive RAG**:最朴素的一条直线——`索引 → 检索一次(top-k)→ 拼进 prompt → 生成`。死穴一堆(召回差也将就、不重写 query、搞不定多跳……),详见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]。对应 flow 就是纯 **linear(线性)**。
- **Advanced RAG**:在直线两端加料补救——**检索前(pre-retrieval)**做查询变换/路由([[07 查询变换 Query Transformation|查询变换 Query Transformation]]),**检索后(post-retrieval)**做重排序、压缩、过滤([[10 重排序 Reranking|重排序 Reranking]])。流程仍是一条加长的直线,只是首尾更讲究。
- **Modular RAG**:**打破直线**。把整个系统抽象成**模块化的组件 + 可任意编排的 flow**,允许条件分支、并行分支、循环回路。Advanced RAG 的 pre/post 处理,在这里只是 Modular 框架的一个特例(linear flow)。

一句话:**Naive 是直线、Advanced 是加料的直线、Modular 是可重构的图**。

### 六层模块(modules)

Modular RAG 把系统纵向切成若干**模块**,每个模块下面挂可替换的**算子(operators)**——同一模块换不同算子,就像乐高同一接口换不同积木:

| 模块 | 干什么 | 可换的算子举例 | 关联笔记 |
|---|---|---|---|
| **Indexing 索引** | 把语料切块、编码、建库 | 分块策略 / 层级索引 / 稀疏+稠密 | [[03 分块策略 Chunking|分块策略 Chunking]]、[[04 Embedding 与向量数据库|Embedding 与向量数据库]] |
| **Pre-retrieval 检索前** | 加工 query | 改写 / 分解 / 扩展 / 路由 | [[07 查询变换 Query Transformation|查询变换 Query Transformation]]、[[05 Routing|Routing]] |
| **Retrieval 检索** | 召回候选 | 向量 / 混合 / 多源融合 | [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] |
| **Post-retrieval 检索后** | 加工召回结果 | 重排序 / 压缩 / 过滤 / 精炼 | [[10 重排序 Reranking|重排序 Reranking]] |
| **Generation 生成** | 据证据生成答案 | 受约束生成 / 引用归因 / 忠实校验 | [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] |
| **Orchestration 编排** | 调度上面五层、决定走哪条 flow | 路由 / 调度 / 条件 / 循环控制 | —— |

模块化的好处是**解耦与可复用**:想升级召回?只换 Retrieval 模块的算子,别处不动。想加忠实度校验?往 Generation 模块插一个算子。这正是「LEGO-like reconfigurable」的字面意思。

### 四种 RAG Flow 模式

Modular RAG 最有价值的部分,是把 **Orchestration 能编排出的流程拓扑**归纳成四类模式。这是它超越「直线 RAG」的关键:

1. **Linear(线性)**:模块顺序串联,一条过。`pre → retrieve → post → generate`。**Naive RAG 和 Advanced RAG 都是 linear**,只是模块多少之差。
2. **Conditional(条件)**:中间有个判别节点,**按条件选不同后续路径**。典型就是 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 里的 **Adaptive-RAG**——复杂度分类器把 query 路由到「不检索 / 单步 / 多步」三条路;[[05 Routing|Routing]] 也属此类。
3. **Branching(分支)**:**多条路并行跑、再汇聚**。比如把 query 分解成多个子查询、各自检索、结果聚合;或同一 query 走多个检索源并行召回再融合(对应 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 的并行召回 + RRF 融合思路)。
4. **Loop(循环)**:**带反馈回边,可反复迭代**——检索 → 生成/自评 → 不满意就回去重写 query 再检索,直到收敛。这是四种里最强的,**Loop pattern 就是 [[36 Agentic RAG|Agentic RAG]] 的雏形**:CRAG 的「召回差→纠错再检索」、Self-RAG 的「ISSUP 不通过→重生成」、多跳检索的 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] 迭代,全是 loop。再往前一步——把 retrieve 做成 agent 能自主反复调用的工具——就跨进 Agentic RAG。

可以这样串记:**Naive=linear,Adaptive/路由=conditional,多查询/多源=branching,自评纠错/多跳=loop(→ Agentic RAG)**。

**RAG 范式 / flow 选型卡**(按查询形态查):

| 查询形态 | 选哪种 flow / 范式 | 为什么 |
|---|---|---|
| 简单事实问答、查询模式单一 | **Linear**(Naive/Advanced) | 一条直线最省心,过度模块化只增延迟和复杂度;先跑通基线 |
| 简单/复杂混流、要按难度分路 | **Conditional**([[05 Routing|Routing]]/Adaptive-RAG) | 复杂度分类器把简单 query 分流走快路径,只对难题开多轮,省成本 |
| 一个问题含多个可拆子问 / 多源 | **Branching**(子查询并行 + RRF 融合) | 无依赖分支并行铺开一次汇总,比串行快;注意并行执行别串行 |
| 召回质量参差、需自评纠错 / 多跳 | **Loop**(CRAG/Self-RAG/IRCoT) | 带回边反复改写再查直到收敛,是 [[36 Agentic RAG|Agentic RAG]] 雏形;必设 `max_iter` |
| 全局/汇总型问题(整库主题) | **GraphRAG** | 向量召回只取局部 top-k,答不了全语料汇总,见 [[14 GraphRAG 知识图谱检索|14]] |
| 流程要 agent 运行时自主决定 | **Agentic RAG** | 控制权从「预编排图」交给 LLM,更灵活但更难调、成本飘,见 [[36 Agentic RAG|36]] |

> 工程主流(2025):**默认 Modular(人编排、可审计、延迟可控),关键不确定处才放权给 agent**。

**branching vs loop 别混(一个具体例子)**。同一个 query「比较 A 和 B 公司的营收与员工数」:**branching** 把它拆成 2 个子查询(「A 的营收员工」「B 的营收员工」)**同时并行**各检索一次,4 路结果**一次性汇聚**——分支之间无依赖、无回头,DAG 没有回边。**loop** 则是同一个 query **改写 3 次串行迭代**:第 1 轮查得不够 → 自评「证据不足」→ 改写 query → 第 2 轮再查 → 仍不够 → 再改 → 第 3 轮……带**回边**反复跑同一条路,直到收敛或撞 `max_iter`。一句话:**branching 是"并行铺开、一次汇总"(无回边),loop 是"串行试错、反馈重来"(有回边)**。

## 可跑最小代码

下面用极简的模块注册 + flow 编排器,演示「同一组模块如何拼成四种不同 flow」:

```python
# ---- 模块=可替换的算子,统一接口:吃 state、吐 state ----
def m_pre(s, llm):     s["query"] = llm(f"改写为更利于检索的查询:{s['raw']}"); return s
def m_retrieve(s, db): s["docs"]  = db.search(s["query"], k=5); return s
def m_post(s, rrk):    s["docs"]  = rrk.rerank(s["query"], s["docs"])[:3]; return s
def m_generate(s, llm):s["ans"]   = llm(f"据证据答:{s['docs']}\nQ:{s['raw']}"); return s

# ---- Flow 1:Linear(Naive/Advanced)----
def flow_linear(s, llm, db, rrk):
    return m_generate(m_post(m_retrieve(m_pre(s, llm), db), rrk), llm)

# ---- Flow 2:Conditional(Adaptive 式按复杂度选路)----
def flow_conditional(s, llm, db, rrk, classifier):
    level = classifier(s["raw"])
    if level == "simple":
        s["ans"] = llm(f"直接回答:{s['raw']}"); return s          # 跳过检索
    if level == "single":
        return flow_linear(s, llm, db, rrk)
    return flow_loop(s, llm, db, rrk)                              # 复杂 → 转 loop

# ---- Flow 3:Branching(子查询并行 → 汇聚)----
def flow_branching(s, llm, db, rrk):
    subs = llm(f"把问题拆成2-3个子查询,每行一个:{s['raw']}").splitlines()
    pooled = []
    for sub in filter(str.strip, subs):                           # 各分支并行检索
        pooled += db.search(sub.strip(), k=3)
    s["docs"] = rrk.rerank(s["raw"], pooled)[:5]                   # 汇聚后重排
    return m_generate(s, llm)

# ---- Flow 4:Loop(自评纠错,Agentic RAG 雏形)----
def flow_loop(s, llm, db, rrk, max_iter=3):
    for _ in range(max_iter):
        s = m_post(m_retrieve(m_pre(s, llm), db), rrk)
        if "yes" in llm(f"证据够回答「{s['raw']}」吗?yes/no\n{s['docs']}").lower():
            break
        s["raw"] = llm(f"上轮证据不足,换个角度改写查询:{s['raw']}")  # 回边:重写再查
    return m_generate(s, llm)
```

要点:① 四个 `m_*` 模块**接口一致**(state→state),可任意重排/替换,这就是「乐高积木」;② `flow_linear` 是 Naive/Advanced,`flow_conditional` 是 Adaptive,`flow_branching` 是多子查询并行,`flow_loop` 带回边自评——**同一组模块,编排出四种拓扑**;③ `flow_loop` 的回边(改写再查)就是 [[36 Agentic RAG|Agentic RAG]] 的灵魂,记得 `max_iter` 护栏;④ 真实框架(LangGraph)把这套 state + 节点 + 条件边显式建成图,见 [[20 RAG 开源生态全景|RAG 开源生态全景]]。

## 何时用 / 坑

**该上 Modular RAG 思维**:系统要长期演进、需要灵活替换组件(今天换个 reranker、明天加个网搜源)、或要支持多种查询模式(简单/复杂混流)。把 RAG 当乐高拆,比写死一条管线更可维护、可实验(A/B 不同算子)。

**不必上**:一次性 demo、查询模式单一且稳定的场景——直接 linear 一条龙更省心,过度模块化是负担。

**坑**:
- **过度工程化**:不是所有 RAG 都需要 conditional/branching/loop。简单事实问答 linear 足矣,硬上模块化和 flow 编排只增加复杂度和延迟。先 Naive 跑通,有明确瓶颈再模块化。
- **flow 越复杂越难调试**:branching/loop 的失败点散在多个分支/多轮里,定位 bug 比直线难得多。需要 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 级别的 trace/日志。
- **loop 不收敛**:和所有带回边的系统一样,自评一直说「不够」会反复空转。`max_iter` 是必备护栏(同 [[36 Agentic RAG|Agentic RAG]] 的最大跳数)。
- **模块边界看似清晰、实则耦合**:换 Indexing 的分块策略,往往得连带调 Retrieval 和 Post-retrieval(块变小→召回数要变→重排策略要改)。模块解耦是理想,实践中相邻模块常牵一发动全身。
- **「模块化」≠「自动变好」**:Modular RAG 是组织框架,不是性能银弹。每个模块里塞的算子(用哪个 embedding、哪个 reranker)才决定效果——框架只负责让你好换好试。

## 关键事实

- Modular RAG = 把 RAG 拆成**可插拔模块(modules)+ 可替换算子(operators)+ 可编排 flow**,从「写死的直线」升级为「可重构的图」(LEGO-like)。
- 演进谱系:**Naive(纯 linear,见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]])→ Advanced(加 pre/post 的 linear)→ Modular(模块化 + 任意 flow)**;Advanced 是 Modular 的 linear 特例。
- 六层模块:**Indexing / Pre-retrieval / Retrieval / Post-retrieval / Generation / Orchestration**;Orchestration 负责调度其余五层、决定走哪条 flow。
- 四种 flow:**Linear(=Naive/Advanced)、Conditional(=[[05 Routing|Routing]]/Adaptive 路由)、Branching(多子查询/多源并行汇聚)、Loop(自评纠错/多跳迭代)**。
- **Loop pattern 是 [[36 Agentic RAG|Agentic RAG]] 的雏形**:CRAG/Self-RAG([[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]])、IRCoT([[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]])都是 loop 的实例;把 retrieve 变成 agent 自主反复调用的工具,即跨入 Agentic RAG。
- 价值是**工程组织 + 可实验性**(好换好试),不是性能银弹;实际效果取决于各模块里塞的具体算子。
- 落地框架(LangGraph 把模块+条件边+循环建成图,LlamaIndex 提供模块化检索组件)见 [[20 RAG 开源生态全景|RAG 开源生态全景]]。

## 工业界实践

Modular RAG 不是论文里的玄学,它就是当下生产 RAG 的**事实架构**。落地时它从「概念框架」变成「一张显式的图 + 一套可观测的算子」。

### 落地框架:LangGraph 是事实标准

- **LangGraph**(LangChain 出品)是 2025 把 Modular RAG 四种 flow 真正工程化的主力框架。它把课本里的「state + 模块 + 条件边 + 循环」原样建成图:
  - `StateGraph`——一个**带类型的共享状态**,每个节点读它、写它(正是上文 `m_*(s)→s` 的接口)。
  - `add_node()`——注册模块/算子;`add_conditional_edges()`——实现 **conditional flow**(按状态路由到不同节点);带回边的节点 + `add_edge` 回指自身实现 **loop flow**。
  - `checkpointer`(MemorySaver / 持久后端)——存检查点,**可恢复、可断点续跑、可 human-in-the-loop**(在某节点暂停等人审再继续)。这是生产 RAG 区别于 demo 的关键:长流程要能挂起、审计、重放。
- **LlamaIndex** 提供模块化检索组件(`QueryPipeline`、各种 retriever / postprocessor / response synthesizer),把 Indexing/Retrieval/Post-retrieval 拆成可换件;**Haystack 2.x** 的 Pipeline 是显式 DAG,组件即模块,天然 branching;**DSPy** 把模块当「可编译/可优化的程序」,自动调 prompt 与算子选择。定位差异:LangGraph 偏「有状态的图编排 + agent 循环」,LlamaIndex 偏「检索/索引组件库」,Haystack 偏「生产 DAG 管线」,DSPy 偏「可优化的声明式程序」。

### 典型生产架构:Corrective / Adaptive 的图实现

把 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 落成 LangGraph 图,是 2025 最常见的「会自纠错的 RAG」模板:

```
        ┌─────────────┐
query → │ route(分类) │──simple──→ generate(直接答)
        └─────┬───────┘
         single/multi
              ↓
      ┌───────────────┐    docs    ┌──────────────┐
      │   retrieve    │──────────→ │ grade_docs   │ 给每篇打相关分
      └───────┬───────┘            └──────┬───────┘
              ↑ rewrite                   │ 相关不足
              └──── transform_query ←──────┤ (conditional edge)
                                          │ 相关够
                                          ↓
                                     ┌──────────┐
                                     │ generate │→ hallucination/answer 检查
                                     └────┬─────┘ 不通过→回 retrieve/rewrite
                                          ↓ 通过
                                        answer
```

- `grade_documents` 节点 = CRAG 的相关性评估;`transform_query` 回边 = loop flow 的重写再查;`generate` 后接 hallucination grader / answer grader = Self-RAG 的 ISSUP/ISUSE 校验。**全部用 conditional edge 串起来**,这就是 Modular RAG 四种 flow 在一张图里的合体。

### 规模化、可观测、踩坑

- **召回/延迟/成本**:flow 越复杂(branching 多源、loop 多轮),**尾延迟和 token 成本越爆**。生产做法:① conditional 路由先把简单 query 分流走 linear,**只对复杂 query 才进 loop/branching**(成本与质量的关键杠杆);② branching 分支**并行**执行(`asyncio` / LangGraph 并行节点),别串行;③ loop 设硬 `max_iter` + 成本预算护栏(reflect 一直说「不够」会空转烧钱)。
- **可观测**:Modular RAG 的 bug 散在多分支多轮里,**没有 trace 就没法调**。生产标配 **LangSmith / Langfuse / Arize Phoenix / OpenTelemetry GenAI** 全链路 trace——逐节点看输入输出、每跳 token、路由决策、循环次数。对应库内 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
- **评估**:模块化的好处是**可逐模块 A/B**——换 reranker、换 embedding、换路由阈值,各自有指标(检索召回@k、路由准确率、端到端答案质量),用 RAGAS / 自建评测集做离线回归。
- **最佳实践**:先用 linear 跑通基线、定位真实瓶颈,再有针对性地模块化;**幂等节点 + 可重放检查点**让长流程能恢复;**版本化 flow 拓扑**(把图当代码,改了能 diff、能回滚)。

## 面试高频

**Q1:Naive / Advanced / Modular RAG 怎么区分?**
A:Naive=纯 linear(索引→一次检索→拼 prompt→生成);Advanced=在直线两端加料(pre-retrieval 查询变换/路由 + post-retrieval 重排/压缩),**仍是直线**;Modular=打破直线,抽象成「模块 + 可任意编排的 flow(可条件、可并行、可循环)」,Advanced 是它的 linear 特例。追问「Modular 凭什么更强」→ 答:能表达 conditional/branching/loop 这些**直线表达不了的拓扑**,从而收编自纠错、多跳、自适应路由等方法。

**Q2:Modular RAG 的四种 flow 各对应什么真实技术?**
A:Linear=Naive/Advanced;Conditional=[[05 Routing|Routing]] / Adaptive-RAG(复杂度分类器选路);Branching=多子查询分解并行 + 结果融合(RRF)/ 多源并行召回;Loop=带回边的自评纠错(CRAG / Self-RAG)、多跳迭代(IRCoT / FLARE)。陷阱:把 branching 和 loop 混为一谈——branching 是**并行多路再汇聚**(无回边),loop 是**串行带反馈回边**(同一路反复跑)。

**Q3:Loop flow 和 Agentic RAG 什么关系?**
A:**Loop pattern 是 Agentic RAG 的雏形**。把 loop 的「检索→自评→不满意就重写再检索」固化成预定义图,就是 corrective/adaptive RAG;再把 retrieve 升级成 **agent 能自主决定何时调、调几次、调哪个工具**,控制权从「写死的图」交给 LLM,就跨进 [[36 Agentic RAG|Agentic RAG]]。区别在「谁决定流程」:Modular RAG 的 flow 是工程师预先编排的,Agentic RAG 的流程是 agent 运行时自主决策的。

**Q4:既然 Modular 这么好,为什么不无脑上?**
A:① 过度工程化——简单事实问答 linear 足矣,硬上 flow 编排只增延迟和复杂度;② 调试难——失败点散在多分支多轮,没 trace 调不动;③ loop 不收敛风险;④「模块化≠自动变好」,框架只让你好换好试,效果取决于塞进去的具体算子。**标准答:先 Naive 跑通基线、有明确瓶颈再模块化。**

**Q5(追问/陷阱):模块解耦是真的吗?**
A:理想是换一个模块别处不动,**实践常牵一发动全身**——改 Indexing 的分块粒度,往往要连带调 Retrieval 的 top-k 和 Post-retrieval 的重排策略(块变小→召回数要变→压缩比要改)。所以「模块边界清晰」是设计目标而非现实保证,相邻模块要一起评估。

## 知识拓展

- **与 Agentic RAG 的边界**:Modular RAG 是**控制流由人编排**的图;Agentic RAG 是**控制流由 LLM 决策**的图。前者可预测、可审计、延迟可控;后者更灵活但更难调、成本更飘。2025 的工程主流是「**默认 Modular,关键不确定处才放权给 agent**」的混合体——见 [[36 Agentic RAG|Agentic RAG]] 与 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]]。
- **反模式**:① 「flow 套娃」——条件里再嵌分支再嵌循环,深到没人看得懂,出问题定位以小时计;② 「无护栏 loop」——不设 `max_iter`/预算,生产会被一个边角 query 烧穿成本;③ 「模块化但不可观测」——拆得再漂亮,没有逐节点 trace 等于黑盒。
- **前沿**:把 flow 拓扑本身做成**可学习/可优化**对象——**DSPy**(Khattab et al. 2023)把模块化 RAG 当声明式程序、自动编译 prompt 与算子;**自动 agent 调优**(2025–2026 兴起)用数据驱动反馈循环系统性优化路由阈值、决策门限与工具编排,而非人手调。趋势是「编排从手写规则 → 数据驱动自动优化」。
- **延伸阅读**:Modular RAG 的四种 flow 把库内各篇串成一张网——Indexing→[[03 分块策略 Chunking|分块策略 Chunking]]/[[04 Embedding 与向量数据库|Embedding 与向量数据库]],Pre→[[07 查询变换 Query Transformation|查询变换 Query Transformation]]/[[05 Routing|Routing]],Retrieval→[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]],Post→[[10 重排序 Reranking|重排序 Reranking]],Generation→[[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]],Orchestration→[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]]/[[36 Agentic RAG|Agentic RAG]]。落地框架全景见 [[20 RAG 开源生态全景|RAG 开源生态全景]]。

## 来源

- Gao, Xiong, Wang, Wang. *Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks*. 2024. arXiv:2407.21059. <https://arxiv.org/abs/2407.21059>
- Khattab et al. *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*. 2023. arXiv:2310.03714(模块化 RAG 当可优化程序的代表)。
- LangGraph 官方文档:StateGraph / conditional edges / checkpointer 与 Corrective/Adaptive RAG 模板(2025)。
- (同作者团队)Gao et al. *Retrieval-Augmented Generation for Large Language Models: A Survey*. 2023. arXiv:2312.10997(Naive / Advanced / Modular 三分法的出处)。
