[[14 GraphRAG 知识图谱检索|GraphRAG 知识图谱检索]] 的本质是:用 LLM 把语料抽成一张**实体-关系知识图谱**,再在图上做检索与汇总,而不是只在扁平的 chunk 向量里做近邻召回。它解决纯向量 [[01 什么是 RAG|什么是 RAG]] 答不了的一类问题——**全局/汇总型问题**(「整个语料的主题是什么」「这批文档总体在讲什么」)。

向量 RAG 的硬伤是:它只会取 top-k 个**最相似的局部片段**,而「主题」「总体趋势」这种信号不在任何单个片段里,是分散在全语料的整体属性。GraphRAG 把语料先压成图 + 层次化社区摘要,于是能在「整片森林」尺度上回答,而非只看几棵树。它也是 [[19 Agent 记忆系统|Agent 记忆系统]] 里「结构化记忆 / 知识图谱记忆」的检索面。

## 本质:为什么向量召回答不了全局问题

把问题分两类:

- **局部问题(local)**:「X 在文档里说了什么」「A 和 B 的关系是什么」。答案就在某几个片段里,**向量近邻召回擅长**,这是经典 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 的主场。
- **全局问题(global)**:「整个语料的主要主题有哪些」「这 1000 份事故报告反映出哪些共性根因」。答案**不存在于任何单个片段**,是全语料的汇总属性。

微软 GraphRAG 论文把后者明确界定为 **query-focused summarization(QFS,面向查询的摘要)** 任务,而非检索任务——你不是要「找到那一段」,而是要「综合全部内容生成一个面向问题的摘要」。向量 RAG 取 top-k 时,k 个片段既不可能覆盖全语料,语义相似度也无法定位「主题」这种全局信号,于是答案**以偏概全或直接漏主题**。

GraphRAG 的洞见:**先把语料离线压成结构**(图 + 各层级社区摘要),把「综观全局」的重活提前做掉;在线只需对预生成的社区摘要做 map-reduce,就能在全语料尺度上作答。

![[GraphRAG vs 向量RAG.png]]

## 机制:微软 GraphRAG 的四步流水线

微软 GraphRAG(Edge et al. 2024,见来源)分**离线建图**与**在线查询**两阶段:

### 离线索引(一次性,LLM 密集)
1. **切 chunk + LLM 抽取实体与关系**。逐 chunk 让 LLM 抽出实体(人、组织、概念)、实体间关系、以及关系描述,汇成一张**实体知识图谱**(节点=实体,边=关系)。同一实体在多处出现会被归并,描述被聚合。
2. **Leiden 社区检测**。在图上跑 **Leiden 算法**(一种层次化社区发现,比 Louvain 更稳),把紧密相连的实体聚成**社区**(community),且是**层次化的**:顶层是大主题簇,逐层往下是更细的子簇。
3. **LLM 为每个社区生成摘要**。对每个层级的每个社区,让 LLM 把该社区内的实体、关系、原文要点**总结成一段社区摘要**——这步把「全语料综观」预先算好、存库。

### 在线查询(map-reduce)
4. **全局问题 → map-reduce 社区摘要**。面对全局问题:
   - **MAP**:把(某层级的)每个社区摘要分别喂给 LLM,各自生成一个「部分答案」,并自评一个 **helpfulness 分数**(这条社区摘要对回答该问题有多大帮助)。
   - **REDUCE**:按分数筛掉无关的部分答案,把剩下的汇总成**最终全局答案**(带引用)。

局部问题则走 **local search**:从问题里的实体出发,取其图邻域(邻居实体、相关关系、关联文本单元)拼上下文作答——这一支更接近经典 RAG,但带图结构增强。

![[GraphRAG-流程.png]]

代价很直白:**离线建图要对每个 chunk 调一次 LLM 抽取、对每个社区调一次 LLM 摘要**,语料大时 token 成本与建索引时间都高;在线 map-reduce 也要遍历社区摘要。所以 GraphRAG 适合「值得为全局洞察付一次重索引成本」的高价值语料,不适合海量、低价值、只问局部事实的场景。

## 可跑最小代码

下面用伪代码勾出四步骨架(生产直接用 `microsoft/graphrag` 库,无需自己实现):

```python
# ① 抽实体关系建图
graph = Graph()
for chunk in split(corpus):
    # LLM 按 prompt 抽 (实体, 关系, 关系描述) 三元组
    triples = llm_extract_entities_relations(chunk)
    for (e1, rel, e2, desc) in triples:
        graph.add_edge(e1, e2, relation=rel, desc=desc, source=chunk.id)

# ② Leiden 层次化社区检测
communities = leiden_hierarchical(graph)   # 返回多层级:level0 粗, level1, ... 细

# ③ 为每个社区预生成摘要(离线存库)
summaries = {}
for level in communities.levels:
    for comm in level:
        # 把社区内实体/关系/原文喂给 LLM 生成一段摘要
        summaries[comm.id] = llm_summarize_community(comm.entities, comm.edges)

# ④ 在线:全局问题走 map-reduce
def global_search(question, level):
    partials = []
    for comm in communities.at(level):                 # MAP
        ans, score = llm_answer_from_summary(question, summaries[comm.id])
        if score > 0:                                   # 自评 helpfulness
            partials.append((ans, score))
    partials.sort(key=lambda x: -x[1])                  # REDUCE
    return llm_reduce(question, [a for a, _ in partials])   # 汇总成终答

# 局部问题改走 local_search:从问题实体取图邻域拼上下文(类似增强版 RAG)
```

要点:① 建图与摘要是**离线一次性**的重活,把全局综观提前算好;② 在线 global 用 **map-reduce 社区摘要**,local 用 **图邻域**;③ helpfulness 自评用来在 reduce 阶段筛掉无关社区。

## 变体:LightRAG 与图+向量混合

纯微软 GraphRAG 建图贵、增量更新难。**LightRAG**(Guo et al. 2024,见来源)是更轻的图增强 RAG:把图结构融进文本索引,用**双层检索(dual-level retrieval)**——低层抓具体实体及其关系(细节问题),高层抓主题/概念(宽泛问题);并**把图结构与向量表示结合**,既能顺关系召回相关实体,又保留向量的语义召回,响应更快且支持增量更新。

更一般地,生产里常见**图 + 向量混合检索**(可类比 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 的思路,但这里融合的是「图结构信号」而非「关键词信号」):向量负责语义近邻召回,图负责多跳关系遍历与全局汇总,两路结果再合并。图遍历本身也是一种**多跳能力**,与 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] 互补——后者靠 LLM 迭代检索做多跳,GraphRAG 靠图上的边显式做多跳。检索到的实体最终仍要落到 [[04 Embedding 与向量数据库|Embedding 与向量数据库]] 或图数据库里存取。

## 对比表

| 维度 | 向量 RAG | GraphRAG(微软) | LightRAG |
|---|---|---|---|
| 索引结构 | 扁平 chunk 向量 | 实体图 + 层次社区摘要 | 图结构 + 向量,双层 |
| 擅长问题 | 局部事实 | **全局/汇总** + 局部 | 全局与局部兼顾,偏轻量 |
| 多跳关系 | 弱 | 强(图遍历) | 强 |
| 建索引成本 | 低(只 embed) | **高**(逐 chunk + 逐社区调 LLM) | 中 |
| 增量更新 | 易 | 难(重跑社区检测) | 较易 |
| 在线开销 | 低(一次近邻) | 高(map-reduce 社区) | 中 |

一句话:**向量答「这段说了什么」,GraphRAG 答「整个语料是什么」**;LightRAG 是在两者间找轻量折中。

## 何时用 / 坑

**该上 GraphRAG**:语料有丰富实体关系(报告集、案例库、知识密集文档)、用户问的是全局/汇总/多跳关系型问题、且语料价值高到值得付一次重建索引成本。

**不该上**:只问局部事实(向量 RAG 又快又便宜)、语料海量且廉价(建图成本扛不住)、或文档间几乎无实体关系(图退化、收益低)。

**坑**:
- **建图成本爆炸**:逐 chunk 抽实体 + 逐社区摘要,大语料 token 账单与耗时都吓人。先在子集上验证全局问答确实需要再上。
- **抽取质量是上限**:LLM 抽错实体/关系,图就歪,后续全错。需校验抽取 schema、做实体归并。
- **增量更新难**:新增文档可能改变社区结构,严格做要重跑 Leiden;工程上常做近似增量。
- **全局搜索慢且贵**:map-reduce 要遍历该层级所有社区摘要,问题简单时是杀鸡用牛刀;先分流——局部问题走 local/向量,只有全局问题才走 global。
- **过度图化**:不是所有 RAG 都需要图。文档间无关系时,图带来的复杂度远超收益。

## 关键事实

- GraphRAG = **LLM 抽实体关系建图 → Leiden 社区检测 → 各社区预生成摘要 → 全局问题 map-reduce 社区摘要**。
- 解决向量 RAG 的盲区:**全局/汇总型问题**(query-focused summarization),向量只能局部 chunk 召回,答不了「整个语料的主题」。
- 微软 GraphRAG:Edge et al. 2024,arXiv:2404.16130,《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》;开源库 `microsoft/graphrag`。
- **local search** 走图邻域(类增强 RAG),**global search** 走 map-reduce 社区摘要——两支分别对应局部与全局问题。
- 代价:**离线建图 LLM 密集、贵且增量更新难**;适合高价值语料 + 全局问题,不适合海量廉价 + 纯局部事实。
- **LightRAG**(Guo et al. 2024,arXiv:2410.05779)是轻量变体:图 + 向量结合、双层检索、响应更快、支持增量。
- 生产常见**图 + 向量混合**:向量管语义近邻,图管多跳关系与全局汇总;与 [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] 互补(显式图边 vs LLM 迭代)。

## 工业界实践

GraphRAG 从 2024 的论文,到 2025 已经是一条**有完整工具链、有成本分层方案**的生产路线。工业界关心的不是「能不能建图」,而是「**怎么让建图别破产、怎么把全局/局部查询分流、怎么增量更新**」。

### 主流工具与定位

- **`microsoft/graphrag`**(微软官方库):论文的参考实现,提供 indexing pipeline(抽实体关系→Leiden→社区摘要)+ 三种查询模式。**定位:全量、重索引、质量上限高,但贵**。
- **DRIFT Search**(微软 2024 末新增):**Dynamic Reasoning and Inference with Flexible Traversal**。把社区信息注入 local search 起点,**融合 global 的广度与 local 的细粒度**——先用社区摘要扩大检索起点的覆盖面,再沿图做 local 遍历。定位:在 global 太贵、local 太窄之间取折中,显著扩大了答案用到的事实多样性。
- **LazyGraphRAG**(微软 2024 末发布):**不预先做全图摘要**,把重 LLM 分析**推迟到查询时**,在线把向量检索与图检索即时结合。**定位:几乎为零的前置索引成本**,在与向量 RAG 相当的查询成本下,local 查询质量超过 long-context 向量 RAG、GraphRAG DRIFT 与 GraphRAG local search。它直接回应了原版「建图破产」的最大痛点。2025-08 起 GraphRAG/LazyGraphRAG 已并入 Azure 的 **Microsoft Discovery**(科研 agentic 平台)。
- **LlamaIndex `KnowledgeGraphIndex` / `PropertyGraphIndex`**、**Neo4j + LLM(`neo4j-graphrag` / LLM Graph Builder)**、**LangChain `LLMGraphTransformer`**:把 LLM 抽三元组 + 图数据库(Neo4j / NebulaGraph / Memgraph)落地,生产里常配 **Neo4j 的向量索引**做「图 + 向量」一库混检。
- **LightRAG**(`HKUDS/LightRAG`):图 + 向量双层检索的轻量开源实现,响应快、支持增量,见上文「变体」。

### 典型生产架构:查询分流 + 图向量混库

```
                    ┌──────────────┐  global  ┌─────────────────────┐
question → router → │ 复杂度/类型分类│─────────→│ map-reduce 社区摘要  │→ 全局综观答案
                    └──────┬───────┘          └─────────────────────┘
                           │ local/具体实体
                           ↓
                    ┌──────────────────────────────┐
                    │ DRIFT / local: 实体→图邻域     │→ 局部精确答案
                    │  + 向量近邻(图向量混库)       │
                    └──────────────────────────────┘
```

关键是**别让所有 query 都走 global**(map-reduce 遍历全社区,杀鸡用牛刀):用 router 把「整个语料的主题」类问题分到 global,把「X 和 Y 的关系」类分到 local/DRIFT。这与 [[05 Routing|Routing]]、[[13 Modular RAG|Modular RAG]] 的 conditional flow 是同一思路。

### 规模化:召回 / 延迟 / 成本

- **成本三段分层**:全量 GraphRAG(质量高、最贵)→ DRIFT(折中)→ LazyGraphRAG(按需、最省)。**先在子集验证「确实需要全局问答」再决定层级**,否则一上来全量建图会被 token 账单劝退。
- **建图是离线一次性重活**:逐 chunk 调 LLM 抽取 + 逐社区调 LLM 摘要,大语料动辄数百万 token。生产做法:并发批处理、用便宜模型做抽取、对低价值 chunk 跳过建图。
- **增量更新**是 GraphRAG 最大工程痛点:新文档可能改变社区结构,严格做要重跑 Leiden。生产常用**近似增量**(只重算受影响子图)或直接选 LightRAG/LazyGraphRAG 这类原生支持增量的方案。
- **评估**:微软 2025 发布 **BenchmarkQED**,把 RAG/GraphRAG 的全局问答评测自动化(自动出题 + LLM 评分),解决「全局摘要质量没法用传统检索指标(召回@k)衡量」的难题——因为 global 不是「找到那一段」,而是 query-focused summarization。

### 踩坑与最佳实践

- **抽取质量是上限**:LLM 抽错实体/关系,图就歪,后续全错。固定抽取 schema、做**实体归一/消歧**(同一实体的多种写法归并)、抽完抽样人审。
- **社区粒度要调**:Leiden 层级太粗→摘要笼统;太细→社区太多、map-reduce 太贵。按问题尺度选层级。
- **图向量混库降本**:用 Neo4j/PG 的向量索引把「向量近邻」和「图遍历」放一库,省去跨系统同步;local 查询先向量召种子实体,再沿图扩邻域。

## 面试高频

**Q1:GraphRAG 解决向量 RAG 的什么盲区?为什么向量答不了?**
A:解决**全局/汇总型问题**(query-focused summarization),如「整个语料的主要主题是什么」。向量 RAG 只取 top-k 个**最相似的局部片段**,但「主题」「总体趋势」**不在任何单个片段里**,是分散在全语料的整体属性;k 个片段既覆盖不了全语料、相似度也定位不到「主题」这种全局信号,于是以偏概全或漏主题。GraphRAG 把「综观全局」预先压成图 + 层次社区摘要,在线 map-reduce 即可全语料作答。

**Q2:微软 GraphRAG 的完整流水线?**
A:离线四步——① 切 chunk + LLM 抽实体关系建图;② **Leiden** 层次化社区检测;③ LLM 给每个社区生成摘要;④ 在线 global 查询走 **map-reduce**(MAP:每个社区摘要各出部分答案 + 自评 helpfulness;REDUCE:筛掉无关、汇总成带引用的终答)。局部问题走 **local search**(从问题实体取图邻域拼上下文)。追问「为什么用 Leiden 不用 Louvain」→ Leiden 修复了 Louvain 可能产生不连通社区的缺陷,更稳、收敛更好。

**Q3:GraphRAG 的代价是什么?什么时候不该用?**
A:代价=**离线建图 LLM 密集(逐 chunk + 逐社区调 LLM,贵)+ 增量更新难(改社区结构要重跑 Leiden)+ global 在线遍历社区慢**。不该用:只问局部事实(向量又快又便宜)、语料海量且廉价(建图扛不住)、文档间几乎无实体关系(图退化、收益低)。

**Q4(进阶):听说过 LazyGraphRAG / DRIFT 吗?解决什么?**
A:都是微软对原版「太贵」的回应。**LazyGraphRAG** 不预建全图摘要、把重 LLM 分析推迟到查询时、在线即时融合向量与图检索,**前置索引成本几乎为零**,local 质量还超过 long-context 向量 RAG。**DRIFT search** 把社区信息注入 local 起点,融合 global 广度与 local 细粒度。这题能答出来直接区分「读过论文」和「跟进了 2025 进展」。

**Q5(陷阱):GraphRAG 和多跳检索什么关系?**
A:都做多跳,但机制不同。**GraphRAG 靠图上的边显式多跳**(图遍历);[[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]] 靠 **LLM 迭代检索**隐式多跳。二者互补,生产可组合(图给结构化多跳骨架,LLM 迭代补图里没显式连的跳)。陷阱:别说「GraphRAG 就是为多跳」——它的杀手锏是**全局摘要**,多跳只是副产物。

## 知识拓展

- **评估的特殊性**:GraphRAG 的 global 答案是**摘要**,不是「检索到的片段」,所以传统检索指标(召回@k、MRR、nDCG)失效。要用 **comprehensiveness / diversity / empowerment** 这类 LLM-as-judge 维度,或微软 2025 的 **BenchmarkQED** 自动化评测。这点和 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] 的「生成质量评估」相通。
- **安全/权限交叉**:GraphRAG 把语料压成实体图,**权限治理更棘手**——一个社区摘要可能聚合了不同密级文档的信息,造成「摘要级越权」(单看每篇都没权限问题,聚合后泄露)。生产要把 [[16 检索安全与访问控制|检索安全与访问控制]] 的 ACL 下沉到**建图与社区摘要阶段**,而非只在最终检索过滤。知识图谱本身也是投毒目标——污染实体/关系能系统性误导图遍历,参见 [[AI 安全/11 向量与嵌入弱点与 RAG 投毒]]。
- **反模式**:① 「无脑全图」——文档间无实体关系还硬建图,复杂度远超收益;② 「只建图不分流」——所有 query 都走 global,成本爆炸;③ 「建一次再不更新」——语料变了图不变,答案越来越陈旧而无人察觉。
- **前沿与延伸**:GraphRAG 是 [[19 Agent 记忆系统|Agent 记忆系统]] 里「结构化/知识图谱记忆」的检索面;2025 趋势是**成本分层(Lazy/DRIFT)+ 图向量混库 + 自动化全局评估**三件套。延伸方向:把社区摘要做成**可增量维护的层次记忆**,让 agent 长期运行时图随交互演进——与 Modular RAG 的 loop flow 和 Agentic RAG 的长期记忆汇流。

## 来源

- Darren Edge, Ha Trinh, Newman Cheng, et al. 《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》. arXiv:2404.16130 (2024). 微软研究院,GraphRAG 原始论文,提出 LLM 建图 + Leiden 社区检测 + 社区摘要 map-reduce。
- Microsoft Research. *Introducing DRIFT Search*(2024)与 *LazyGraphRAG sets a new standard for quality and cost*(2024)博客;*BenchmarkQED: Automated benchmarking of RAG systems*(2025)。GraphRAG 的成本分层与全局评估进展。
- Zirui Guo, Lianghao Xia, Yanhua Yu, et al. 《LightRAG: Simple and Fast Retrieval-Augmented Generation》. arXiv:2410.05779 (2024). 图结构 + 向量双层检索的轻量图增强 RAG。
- 开源实现:`microsoft/graphrag`(微软官方库)、`HKUDS/LightRAG`。
