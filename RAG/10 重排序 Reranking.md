[[10 重排序 Reranking|重排序 Reranking]] 的本质是:召回之后,用一个**更准但更慢**的 cross-encoder 模型把候选文档重新精排,再把少量上下文喂给生成。它和召回是分工:召回(bi-encoder/向量/BM25)负责**从百万级里捞出几十上百个候选**(要快、要召回率高),重排负责**在这几十个里挑出真正最相关的几个**(要准、可以慢)。这就是 **两阶段 retrieve-then-rerank** 架构。

它接在 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 之后、[[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] 之前,是把检索质量"收口"的关键一环:召回阶段为了不漏召得宽,难免带噪音;重排用 query 和 doc 的**深度交互**把噪音压下去,直接决定喂进生成的上下文有多干净。

## 本质:为什么要两阶段

[[04 Embedding 与向量数据库|Embedding 与向量数据库]] 的向量检索用 **bi-encoder(双塔)**:query 和 doc **各自独立**编码成向量,事后算余弦相似度。好处是 doc 向量可**离线预算 + 建 ANN 索引**,在线只编码 query,毫秒级扫百万库。代价:query 和 doc 在编码时**互相看不到**,丢了 token 级交互,精度有上限。

**cross-encoder** 反过来:把 `[query | SEP | doc]` **拼成一条**喂进同一个 Transformer,query 的每个 token 能和 doc 的每个 token 做**交叉注意力**,直接输出一个相关性分。精度高得多,但**每个 query-doc 对都得现场过一遍模型**,无法预算、无法建索引——扫百万库不可能。

![[bi-vs-cross-encoder.png]]

结论是显而易见的分工:**别让 cross-encoder 扫全库,只让它精排 bi-encoder 海选出的小候选集**。

像招聘:**HR 海选**靠简历关键词从上千份里快速筛出一批候选(快、宁可多放进来也别漏掉好苗子,对应召回);**面试官精排**对候选逐个深聊、当面交互,挑出少量人(慢、准,对应 cross-encoder 重排)。没人会让面试官把上千人全部面一遍(cross-encoder 扫全库不可行),也没人敢只看简历就直接发 offer(跳过重排直接喂向量 top-k)。

![[重排序 Reranking.png]]

可从 bi-encoder/混合检索海选 **top-100** → cross-encoder 精排 → 取 **top-5** 喂生成开始试验；它是**起始搜索点**,不是通用常数。100 次 cross-encoder 前向仍要实测 p95 延迟、文本长度与硬件；百万次逐对前向通常不具备可行的在线延迟。

## 机制

### 两阶段漏斗
1. **召回(recall-oriented)**:[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 或纯向量,可先把 top-50、100、200 作为搜索空间。指标是**Recall@N**——宁可多召也别漏掉正确答案,因为漏了重排也救不回。
2. **精排(precision-oriented)**:cross-encoder 对每个候选打分,重排,取 top-k。指标是**精度/排序质量**(nDCG)——把真正相关的顶到最前。

为什么不一步到位:召回要快要全,通常用可索引的 bi-encoder、稀疏检索或混合检索；精排要准,常用 cross-encoder。两阶段让两类目标分别优化,而不是把某一模型当作唯一方案。

### 重排模型候选(按版本与约束选)
- **Cohere Rerank v4.0**:`rerank-v4.0-pro` 与 `rerank-v4.0-fast` 是两个独立候选；官方 `/v2/rerank` API 都以 `model`、`query`、`documents`、`top_n` 调用。v4.0 的 query 与每篇 document **共享** 32,768 token 总窗口(其中 4 token 保留)：query 最多占一半(16,384)；若 query 为 $q$ token，则 document 会按每段最多 `32,764-q` token 取用/切分，并同时受 `max_tokens_per_doc` 限制。该 API 参数默认是 **4,096**，长文档会据此自动截断；若要使用更长的实际输入，必须显式设置并记录截断策略。支持 YAML 形式的结构化文档与 100+ 语言。`pro`/`fast` 的选择应由本库标注集的 nDCG、p95、价格和并发实测决定。
- **Cohere Rerank v3.5**:`rerank-v3.5` 是另一版本化候选，同样支持 100+ 语言与结构化文档；其 query 与 document 共享 4,096 token 总窗口(3 token 保留)，query 最多占一半(2,048)；若 query 为 $q$ token，每段 document 最多取 `4,093-q` token，亦受 `max_tokens_per_doc` 限制。因此不能把 v3.5 与 v4.0 当作仅改模型名的无风险替换。
- **自托管 cross-encoder 候选**:如 BAAI bge-reranker、Jina reranker、Mixedbread mxbai-rerank、Qwen3-Reranker。先固定切块规则、最大输入长度和 batch，再在自己的 query–相关文档标注集上比较，而非用社区排名替代线上评测。
- **LLM-as-reranker / RankGPT**:让 LLM 直接输出候选的**排列(permutation)**，而非逐个独立打分。RankGPT 论文在 TREC 的特定集合、候选窗口、提示与模型设置下比较 listwise 排序；这个机制可在候选间显式比较，但 token 成本、提示长度、窗口策略和延迟也随之上升。不要把论文中的单一实验差值外推成固定收益或普适的“最准”。

**参数不是常数**:可从 `N∈{50,100,200}`、`k∈{3,5,10}`、`batch_size∈{8,16,32}` 开始网格或逐步搜索。先看 `Recall@N` 是否把答案送入候选集，再看 `nDCG@k`/MRR 是否改善排序；同时记录 p95、输入文本长度、batch、模型与 CPU/GPU/并发。若 N 增大只增加 p95 而不改善 Recall@N/nDCG@k，就应缩回 N 或改善召回。

### MMR 去冗余
重排只保证"每篇都相关",不保证"几篇之间不重复"。若 top-5 是 5 篇近重复文档,信息密度极低还挤占上下文。**MMR(Maximal Marginal Relevance)**在选下一篇时同时奖励"与 query 相关"、惩罚"与已选文档相似",`MMR = λ·rel(d,q) - (1-λ)·max sim(d, 已选)`,在相关性和多样性间权衡。常作为重排后的一道去冗余工序,尤其和 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 配合时(避免选进同一父文档的多个相邻句子)。

**MMR 手算(λ=0.7,3 候选选 top-2)**。设相关性 $\mathrm{rel}(A)=0.90,\ \mathrm{rel}(B)=0.85,\ \mathrm{rel}(C)=0.60$,候选间相似度 $\mathrm{sim}(A,B)=0.95$(A、B 近重复)、$\mathrm{sim}(A,C)=0.30$。第一轮无已选,只剩相关项 $\lambda\cdot\mathrm{rel}$:$A=0.63,\ B=0.595,\ C=0.42$,**A 当选**。第二轮对 A 算多样性惩罚:

$$\mathrm{MMR}(B)=0.7\times0.85-0.3\times0.95=0.595-0.285=0.31$$
$$\mathrm{MMR}(C)=0.7\times0.60-0.3\times0.30=0.42-0.09=0.33$$

B 虽然比 C 更相关(0.85 vs 0.60),但和已选的 A 几乎重复(sim 0.95)被狠扣,**最终选 C 而非 B**——这正是 MMR 压近重复、换来覆盖面的体现。

## 可跑最小代码

下面是一份只用 Python 标准库、可直接 `python3 rerank_demo.py` 跑的完整流程。`toy_score` 只为让示例自包含；生产中应把它替换为 `CrossEncoder(...).predict([(query, doc), ...], batch_size=...)` 的真实 cross-encoder 分数。这里的 `top_n=100`、`top_k=5`、`batch_size=16` 只是一组**起始参数**；生产应把它们连同候选文本长度、硬件和 p95 一起测。

```python
# rerank_demo.py —— ❌ 扫全库 vs ✅ 只精排召回候选，再接 MMR；无需第三方依赖。
from collections import Counter
from math import sqrt


def vectorize(text):
    return Counter(ch for ch in text.lower() if not ch.isspace())


def cosine(a, b):
    dot = sum(a[key] * b.get(key, 0) for key in a)
    norm_a = sqrt(sum(value * value for value in a.values()))
    norm_b = sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def toy_score(query, document):
    """可运行的占位相关性分；生产替换为 batch cross-encoder 打分。"""
    return cosine(vectorize(query), vectorize(document))


def rerank_everything(query, all_documents, batch_size=16):
    # ❌ 朴素：真实 cross-encoder 在这里会对全库每个 (query, doc) 前向一次。
    del batch_size  # toy_score 不批处理；保留参数以展示真实模型的批量位置。
    return sorted(all_documents, key=lambda doc: toy_score(query, doc), reverse=True)


def rerank_candidates(query, candidates, top_k=5, batch_size=16):
    # ✅ 高效：ANN/BM25/混合检索已给出 candidates；真实模型只精排这 N 个。
    del batch_size
    ranked = sorted(candidates, key=lambda doc: toy_score(query, doc), reverse=True)
    return ranked[:top_k]


def mmr(query_vec, candidate_vecs, candidates, k=5, lam=0.7):
    """向量须来自同一 embedding 空间；这里使用自包含的字符向量。"""
    selected, remaining = [], list(range(len(candidates)))
    while remaining and len(selected) < k:
        best = max(
            remaining,
            key=lambda i: lam * cosine(query_vec, candidate_vecs[i])
            - (1 - lam) * max(
                (cosine(candidate_vecs[i], candidate_vecs[j]) for j in selected),
                default=0.0,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


query = "如何减少 RAG 的重复上下文？"
all_documents = [
    "MMR 在相关性与多样性之间权衡，可减少近重复结果。",
    "重排序在召回候选集内重新排列，救不回未召回的文档。",
    "父文档回扩可在段落命中后补足上下文。",
    "HNSW 是常见的向量近邻索引。",
]
retrieved_top_n = all_documents[:3]  # 真实系统由 ANN/BM25/混合检索产生。

print("❌ 全库排序:", rerank_everything(query, all_documents, batch_size=16))
ranked = rerank_candidates(query, retrieved_top_n, top_k=3, batch_size=16)
print("✅ 候选精排:", ranked)
answer_context = mmr(vectorize(query), [vectorize(doc) for doc in ranked], ranked, k=2)
print("✅ 精排后 MMR:", answer_context)
assert len(answer_context) == 2 and set(answer_context).issubset(set(retrieved_top_n))
```

要点:① cross-encoder 输入是 **(query, doc) 对**,现场算,不能预存;② 只对召回候选批量打分，而非扫全库;③ `N`、`k`、batch 由 Recall@N、nDCG@k、p95、文本长度和硬件共同决定;④ MMR 在"已经相关"的候选里再去重复,提升上下文信息密度。

## 对比表

| 维度 | bi-encoder(召回) | cross-encoder(重排) | LLM/RankGPT(重排) |
|---|---|---|---|
| 编码方式 | query/doc 各自独立 | 拼接联合 | 拼接,输出排列 |
| query-doc 交互 | 无(事后算余弦) | token 级交叉注意力 | 全注意力 + 推理 |
| 排序质量 | 取决于模型与语料,通常用于候选召回 | 取决于模型与语料,常用于精排 | 强依赖所用 LLM、提示、窗口与评测集 |
| 速度 | 快(可预算/索引) | 慢(每对现算) | 通常受 token 数与调用方式限制 |
| 能扫全库? | 能 | 不能 | 不能 |
| 用在 | 召回海选 | 候选精排 | 候选少且已证明排序收益覆盖成本的场景 |

## 何时用 / 坑

**该评估重排**:当召回带噪、相关文档进入候选却排不到前列，或 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] / [[07 查询变换 Query Transformation|查询变换 Query Transformation]] 扩大了候选集时，可把重排作为把"召得多"变成"喂得准"的候选方案。先以同一标注集比较重排前后的 nDCG@k、Recall@N 与端到端 p95，再决定是否常驻。

**坑**:
- **跳过重排直接喂 top-k 向量结果**:向量召回的 top-5 排序常不够准,真正最相关的可能排在第 8、第 15。不重排,生成层就吃到次优上下文。
- **召回 top_n 设太小**:重排只能在候选集内重排,候选里就没正确文档,重排无力回天。把 50、100、200 作为起点，以 Recall@N 选 N，而不是背固定数字。
- **重排延迟**:cross-encoder 对每个候选都有实打实的前向成本；batch、文本长度、模型大小、CPU/GPU 与并发都会影响 p95。延迟敏感场景先测候选数与 batch，再考虑较小模型或 late interaction。
- **只相关不多样**:不接 MMR,top-5 可能是 5 篇近重复,信息量低。需要覆盖面时加 MMR。
- **长文档截断**:cross-encoder 有最大长度,长 doc 被截会丢关键段。配合 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]]——重排在小粒度(句/段)上做,再回扩到父文档喂生成。
- **域不匹配**:通用 reranker 在专业域(法律/医疗/代码)可能不准,必要时微调或换域内模型。

## 关键事实

- 重排 = 召回后用 **cross-encoder 精排**:query+doc 拼一起过 Transformer 打分；相对 bi-encoder 多了交互，也增加了逐对计算成本。
- **两阶段 retrieve-then-rerank**:bi-encoder/稀疏/混合召回先保 Recall@N，再由 cross-encoder 精排。`N=50–200`、`k=3–10` 适合作为试验起点，不能替代评测。
- bi-encoder **独立编码**可预算/建索引但丢交互;cross-encoder **拼接联合编码**保留 token 级交互但每对现算、不能扫全库。
- Cohere 可按版本选 `rerank-v4.0-pro`、`rerank-v4.0-fast` 或 `rerank-v3.5`：每个 query-document 对分别共享 32,768 / 4,096 token 总窗口，query 最多占一半，document 按 `32,764-q` / `4,093-q` 余量切分并受 `max_tokens_per_doc` 约束；两者均是多语言候选。
- **MMR** 在重排后去冗余:`λ·相关 - (1-λ)·与已选最大相似`,避免 top-k 全近重复。
- 位置:[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 召回 → 重排收口 → [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]];与 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 配合(小粒度重排 + 回扩父文档)。

## 工业界实践

重排是在不重建索引的前提下可快速实验的一层，但是否值得常驻必须由线上预算与离线标注集共同证明。

**候选选型卡**

| 场景 | 可试候选 | 先验收什么 |
|---|---|---|
| 托管 API、多语言或结构化记录 | Cohere Rerank v4.0 pro/fast；兼容性需要时也测 v3.5 | 版本上下文上限、nDCG@k、p95、价格与数据合规 |
| 可自托管 | bge-reranker、Jina、mxbai、Qwen3-Reranker 等 cross-encoder | 相同文本截断与 batch 下的 nDCG@k、GPU/CPU 吞吐、p95 |
| 候选很少，且可接受 prompt 成本 | listwise LLM/RankGPT 风格 | 候选窗口、提示稳定性、失败回退、端到端成本 |
| 想保留 token/page 级信号 | ColBERT/多向量 late interaction | Recall、MaxSim 重排质量、向量存储、p95 |
| 专业域不准 | 用域内 `(query, positive, negative)` 数据微调或更换候选 | 留出的域内集、失败案例、漂移监控 |

**起始架构，不是固定配置**
```
混合检索(BM25 + 向量,RRF 融合)→ N∈{50,100,200} 候选
   → cross-encoder rerank(一个版本化候选)→ k∈{3,5,10}
   → (可选) MMR 去冗余 → grounded 生成
```

### 别把 named vectors 当成 MaxSim

**named vector(命名向量)**说的是一个点上有多个以名字区分的表示，例如 `title`、`body`、`image`；每个名字通常对应一个单向量空间，查询时选择其中一个或融合多个。它本身**不产生 token 级交互**。**token/page multi-vector(多向量)**说的是同一个表示的值是 $m\times d$ 矩阵：一个 token、chunk 或页面 patch 一行。late interaction 的 **MaxSim** 是先对每个 query 向量取其与文档各行的最大相似度，再对 query 向量求和：

$$s(q,D)=\sum_{i=1}^{|q|}\max_{j\in[1,|D|]} q_i^\top d_j$$

两者是不同维度的概念：一个**命名字段也可以装 multi-vector**，但仅仅“有多个 named vectors”不等于实现了 ColBERT/MaxSim；MaxSim 也不是 cross-encoder，因为 document 向量仍可离线生成。

| 引擎 | named vectors 的具体接口 | token/page multi-vector 与版本门槛 |
|---|---|---|
| **Qdrant** | 建 collection 时 `vectors: {"dense": {...}, "colbert": {...}}`；查询用 `POST /collections/{collection}/points/query` 的 `using: "dense"` 或 `using: "colbert"`。 | **v1.10.0+** 的 Query API 支持 multi-vector：在 `VectorParams` 配 `multivector_config=MultiVectorConfig(comparator=MAX_SIM)`，`query_points(prefetch=Prefetch(query=dense_q, using="dense", limit=N), query=colbert_q, using="colbert")` 可先单向量召回再 MaxSim 重排。只做重排时可给 `colbert` 设置 `hnsw_config=HnswConfigDiff(m=0)`，避免为它另建 ANN。 |
| **Vespa** | 不以“named vector”作统一 API；用不同 schema field，例如 `title_embedding`、`body_embedding`，以 YQL `nearestNeighbor(field, query_tensor)` 取候选。 | 用 mixed tensor field，例如 `tensor<float>(token{}, x[128])`，配 rank profile 的 `sum(reduce(sum(query(qt) * attribute(colbert), x), max, token), qt)` 实现 MaxSim；用 `ranking=...` 与 YQL 的首阶段候选配合。该能力是 schema/rank-profile 张量表达式，而非名为 multivector 的开关；用 Vespa 8 的当前受支持版本并以 `vespa prepare` 验证配置。若使用原生 `colbert-embedder` 的**数组/多段输入**，门槛为 **Vespa 8.303.17+**。 |
| **Weaviate** | collection 的 `vectorConfig` 定义多个 named vector；查询指定 `target_vector`。多个名字是多个空间，不自动 MaxSim。 | 官方配置页标注 multi-vector **Added in v1.30**，且当前仅用于 **named vector 的 HNSW**。建集合用 `Configure.MultiVectors.self_provided(name="colbert")`，写入该字段的二维向量；查询 `collection.query.near_vector(near_vector=query_matrix, target_vector="colbert", limit=k)`。低于 1.30 不应假设该 API/语义存在。 |

**规模化(召回/延迟/成本/索引)**
- **延迟模型**:cross-encoder 延迟随候选数增长，还受单对 token 数、batch、模型、CPU/GPU、并发影响。控延迟的杠杆:① 以 Recall@N/nDCG@k 收缩 N；② 换小候选；③ batch 推理 + GPU；④ ONNX/TensorRT；⑤ 在模型上下文内按统一规则截断或改小粒度。
- **索引选型(召回阶段)**:召回要"宁多勿漏"给重排留空间。**HNSW** 高召回、`efSearch` 可在线调,延迟敏感场景主选;**IVF / IVF-PQ** 内存省但召回受量化损伤,大库省成本时用,需把 `nprobe` 调大保召回;**标量/二值量化 + 全精度重排(rescore)** 是省内存又保精度的常用组合。无论哪种,**召回的损失重排救不回**,所以索引参数偏召回侧。
- **成本**:闭源 API(Cohere)按调用计费,大流量下重排成本可观,常对热 query 做**语义缓存**;开源自托管(bge/Qwen3)换 GPU 成本,但可控、可微调。

**评估与可观测**
- 离线:用 **nDCG@k / MRR / Recall@k / Hit@k** 对比"重排前后"在自建标注集上的提升,这是证明 reranker 值不值得加的直接证据。
- 端到端:**Ragas** 的 `context_precision`(召回的料里相关比例,直接受重排影响)、`context_recall`;**TruLens / LangSmith / Phoenix** trace 每个候选的 rerank 分,定位"正确文档排第几"。
- 常驻基准:**BEIR**(零样本检索/重排)、**MTEB-Reranking**、中文 **C-MTEB**。

**踩坑与最佳实践**
- **召回 top_n 给够**:正确文档不在候选集,重排回天乏术。先保 Recall@N，再谈 rerank；N 从 50/100/200 等起始点实测。
- **重排粒度用小、喂生成回扩父块**:在句/段级 rerank(精度高、不超长),命中后回扩到父文档喂生成(信息全),配 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]]。
- **接 MMR 去冗余**:top-5 全近重复会浪费上下文,需覆盖面时加 MMR。
- **域不匹配就微调**:通用 reranker 在法律/医疗/代码可能不准,bge/Qwen3 可用域内 (query, pos, neg) 三元组微调。
- **别无证据地串联多个 reranker**:每层都会叠加延迟与故障面；只有在消融实验的 nDCG@k/业务指标能覆盖 p95 与成本时才保留。

## 面试高频

**Q1:bi-encoder 和 cross-encoder 有什么区别?为什么 RAG 要两阶段?**
标准答:bi-encoder(双塔)把 query 和 doc **各自独立编码**成向量,事后算余弦,doc 向量可离线预算 + 建 ANN 索引,适合大规模候选召回,但编码时 query/doc 互相看不到、丢 token 级交互;cross-encoder 把 `[query|SEP|doc]` **拼成一条**过 Transformer,token 级交叉注意力直接出相关分,但**每对都得现场过模型、无法预存建索引**,不适合扫全库。所以常分两阶段:可索引召回先优化 Recall@N，再由 cross-encoder 精排；N/k 由 nDCG@k、p95、文本长度与硬件确定。
- 追问"cross-encoder 为什么不能建索引?":它的分依赖 query 和 doc 的**联合表示**,没有"独立的 doc 向量"可预存;换一个 query,所有分都要重算。
- 陷阱:有人答"cross-encoder 更准所以全用它"——错,扫百万库要百万次前向,延迟不可接受。

**Q2:重排放在 RAG 流程哪一步?为什么不一步到位?**
标准答:召回([[08 混合检索 Hybrid Search|混合检索 Hybrid Search]])之后、生成([[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]])之前,把检索质量"收口"。不一步到位是因为召回要使用可扩展的索引信号(bi-encoder、稀疏或混合检索)，精排可承担更贵的 query-doc 联合计算；两类目标和预算不同。

**Q3:重排只保证相关,不保证什么?怎么补?**
标准答:不保证候选之间**不重复**——top-5 可能是 5 篇近重复,信息密度低还挤占上下文。用 **MMR**(`λ·相关 - (1-λ)·与已选最大相似`)在相关性和多样性间权衡去冗余。
- 追问"λ 怎么取?":λ→1 偏纯相关(可能冗余),λ→0 偏纯多样(可能跑题)。可从 0.5、0.7、0.9 起测，以覆盖率、nDCG 与人工答案质量定值。

**Q4:召回 top_n 设多大?设太小会怎样?**
标准答:可从 50、100、200 开始。太小则正确文档可能根本不在候选集里,重排只能在候选内重排、无力回天——**重排救不回召回的漏召**。所以先看 Recall@N，再用 nDCG@k 与 p95 决定 N。
- 追问"那为什么不无限大?":每个候选都要过一次 cross-encoder,top_n 越大延迟越高,需在召回率和延迟间权衡。

**Q5:ColBERT 这种"后期交互"和 bi/cross 是什么关系?**
标准答:介于两者之间。bi-encoder 把 doc 压成**单个向量**(交互全丢),cross-encoder **全 token 联合**(算不动全库),ColBERT 折中:doc 保留**每个 token 的向量**可预存建索引,在线只算 query token 与 doc token 的 MaxSim——保住部分 token 级交互又能扩展。
- 陷阱:问"ColBERT 算 bi 还是 cross"——都不是,是 late interaction 第三类。

**Q6(场景题):向量检索 top-5 里正确答案排第 8,怎么办?**
标准答:① 扩大召回 top_n 让第 8 进候选;② 加 cross-encoder 重排把它顶到前面;③ 检查是否该上混合检索补关键词召回。能指出"先保召回进候选、再靠重排提排序"是关键。

## 知识拓展

**进阶与前沿(带年份)**
- **RankGPT(2023,EMNLP Outstanding)**:LLM 输出候选**排列**而非逐个打分；论文的结果依赖其 TREC 数据、窗口和提示设置。它启发了 listwise 重排，但上线仍要测提示稳定性、token 成本与失败回退。
- **后期交互演进**:**ColBERT(2020)→ ColBERTv2(2021,残差压缩)→ PLAID(2022,检索加速)**。Qdrant、Vespa、Weaviate 的能力与 API 门槛见上表；不要把“多个命名向量”误写成 MaxSim。
- **不要臆测同一基座**:某厂商同时提供 embedding 与 rerank API，或模型名称相近，都不足以证明共享同一 base model；只有模型卡/官方架构说明明确时才能这样写。
- **listwise / setwise LLM 重排**:pointwise 逐对评分，listwise 一次看一组排序(如 RankGPT)，setwise 按集合比较；它们改变的是比较信息与调用形态，是否提升准确率要在同一数据、窗口和预算下验证。
- **重排即压缩**:CRAG 的 decompose-then-recompose、LongLLMLingua 等把"重排 + 过滤无关句"合并,既排序又压上下文,服务 [[20 上下文工程|上下文工程]]。

**底层联系**:bi-encoder 的余弦打分本质是**向量点积/归一化相似度**,见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]]——理解"为什么独立编码 + 点积会丢交互",就理解了为什么需要 cross-encoder。

**边界与反模式**
- **反模式一:跳过重排直接喂向量 top-k**——向量召回排序常不够准,真正最相关的可能排第 8/15,生成层吃到次优上下文。
- **反模式二:重排但召回 top_n 太小**——候选里没正确文档,重排是无米之炊。两个坑常一起犯。
- **反模式三:用通用 reranker 硬套专业域**——法律/医疗/代码语义偏移大,不微调会跑偏。
- **边界**:候选已经很干净(强混合检索 + 精确过滤)时重排边际收益小;延迟极敏感(<50ms)的实时场景,cross-encoder 的逐对前向可能就是预算瓶颈,得退到小 reranker 或 late interaction。

**相关链接**:上游 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 提供候选、[[07 查询变换 Query Transformation|查询变换 Query Transformation]] 扩大候选后更需重排收口;下游接 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]];与 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 配合(小粒度重排 + 回扩父文档);也是 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 中"相关性过滤"的工程化对应物。

## 来源
- Cohere. [Rerank overview](https://docs.cohere.com/docs/rerank-overview)、[Best practices](https://docs.cohere.com/docs/reranking-best-practices) 与 [Rerank API](https://docs.cohere.com/reference/rerank)（官方文档，2026-07-17 核验）：`rerank-v4.0-pro`/`fast` 与 `rerank-v3.5` 的模型名、多语言支持、版本化上下文窗口及 `max_tokens_per_doc` 默认值。
- Qdrant. [Vectors](https://qdrant.tech/documentation/manage-data/vectors/) 与 [Hybrid and multi-stage queries](https://qdrant.tech/documentation/search/hybrid-queries/)（官方文档，2026-07-17 核验）：Query API/multi-vector 自 v1.10.0，`using`、`prefetch`、`MAX_SIM` 语义。
- Vespa. [Embedding: ColBERT](https://docs.vespa.ai/en/rag/embedding.html) 与 [Announcing Vespa Long-Context ColBERT](https://blog.vespa.ai/announcing-long-context-colbert-in-vespa/)（官方文档，2026-07-17 核验）：mixed tensor、rank profile MaxSim；数组输入自 Vespa 8.303.17。
- Weaviate. [Vectorizer and vector index config](https://docs.weaviate.io/weaviate/manage-collections/vector-config) 与 [Vector similarity search](https://docs.weaviate.io/weaviate/search/similarity)（官方文档，2026-07-17 核验）：multi-vector 自 v1.30，限 named-vector HNSW；前者给出 `Configure.MultiVectors`，后者给出 named-vector 检索的 `target_vector` 接口。
- Sun, Yan, Ma, Wang, Ren, Chen, Yin, Ren.《Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents》(RankGPT). arXiv:2304.09542, EMNLP 2023(Outstanding Paper).
- Khattab, Zaharia.《ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT》(后期交互,介于 bi/cross 之间). arXiv:2004.12832, SIGIR 2020.
