[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 是把**稀疏路**（BM25 或 SPLADE，词法和专名）与**稠密路**（embedding 向量，语义和改写）分别召回，再按明确配置的 fusion 规则合成一个候选集。它接在 [[07 查询变换 Query Transformation|查询变换]] 之后、[[10 重排序 Reranking|重排序]] 之前：先问得对，再召得全，最后排得准。

## 直觉：两种雷达，别假定同一个调音台

把检索想成两台雷达。BM25 对 `SKU-XJ9920`、`getUserById`、人名和精确数字敏感；稠密向量更容易把「心脏病发作」和「心肌梗死」连起来。两路各有盲区，因此可以共同供给候选；但**融合方式是引擎、版本和配置的一部分，不是一个跨引擎的默认事实**。

这也是 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 的检索层版本：agent 可按任务选择 grep 或向量；RAG 则可让两路同时参与，再由 fusion 决定谁排在前面。稠密侧的 ANN 基础见 [[04 Embedding 与向量数据库|Embedding 与向量数据库]]。

## 小数字手算：RRF 的名次从 1 开始

令稀疏路排名为 `[A, B, C]`，稠密路排名为 `[B, A, D]`，并暂取平滑常数 $K=60$。RRF 不比较原始分数，而是计算：

$$
\operatorname{RRF}(d)=\sum_{i=1}^{m}\frac{1}{K+\operatorname{rank}_i(d)},\qquad \operatorname{rank}_i\in\{1,2,\ldots\}.
$$

所以 $A$ 的分数为 $1/(60+1)+1/(60+2)=0.03252$；$B$ 为 $1/(60+2)+1/(60+1)=0.03252$；$C$ 仅为 $1/(60+3)=0.01587$。这里**第一名是 rank=1，绝不能从 0 开始枚举**；否则首位会被错误地算成 $1/60$。

`K=60` 是 RRF 论文和部分产品 API 的常用平滑值，不是所有系统的默认值，也不应跳过离线评估就固化它。

## 机制：BM25、dense、SPLADE、ColBERT 各在何处

### 两路候选与两类融合

- **稀疏路**：query 查 BM25 倒排索引，返回词法排名；它擅长专名、代码、ID、精确短语。**SPLADE** 将 BERT 学得的词项权重和扩展保留为稀疏向量，仍可跑倒排索引，因此是学习型的稀疏召回，而不是对 BM25 分数的简单替换。
- **稠密路**：query 经 embedding 后进入 ANN（例如 HNSW、IVF），按向量相似度返回语义排名。
- **RRF**：只融合名次，适合两路分数尺度不可直接比较时；$K$、每路 candidate depth 和去重策略都是可调参数。
- **归一化加权**：先让每路分数进入可比尺度，再做 $\alpha\,s_{dense}+(1-\alpha)\,s_{sparse}$。$\alpha$ 是待验证的工作负载参数，不是「0.5 永远最好」。

### ColBERT 是第三种取舍，不是免费升级

**ColBERT** 为 query 和文档的每个 token 保留向量，以后期交互（late interaction）计算：

$$
\operatorname{MaxSim}(q,d)=\sum_{t\in q}\max_{u\in d}\left(q_t^\top d_u\right).
$$

这保留 token 级匹配，又允许离线索引文档向量；它位于稀疏精确与单向量稠密语义之间。代价不是一个可移植的「几倍存储」结论，而是要在自己的 token 长度、压缩、硬件与候选深度下测量索引大小、构建时间、p95 和排序质量。

![[混合检索 Hybrid Search.png]]

## 可运行代码：❌ 原始分相加，✅ 以 1 为起点的 RRF

```python
# ❌ 不同检索器的 BM25/余弦/点积分布通常不可直接相加。
bad_score = 18.2 + 0.81
```

```python
# ✅ 纯标准库，可直接运行：python3 hybrid_rrf.py
from collections import defaultdict
from math import isclose


def rrf_scores(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse deduplicated ranked IDs; every ranking position starts at 1."""
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: defaultdict[str, float] = defaultdict(float)
    for ranked_ids in rankings:
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError("one retriever returned a duplicate document ID")
        for rank, doc_id in enumerate(ranked_ids, start=1):
            assert rank >= 1
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores = rrf_scores(rankings, k=k)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


if __name__ == "__main__":
    sparse = ["A-exact-id", "B-semantic", "C-lexical"]
    dense = ["B-semantic", "A-exact-id", "D-paraphrase"]
    scores = rrf_scores([sparse, dense], k=60)

    # This catches the classic enumerate(..., start=0) bug.
    assert isclose(rrf_scores([["first"]], k=60)["first"], 1 / 61)
    assert isclose(scores["A-exact-id"], 1 / 61 + 1 / 62)
    assert rrf_fuse([sparse, dense], k=60)[:2] == ["A-exact-id", "B-semantic"]
    print(rrf_fuse([sparse, dense], k=60))
```

代码只实现「得到两份 ID 排名后如何 RRF」；真实系统还须明确每一路的 `top_k`、过滤条件、ID 去重以及融合后传给 [[10 重排序 Reranking|重排序]] 的候选数。

## 引擎行为：写出引擎、最低已核验版本、API 与 fusion 配置

下表是**截至 2026-07-17 查阅官方文档**的可复现基线。它不把某个 SDK 封装层的默认值外推为底层引擎行为。

| 引擎 | 最低已核验版本／边界 | API | fusion 配置与含义 |
|---|---|---|---|
| [Weaviate](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search) | **v1.24+** | `collection.query.hybrid(...)` | 默认 `relativeScoreFusion`：分别归一化 vector 与 BM25 分数后融合；`alpha=0.5` 为等权。需要名次融合时显式选 `rankedFusion`，不要把它叫作默认 RRF。Weaviate 官方文档（访问 2026-07-17）明确 v1.23 及以下的默认才是 `rankedFusion`。 |
| [Pinecone Vector API](https://docs.pinecone.io/guides/search/hybrid-search) | 托管 API；官方未给单独的 server 最低版本 | `index.query(vector=..., sparse_vector=..., top_k=...)` | **单索引**让同一 record 同时带 dense 与 sparse vector，并以 `dotproduct` 查询；在客户端将 query dense 乘 $\alpha$、sparse 乘 $(1-\alpha)$，实现加权。官方没有该查询的内建 `alpha` 参数。若改成**双索引**，应各查一次、按共享 ID 去重，并在客户端做 RRF（或其他已验证 fusion）。官方文档（访问 2026-07-17）特别说明 sparse 权重未与 dense 分数自动对齐。 |
| [Milvus](https://milvus.io/docs/v2.4.x/multi-vector-search.md) | **v2.4.x** 文档已核验；部署与 PyMilvus 应使用匹配版本 | `Collection.hybrid_search(reqs, rerank, limit=...)` | 对每个 `AnnSearchRequest` 后，**显式**传 `RRFRanker(k=...)` 才是在请求中选择 RRF；也可显式传 `WeightedRanker(...)`。不要依赖其他集成层可能提供的默认 ranker。Milvus 官方 v2.4.x 文档（访问 2026-07-17）将 ranker 作为 `hybrid_search` 的参数。 |

同一产品的 client、托管服务和文档都会演进：升级前把实际版本、API request 和 fusion 参数写入实验记录，再复跑评估。`k=60` 仅在你**显式配置** RRF 时才有含义；它不适用于 Weaviate 的 relative-score fusion，也不替代 Pinecone 单索引的 $\alpha$。

## 选型与评估：把性能说成可测量取舍

不要写「HNSW 一定更快」「IVF 适合某个固定规模」「PQ 必然节省某个比例」或「ColBERT 固定膨胀若干倍」。这些受向量维度、数据分布、压缩参数、过滤率、硬件、并发和候选深度共同影响。针对同一份带标注 query 集，固定过滤条件和目标候选数，比较下列配置：

| 方案／要调的旋钮 | 可能的取舍假设 | 必须记录的实测结果 |
|---|---|---|
| HNSW：`M`、`efConstruction`、`efSearch` | 图更密或搜索更深，可能换取更高召回，也可能增加构建资源与查询开销 | Recall@N、nDCG@k、索引大小、构建时间、p95 |
| IVF：`nlist`、`nprobe` | 桶数与探测桶数改变候选范围；不能只按向量总量决定 | Recall@N、nDCG@k、索引大小、构建时间、p95 |
| PQ／SQ：码本、子空间数、bit 数 | 更强压缩可能降低内存／磁盘，但需观察量化误差是否伤害排序 | Recall@N、nDCG@k、索引大小、构建时间、p95 |
| ColBERT：token 上限、压缩／索引、候选 depth | token 级交互可能改善细粒度匹配，也会改变索引和查询成本 | Recall@N、nDCG@k、索引大小、构建时间、p95 |
| BM25 + dense + fusion：每路 `top_k`、$K$ 或 $\alpha$ | 候选更深可能提升后续可选空间，也可能带来延迟与噪声 | Recall@N、nDCG@k、端到端 p95，以及 sparse-only／dense-only／交集贡献 |

一个最低可用实验是：对纯 BM25、纯 dense、混合加权、混合 RRF（若引擎支持或客户端实现）分别计算 Recall@50、nDCG@10、索引大小、构建时间和 p95；再对专名／ID、改写、长尾三类 query 分桶。仅当排序收益覆盖资源代价时，再加入 SPLADE、ColBERT 或更激进的 ANN 压缩。

⚠️ **常见误区**：原始分数直接相加、RRF 使用 0-based rank、把厂商默认 fusion 当通用算法、两路 `top_k` 太浅、或将召回候选直接送入生成。混合是在「召回求全」一层；通常仍应由 [[10 重排序 Reranking|重排序]] 对融合后的候选精排。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1：为什么混合，而不是只用 dense？**  稠密路擅长同义改写，但专名、ID、代码和精确字串可能是稀疏路的强项。应在带标注的目标语料上验证两者互补，而非把互补当成无条件收益。

**Q2：RRF 的公式里 rank 从几开始？**  从 **1** 开始。$\sum_i 1/(K+rank_i)$ 的首位贡献为 $1/(K+1)$；实现需 `enumerate(..., start=1)`，并保留如上断言。

**Q3：Weaviate、Pinecone、Milvus 都默认 RRF 吗？**  不对。Weaviate v1.24+ 默认 relative-score fusion；Pinecone 单索引使用客户端的 $\alpha$ 缩放，双索引才在客户端融合；Milvus 应在 `hybrid_search` 请求中显式选择 `RRFRanker`。三个结论都要随实际版本复核。

**Q4：何时用 SPLADE 或 ColBERT？**  当 BM25+dense 的 Recall@N／nDCG@k 对目标 query 已显示明确缺口，且索引大小、构建时间和 p95 仍在预算内。SPLADE 提供学习型稀疏扩展；ColBERT 提供 token 级后期交互，二者都不是只凭通用倍数就能选型的升级。

## 关键事实

- 混合检索的稳定定义是「独立的 sparse 与 dense 候选 + 已声明的 fusion」，而不是某一个引擎或默认参数。
- RRF 只吃名次，公式中的排名从 1 开始；$K$ 是平滑参数，应作为实验配置记录。出处：[Cormack、Clarke、Büttcher，SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)。
- SPLADE 是学习型稀疏检索与词项扩展，仍面向倒排索引。出处：[Formal、Piwowarski、Clinchant，SIGIR 2021，arXiv:2107.05720](https://arxiv.org/abs/2107.05720)。
- ColBERT 用 token 级 MaxSim 的后期交互；其收益与成本必须连同索引与延迟一起在目标工作负载上测量。出处：[Khattab、Zaharia，SIGIR 2020，arXiv:2004.12832](https://arxiv.org/abs/2004.12832)。
- Weaviate v1.24+ 的默认是 relative-score fusion；Pinecone 单索引是客户端 $\alpha$ 加权，双索引需客户端合并；Milvus 核心 API 可显式使用 `RRFRanker`。版本与 API 依据：[Weaviate 官方文档](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)、[Pinecone 官方文档](https://docs.pinecone.io/guides/search/hybrid-search)、[Milvus v2.4.x 官方文档](https://milvus.io/docs/v2.4.x/multi-vector-search.md)（均访问于 2026-07-17）。

## 来源

- [Weaviate Hybrid search 官方文档](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)，访问 2026-07-17。
- [Pinecone Hybrid search 官方文档](https://docs.pinecone.io/guides/search/hybrid-search)，访问 2026-07-17。
- [Milvus v2.4.x Hybrid Search 官方文档](https://milvus.io/docs/v2.4.x/multi-vector-search.md)，访问 2026-07-17；[Milvus RRF Ranker 官方文档](https://milvus.io/docs/rrf-ranker.md)，访问 2026-07-17。
- Cormack, Clarke, Büttcher. *Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009.
- Formal, Piwowarski, Clinchant. *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking*. SIGIR 2021.
- Khattab, Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*. SIGIR 2020.
