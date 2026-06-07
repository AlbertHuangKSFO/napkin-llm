[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 的本质是:同时跑**稀疏检索**(BM25/SPLADE,词法精确匹配)和**稠密检索**(向量,语义匹配)两路,再把两份排名列表融合成一份。理由是这两类检索器的**强弱正好互补**——稠密擅长语义/换种说法,稀疏擅长专名/罕见词/精确字串;单用任一路都有系统性盲区,合起来召回更稳。

它接在 [[07 查询变换 Query Transformation|查询变换 Query Transformation]] 之后、[[10 重排序 Reranking|重排序 Reranking]] 之前:变换把 query 改好,混合检索把候选召得全,重排把候选排得准。和 [[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 是同一根筋——那篇讲 grep(极致稀疏精确)和向量(稠密语义)在 agent 场景的取舍,这里讲在 RAG 检索层把两者**同时用并融合**。

## 本质:稠密的盲区,正好是稀疏的强项

[[04 Embedding 与向量数据库|Embedding 与向量数据库]] 里的稠密向量检索,把 query 和 doc 都压成一个低维向量算余弦相似度。它的强项是**语义泛化**:"心脏病发作"能召回"心肌梗死"。但它有几类系统性弱点:

- **专有名词 / 罕见词 / 代码标识符**:`getUserById`、`SKU-XJ9920`、生僻人名药名——这些 token 在预训练里出现少,embedding 表示糊,稠密召回可能完全错过。但用户往往**就是要精确匹配这串字符**。
- **精确字串 / 数字 / 否定**:"不含麸质"vs"含麸质",稠密向量可能看作近义;BM25 看到"不"这个词就分开了。
- **域外 / 长尾**:embedding 模型没见过的领域,稠密泛化失效;BM25 是无监督的词频统计,不挑领域,长尾上更稳。

稀疏检索(以 **BM25** 为代表)恰好补这些:它是**词袋 + 倒排索引 + TF-IDF 家族打分**,本质是"query 里的词在 doc 里出现得多不多、稀不稀有"。专名罕见词正是它的舒适区——词一致就命中,不需要语义理解。

所以不是二选一,而是**两路都跑、各取所长**。

![[混合检索 Hybrid Search.png]]

## 机制

### 两路召回
- **稀疏路**:query 过倒排索引,BM25 打分,返回一个排名列表。
- **稠密路**:query 过 embedding 模型成向量,过 ANN 索引(HNSW/IVF,见 [[04 Embedding 与向量数据库|Embedding 与向量数据库]]),返回另一个排名列表。
两路**各自独立、可并行**,各返回 top-k。

### 融合:RRF vs 加权分数
难点是两路的**分数不可比**:BM25 分数范围 0~几十,余弦相似度 0~1,直接相加没意义。两种融合法:

**(1) RRF(Reciprocal Rank Fusion)——主流首选**。不看原始分数,**只看名次**:文档在某路排第 `rank` 位,贡献 `1/(k + rank)` 分(k 常取 60),两路得分相加,按总分重排。
- 出处:Cormack, Clarke, Büttcher《Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods》(SIGIR 2009, pp.758-759)。论文结论:RRF 这个极简方法稳定优于 Condorcet Fuse 和单系统。
- 为什么是它:免去对齐异质打分的麻烦;两路都排前的文档天然综合得分高;鲁棒、几乎无需调参。RAG-Fusion(见 [[07 查询变换 Query Transformation|查询变换 Query Transformation]])融合多个改写 query 用的也是同一个 RRF。

**(2) 加权分数(weighted / convex)**:把两路分数各自归一化(min-max 或 z-score)后加权求和 `α·dense + (1-α)·sparse`。更灵活(能调权重偏向某路),但**需要归一化 + 调 α**,对分布漂移敏感,不如 RRF 省心。

### 进阶稀疏:SPLADE(学习型稀疏)
BM25 是无监督词频;**SPLADE** 是用 BERT 学出来的稀疏表示:它给每个词一个学习到的权重,还能做**词项扩展**(query"car"自动激活"vehicle/automobile"的权重),把"语义"塞进稀疏向量里,仍跑在倒排索引上(快)。本质是"带语义扩展的可学习 BM25"。出处:Formal, Piwowarski, Clinchant《SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking》(arXiv:2107.05720, SIGIR 2021);v2 见 arXiv:2109.10086。

### 介于稀疏稠密之间:ColBERT 后期交互
**ColBERT** 不是"一个向量配一篇文档",而是**每个 token 一个向量**,query 的每个 token 向量去和 doc 所有 token 向量算最大相似度(**MaxSim**),再求和当相关性分。这叫**后期交互(late interaction)**:既保留了 token 级细粒度匹配(接近 cross-encoder 的精度,见 [[10 重排序 Reranking|重排序 Reranking]]),又能**离线预算 doc 的 token 向量**(接近 bi-encoder 的可索引性)。所以它在"稀疏精确 ↔ 稠密语义"光谱上居中偏强。出处:Khattab, Zaharia《ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT》(arXiv:2004.12832, SIGIR 2020)。代价:存储暴涨(每 token 一个向量),工程更重。

## 可跑最小代码

```python
# 混合检索:BM25(稀疏)+ 向量(稠密)各自召回,RRF 融合
def hybrid_search(query, bm25, vstore, embed, k=10, k_rrf=60, topn=5):
    # 1) 两路各自召回排名列表(返回 [doc_id, ...] 按相关性降序)
    sparse_ranked = bm25.search(query, k=k)            # 词法:专名/罕见词强
    dense_ranked  = vstore.search(embed(query), k=k)   # 语义:换种说法也召回

    # 2) RRF 融合:只用名次,免对齐异质分数
    scores = {}
    for ranked_list in (sparse_ranked, dense_ranked):
        for rank, doc_id in enumerate(ranked_list):    # rank 从 0 起
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k_rrf + rank)

    # 3) 按融合分重排,取 top-n(通常再喂 reranker 精排)
    return sorted(scores, key=scores.get, reverse=True)[:topn]
```

要点:① 两路**独立召回**,可并行;② RRF 用 `1/(k_rrf+rank)`,两路都靠前的 doc 累加得分最高,自动奖励"双路共识";③ k_rrf=60 是论文经验值,几乎不用调;④ 输出通常**不直接生成**,而是接 [[10 重排序 Reranking|重排序 Reranking]] 用 cross-encoder 收口。

## 对比表

| 维度 | 稀疏 BM25 | 学习型稀疏 SPLADE | 稠密向量 | ColBERT 后期交互 |
|---|---|---|---|---|
| 匹配方式 | 词法精确 | 词法 + 学习扩展 | 语义 | token 级 MaxSim |
| 专名/罕见词 | 强 | 强 | 弱 | 较强 |
| 换种说法 | 弱 | 中 | 强 | 强 |
| 索引 | 倒排(快) | 倒排(快) | ANN | token 向量(重) |
| 训练 | 无监督 | 需训练 | 需训练 | 需训练 |
| 存储 | 小 | 小 | 中 | 大 |

## 何时用 / 坑

**该用混合检索**:几乎所有生产 RAG 的默认推荐——尤其语料含大量专名/代码/ID/数字、用户会精确查字串、或领域偏长尾时,纯稠密会漏,混合显著更稳。这是修复 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 召回缺口的标配手段。

**坑**:
- **以为稠密万能**:最常见误区。稠密对专名/罕见词/精确匹配确实弱,别用一路打天下。
- **直接相加原始分**:BM25 和余弦分数不可比,相加是错的。用 RRF(只吃名次)或先归一化再加权。
- **两路 k 设太小**:某路 top-k 太浅会漏掉只在另一路强的文档,融合也救不回。两路都给够候选(如各 top-50/100)再融合 + 重排。
- **SPLADE/ColBERT 的工程成本**:SPLADE 要训练和服务模型;ColBERT 存储是 token 级,索引膨胀几倍到几十倍。中小项目 BM25 + 稠密 + RRF 往往够用,别上来就堆 ColBERT。
- **融合后不重排**:混合召回扩大了候选也带进噪音,最好接 [[10 重排序 Reranking|重排序 Reranking]] 收口,否则脏候选进生成层。

## 工业界实践

混合检索是**几乎所有生产 RAG 的默认推荐**,而且现在大多数检索引擎已经把「双路召回 + RRF 融合」做成了原生功能,不必自己拼。

### 主流引擎的原生混合检索

到 2025–2026,**RRF 已成为多数引擎的默认混合融合方法**:Elasticsearch、OpenSearch、Azure AI Search、MongoDB Atlas、Weaviate、Qdrant、Pinecone 都内置「BM25 倒排 + 向量 ANN 同库并行召回 + RRF 融合」,k 默认取 60。这意味着:

- **Elasticsearch / OpenSearch**:倒排索引和向量索引在同一系统里,一条 query 并行跑两路、内置 RRF 融合,无需在应用层拼接。Elasticsearch 的 `rrf` retriever、OpenSearch 的 hybrid query 都是开箱即用。
- **pgvector + Postgres 全文检索**:用 `tsvector`(BM25 风格全文)+ `pgvector` 向量,应用层或扩展(如 ParadeDB / pg_search 提供真 BM25)做 RRF,适合「不想引专用向量库」的中小项目。
- **专用向量库**:Weaviate、Qdrant、Milvus、Pinecone 都有 hybrid search API,直接给 `alpha`(加权)或选 RRF。

工程上**优先用现有引擎的原生混合**,别手写两路再融合——原生实现已优化好并行召回与融合,生产数据点:原生混合相比纯向量约 **+6ms 延迟、约 1.4× 存储**,代价很低。

### 索引选型(稠密侧)

混合检索的稠密路仍跑 ANN 索引,选型沿用 [[04 Embedding 与向量数据库|Embedding 与向量数据库]]:
- **HNSW**:几乎所有库的默认(pgvector / Qdrant / Weaviate / Milvus / Pinecone / Chroma),查询快、召回高,内存占用大。中小规模首选。
- **IVF(倒排文件)**:把向量按 k-means 聚类分桶,内存比 HNSW 省,适合**上亿级**向量。
- **量化(PQ / SQ)**:内存瓶颈时叠加,Product Quantization 压缩 8×~64×、内存降约 75%、精度可控损失,常与 HNSW/IVF 组合。

### 学习型稀疏与后期交互的工程取舍

- **SPLADE**:可学习稀疏 + 词项扩展,仍跑倒排(快),但要训练/服务一个模型,索引词项更密(膨胀)。中大项目且 BM25 在你语料上明显不够时再上。
- **ColBERT**:token 级向量 + MaxSim,精度近 cross-encoder,但**存储膨胀几倍到几十倍**(每 token 一个向量),工程重。生产里多作为「介于召回与重排之间」的强力一档,中小项目通常 `BM25 + 稠密 + RRF` 就够,别上来堆 ColBERT。

### 评估、可观测与最佳实践

- **评估**:用 [[18 RAG 评估|RAG 评估]] 的 Ragas **context recall / context precision** 对比「纯稠密 / 纯 BM25 / 混合」三档。生产经验:**含大量专名/代码/ID/数字的语料,混合相对纯稠密的 context recall 提升最明显**——这正是稠密的盲区、稀疏的强项。
- **两路 k 给够再融合**:某路 top-k 太浅会漏掉只在另一路强的文档,融合也救不回。两路各给 top-50/100,融合后再 rerank 取 top-5~10。
- **融合后接 rerank 收口**:混合扩大候选也带噪,接 [[10 重排序 Reranking|重排序 Reranking]] 的 cross-encoder(BGE-reranker / Cohere Rerank)精排,这是「召回求全→重排求准」的标准两段式。
- **可观测**:记录「每路各自的命中贡献 / 融合后 top-k 里 sparse-only 与 dense-only 的占比」,用来判断两路是否都在干活——若某路贡献长期接近 0,说明它在你语料上没价值,可砍掉省成本。

### 召回-重排两段式(配套全景)

```
query(经 07 查询变换)
  ├─ 稀疏路:BM25 倒排 → top-100
  └─ 稠密路:embed → HNSW ANN → top-100
        → RRF 融合(Σ 1/(60+rank))→ top-50
        → reranker(cross-encoder, 10 重排序)→ top-5
        → 生成
```

## 面试高频

**Q1:为什么要混合检索?稠密向量不是已经能处理语义了吗?**
标准答:稠密擅长语义泛化(「心脏病发作」召回「心肌梗死」),但有系统性盲区——**专名/罕见词/代码标识符/精确字串/数字/否定**。这些 token 预训练里少见、embedding 表示糊,稠密可能完全错过,而用户往往就是要精确匹配这串字符。BM25 这类稀疏检索靠词频精确匹配,恰好补这些盲区。所以不是二选一,是两路都跑、各取所长。
- 追问「举个稠密会漏的例子」→ `getUserById`、`SKU-XJ9920`、「不含麸质」vs「含麸质」(否定词稠密可能看作近义)。

**Q2:两路的分数怎么融合?为什么不直接相加?**
标准答:不能直接相加,因为 BM25 分数范围 0~几十、余弦相似度 0~1,**量纲不可比**,相加无意义。两种融合法:① **RRF(主流首选)**:只看名次,文档在某路排第 rank 位贡献 `1/(k+rank)`(k=60),两路相加重排——免对齐异质分数、鲁棒、几乎不调参;② **加权分数**:两路各自归一化后 `α·dense+(1-α)·sparse`,更灵活但要调 α 且对分布漂移敏感。
- 追问「RRF 出处和为什么 k=60?」→ Cormack et al. SIGIR 2009,k=60 是论文经验值,平滑高排名文档的差距,几乎不用动。

**Q3:SPLADE 和 BM25 的区别?ColBERT 又是什么?**
标准答:**SPLADE** 是用 BERT 学出来的稀疏表示——给每个词学习权重并做**词项扩展**(query「car」自动激活「vehicle/automobile」),把语义塞进稀疏向量,仍跑倒排索引(快),本质是「带语义扩展的可学习 BM25」(arXiv:2107.05720)。**ColBERT** 是**每个 token 一个向量**,query 每个 token 向量与 doc 所有 token 向量算 MaxSim 求和——**后期交互(late interaction)**,精度近 cross-encoder 又能离线预算 doc 向量,居稀疏稠密之间(arXiv:2004.12832);代价是存储暴涨。
- 陷阱:别把 ColBERT 归为「稠密」或「稀疏」——它是 late interaction,在光谱中间。

**Q4(陷阱):混合检索后还需要 rerank 吗?**
标准答:**通常需要**。混合扩大了召回也带进噪音,RRF 只是名次融合、不是精排。接 cross-encoder reranker([[10 重排序 Reranking|重排序 Reranking]])对 top-k 候选做 query-doc 联合编码精排,才能把脏候选挡在生成层之外。混合「求全」、rerank「求准」,是标准两段式。

**Q5:位置——混合检索在整个 RAG 检索栈里处在哪?**
标准答:接在 [[07 查询变换 Query Transformation|查询变换 Query Transformation]] 之后(变换把 query 改好)、[[10 重排序 Reranking|重排序 Reranking]] 之前(重排把候选排准)。它是「召得全」这一环,前面「问得对」、后面「排得准」。

## 知识拓展

- **RRF 是检索栈的通用融合算子**:它不止用于「稀疏+稠密」,RAG-Fusion([[07 查询变换 Query Transformation|查询变换 Query Transformation]])用同一个 RRF 融合多个改写 query 的召回。只吃名次、不吃原始分,所以能融合任何异质排名列表——理解一次,处处可用。
- **稀疏-稠密光谱**:从纯词法到纯语义是一条连续谱——BM25(无监督词频)→ SPLADE(可学习稀疏+扩展)→ ColBERT(token 级 late interaction)→ 稠密 bi-encoder(单向量语义)。越往右语义泛化越强、对专名越弱,越往左相反。混合检索本质是取这条谱的两端并融合。
- **与相似度计算的根**:稠密路最终落到向量相似度(余弦/点积),其数学基础见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]];ColBERT 的 MaxSim 也是 token 向量间的点积取最大,是同一套相似度运算的细粒度版本。
- **与 Agentic Search 同根**:[[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 讲 agent 场景里 grep(极致稀疏精确)和向量(稠密语义)的取舍,本质和混合检索是同一个「词法 vs 语义」权衡——只是 RAG 检索层选择「同时用并融合」,agent 场景则按任务动态选用。
- **边界与反模式**:① 「稠密万能」是头号误区(专名/精确匹配会漏);② 直接相加原始分是错的(量纲不可比);③ 两路 k 设太浅是反模式(漏掉单路强项);④ 中小项目上来就堆 SPLADE/ColBERT 是过度工程(`BM25+稠密+RRF` 通常够);⑤ 融合后不接 rerank 是反模式(脏候选进生成)。
- **前沿**:学习型稀疏(SPLADE v2/v3、BGE-M3 的 sparse 头)与 late interaction(ColBERTv2、PLAID 索引压缩)持续在「兼得稀疏的可索引性与稠密的语义」上推进;BGE-M3(2024)更是单模型同时输出 dense / sparse / multi-vector 三种表示,一个模型把混合检索三路全包,是值得关注的统一方向。

## 关键事实

- 混合检索 = **稀疏(词法精确)+ 稠密(语义)两路召回 + 融合**;核心理由是两者盲区互补,稠密对专名/罕见词/精确匹配弱,稀疏补。
- 融合首选 **RRF**(Cormack et al. SIGIR 2009):只用名次 `Σ 1/(k+rank)`,免对齐异质分数,鲁棒少调参;次选归一化加权分数。
- **SPLADE**(arXiv:2107.05720, SIGIR 2021)= 可学习的稀疏 + 词项扩展,仍跑倒排索引;把语义塞进稀疏。
- **ColBERT**(arXiv:2004.12832, SIGIR 2020)= token 级向量 + MaxSim 后期交互,精度近 cross-encoder、可离线预算,居于稀疏稠密之间;代价是存储膨胀。
- 位置:接在 [[07 查询变换 Query Transformation|查询变换 Query Transformation]] 之后、[[10 重排序 Reranking|重排序 Reranking]] 之前;与 [[04 Embedding 与向量数据库|Embedding 与向量数据库]] 是同层的稠密侧基础。
- 同根:[[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]] 讲 agent 场景的稀疏(grep)/稠密取舍,这里是 RAG 检索层"同时用并融合"。

## 来源
- Cormack, Clarke, Büttcher.《Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods》. SIGIR 2009, pp.758-759.
- Formal, Piwowarski, Clinchant.《SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking》. arXiv:2107.05720, SIGIR 2021;v2 arXiv:2109.10086.
- Khattab, Zaharia.《ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT》. arXiv:2004.12832, SIGIR 2020.
