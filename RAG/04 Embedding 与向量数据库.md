[[04 Embedding 与向量数据库|Embedding 与向量数据库]] 是 RAG 检索的引擎室:**embedding** 把文本压成一串浮点向量,语义相近的文本在向量空间里靠得近;**向量数据库**则用 ANN 索引在给定的召回、延迟和容量目标下找出 query 的候选。前者决定「编码得准不准」,后者决定「找得快不快、占多少资源」。它直接承接 [[03 分块策略 Chunking|分块策略 Chunking]] 切出来的块,是 [[01 什么是 RAG|什么是 RAG]] 里 query→doc 匹配的物理实现。

## 本质:语义检索 = 把「找相似含义」变成「找近向量」

关键词检索(BM25)只会匹配字面;但「汽车」和「轿车」、"car" 和 "automobile" 字面不同、含义相同。**dense embedding** 的核心是:把文本映射到一个高维向量空间,让训练目标逼着**语义相近→向量相近**。于是「找语义相关文档」这个模糊问题,被还原成一个干脆的几何问题:**在向量空间里找离 query 向量最近的点**。

这就引出两个必须分别解决的子问题,本篇两半各管一个:
1. **怎么把文本编码成好向量?** → embedding 模型(bi-encoder)。
2. **怎么在海量向量里快速找最近邻?** → 向量数据库的 ANN 索引。

## 机制 A:bi-encoder 双塔与相似度

### bi-encoder(双塔)
RAG 检索用的是 **bi-encoder**:query 和 doc **各自独立**过同一个编码器,各得一个向量,再算相似度。关键好处——**doc 向量可以离线预先算好建库**,查询时只需编码 query 一次,然后做向量最近邻。这让它能 scale 到亿级。

对照 **cross-encoder**:把 (query, doc) 拼一起喂进模型输出一个相关度分,精度高得多但**无法预建库**(每个 query-doc 对都要现算),只能用在召回后的 [[10 重排序 Reranking|重排序 Reranking]] 阶段对 top-k 精排。**双塔召回 + 交叉精排** 是标准两段式。

`sentence-transformers`(SBERT)是 bi-encoder 的代表实现:在 BERT 上做对比学习,让正例对(相关句)向量拉近、负例推远,输出句向量。

![[Embedding 与向量数据库.png]]

### 相似度度量
- **cosine 余弦**:看方向夹角,忽略模长。**最常用**,对向量缩放不敏感。
- **dot product 点积**:方向 + 模长都算。若向量已 L2 归一化,点积 = cosine。许多库默认要求归一化后用点积(快)。
- **欧氏距离 L2**:几何直线距离。归一化向量下与 cosine 单调等价。
实践:**大多数 embedding 模型设计为配 cosine**;用哪个取决于模型怎么训的,别乱换。

**「归一化后 dot = cosine」手算**。取 $a=[3,4]$、$b=[4,3]$。原始点积 $a\cdot b = 3\times4+4\times3=24$,模长 $\|a\|=\|b\|=\sqrt{3^2+4^2}=5$,故 $\cos = \frac{24}{5\times5}=0.96$。现把两者 L2 归一化:$\hat a=[0.6,0.8]$、$\hat b=[0.8,0.6]$,归一化后再点积 $\hat a\cdot\hat b = 0.6\times0.8+0.8\times0.6=0.96$——与 cosine **分毫不差**。所以库里「先归一化、再用内积索引」(快)和「直接算 cosine」是同一件事。

## 机制 B:ANN 近似最近邻索引

精确最近邻要拿 query 和**每个** doc 算相似度,O(N)——亿级库下慢到不可用。**ANN(Approximate Nearest Neighbor)** 牺牲一点点召回(可能漏掉个别真最近邻)换数量级的提速。三大族系:

### HNSW(图索引)
**Hierarchical Navigable Small World**,源自 **Malkov & Yashunin 2016(arXiv:1603.09320)**。构建一个**多层近邻图**:底层 L0 含全量节点、连边稠密;往上每层节点指数减少、像「高速公路」。查询时从顶层稀疏图进入,**贪心跳到当前层离 q 最近的点,再降一层**,逐层逼近,最后在 L0 精搜。复杂度近 **O(log N)**。优点:召回高、查询快;缺点:**内存占用大**(要存图的边)、建索引慢、增删麻烦。绝大多数向量库的默认索引。

**HNSW 内存手算与容量模型**。1 千万条 768 维 float32 向量,光向量本体仍是 $10^7\times768\times4\,\text{B}=3.072\times10^{10}\,\text{B}\approx30.72\,\text{GB}$。但 HNSW 总内存不能可靠地写成固定倍数,应写成

$$B_{\text{HNSW}}=N\cdot d\cdot b+B_{\text{graph}}(N,M,M_0,\text{level distribution},\text{id width})+B_{\text{metadata}}+B_{\text{allocator}}.$$

其中 $N$ 是向量数,$d$ 是维度,$b$ 是每维字节数(如 float32 为 4),$M/M_0$ 是各层最大邻居数。图边的实际宽度、上层比例、ID 宽度和实现的内存分配都会改变 $B_{\text{graph}}$。因此先按 30.72 GB 预留向量本体,再在目标实现、目标参数和目标硬件上测量**常驻内存、磁盘、Recall@k/nDCG@k 与 p95 延迟**;达不到预算才比较量化、IVF 或分片,而不是套用「固定 1.5 倍」结论。

「逐层下降」是 HNSW 的灵魂,但它发生在哪、为什么近 $O(\log N)$,光看静态分层图看不出来——下图把**一次查询的导航过程**画出来:从顶层稀疏图(像「高速公路」一跳跨很远)进入,贪心跳到本层离 q 最近的点,降一层、再贪心,逐层逼近,最后在全量稠密的 L0 邻域内精搜出 top-k(`efSearch` 控候选队列大小,是召回/延迟旋钮)。每降一层节点数指数增多、层数 $\approx\log N$,所以逐层只走几步即可。

![[rag-HNSW分层导航.png]]

### IVF(倒排分桶)
**Inverted File**:先把所有向量 **k-means 聚类**成若干桶(质心)。查询时只算 q 到各质心的距离,**只搜最近的 nprobe 个桶**,跳过其余。若列表仍保留全精度向量,可先用

$$B_{\text{IVF-flat}}\approx N\cdot d\cdot b+n_{\text{list}}\cdot d\cdot b+N\cdot b_{\text{id}}+B_{\text{lists}}$$

估算;它是否比别的索引省内存取决于 ID、列表与实现开销,不能只由「IVF」这个名字断言。`nlist`、`nprobe` 和训练样本都要随语料测量,并报告 Recall@k/nDCG@k、p95 延迟、常驻内存和磁盘占用。`nprobe` 调小会**漏召回**(真最近邻落在没搜的桶里),边界样本尤甚。

### PQ(乘积量化)
**Product Quantization**:把每个向量切成 $m$ 段,每段用一个小**码本**(k-means 得到的码字集合)近似,只存「每段最近码字的 id」。若每个子码本有 $K$ 个码字,代码存储可先估为

$$B_{\text{codes}}=N\cdot m\cdot\lceil\log_2K/8\rceil,$$

还必须加上码本 $B_{\text{codebook}}\approx m\cdot K\cdot(d/m)\cdot b$、ID、IVF 质心与索引元数据。PQ 是有损近似;压缩比、Recall@k/nDCG@k、p95 延迟和是否保留全精度向量用于重打分，都必须在同一份语料上实测。常与 IVF 组合成 **IVF-PQ**,但并不预先保证某个规模、压缩倍数或召回。

此外 **ScaNN**(Google)用各向异性量化 + 部分重排,在精度/速度上很强,是另一条工业级路线。

![[向量索引-ANN.png]]

## 向量库谱(各自定位)

| 库 | 形态 | 定位 / 一句话 |
|---|---|---|
| **FAISS** | 库(Meta) | 不是数据库,是 ANN 算法库。要自己管持久化/过滤,做原型/嵌入式首选 |
| **pgvector** | Postgres 扩展 | 数据已在 Postgres 就地加向量列,**最省运维**,中小规模够用 |
| **Chroma** | 轻量嵌入式 | 开发友好、几行起步,本地原型 / 小项目 |
| **Qdrant** | 独立向量库(Rust) | 性能强、**payload 过滤**好,云原生,生产常选 |
| **Weaviate** | 独立(含 GraphQL) | 自带模块化向量化、混合检索、schema,功能全 |
| **Milvus** | 独立(分布式) | 多索引(IVF/HNSW/PQ)与分布式能力；是否适合取决于数据、过滤、运维与实验卡 |
| **Pinecone** | 全托管 SaaS | 不想运维就交给它,按用量付费,上手快 |

**选型起点**:数据在 Postgres 可先评 pgvector;本地原型可先评 Chroma / FAISS;需要 payload 过滤可将 Qdrant / Weaviate 纳入候选;需要分布式时可将 Milvus 纳入候选;需要全托管则评 Pinecone。最终仍用相同语料和实验卡验证。

## 候选模型、MTEB 快照与可复现实验卡

模型不是「主流榜」,而是进入同一份评测卡的候选。以下是**2026-07-17 查阅**的官方文档/发布方 artifact;它们既不代表当前排名,也不构成对任何语料的推荐。

| 候选 artifact / API model | 应记录的版本信息 | 文档或 artifact（查阅日） | 试验时应确认 |
|---|---|---|---|
| `text-embedding-3-small` | API model ID、响应/SDK 版本；若服务不公开不可变 revision，显式记为「provider revision 未公开」 | [OpenAI small 模型页（2026-07-17）](https://developers.openai.com/api/docs/models/text-embedding-3-small) | [Create Embeddings API 的 `dimensions` 字段（2026-07-17）](https://developers.openai.com/api/reference/resources/embeddings/methods/create)仅由 `text-embedding-3` 及后续模型支持；记录实际请求值、归一化/距离、区域与评测日 |
| `text-embedding-3-large` | API model ID、响应/SDK 版本；若服务不公开不可变 revision，显式记为「provider revision 未公开」 | [OpenAI large 模型页（2026-07-17）](https://developers.openai.com/api/docs/models/text-embedding-3-large) | 同样按 [Create Embeddings API 的 `dimensions` 字段（2026-07-17）](https://developers.openai.com/api/reference/resources/embeddings/methods/create)记录实际请求值 |
| `embed-v4.0` | 完整 API model ID、SDK 版本 | [Cohere Embed 模型表（2026-07-17）](https://docs.cohere.com/docs/models#embed) | [Cohere Embeddings 指南（2026-07-17）](https://docs.cohere.com/docs/embeddings)中的 `input_type`、`output_dimension`、`embedding_types` 与响应 embedding 类型，必须与实际语料/检索角色一致 |
| `BAAI/bge-m3` | Hugging Face artifact 的**完整 commit SHA**、FlagEmbedding/transformers 版本、所用编码指令 | [BAAI 模型卡（2026-07-17）](https://huggingface.co/BAAI/bge-m3) | dense、sparse、multi-vector 的具体打分 recipe 与检索阶段 |

**MTEB(Massive Text Embedding Benchmark,Muennighoff et al. 2022,arXiv:2210.07316)** 的任务、数据集和 leaderboards 会随代码与数据版本变化，所以每次读取都只是一个快照，而不是「2026 榜首」的事实。引用或复跑时记录 `mteb` 包/仓库 revision、任务列表、数据集 revision 与 split、metric、运行日期；项目入口见 [MTEB 官方仓库（2026-07-17）](https://github.com/embeddings-benchmark/mteb)。最终仍以本领域标注 query—文档对的实验为准([[18 RAG 评估|RAG 评估]])。

**实验卡（每次比较必须填全）**:

| 字段 | 要记录的值 |
|---|---|
| artifact / revision | API model ID 与可见 provider revision，或开源模型的完整 commit SHA；SDK、推理引擎版本 |
| instruction 与表示 | query/document instruction 或前缀、`dense/sparse/multi-vector` recipe、维度、dtype、是否 L2 归一化 |
| 语料 | 语料版本、语言、去重规则、chunker 版本、chunk 大小/overlap、metadata/ACL 过滤 |
| 检索链路 | index 类型和完整参数、top-k、是否 hybrid、是否 rerank；若重排，另记 reranker artifact/revision 与候选数 |
| 结果 | Recall@k、nDCG@k（及标注定义）、p50/p95 延迟、吞吐、构建时间、常驻内存和磁盘占用 |
| 环境 | CPU/GPU/内存、并发、数据量、随机种子、评测日期；保存配置、日志和原始结果 |

**BGE-M3 的边界**:dense、sparse 与 multi-vector 是三种**召回表示**，可用于各自检索或融合；multi-vector 的 token 级匹配也仍是检索打分，并不是 cross-encoder 的 rerank signal。若要对召回 top-k 做精排，应明确接入独立的 [[10 重排序 Reranking|重排序 Reranking]] 模型，并在实验卡里记录它。

## 可跑最小代码

下面的零依赖代码用**中文 query 与中文语料**验算「归一化向量 + top-k」链路；它明确验证无关的「猫」被排除，且以 `汽车` 与 `轿车` 的不同特征消除并列分数。它是确定性的教学 toy vectorizer，不替代上表中已固定 artifact/revision 的真实 embedding 模型。真实实验不得把它的输出当作模型质量或 ANN 结论。

```python
from math import sqrt

docs = [
    "猫是一种常见宠物。",
    "汽车使用燃油或电力提供动力。",
    "轿车通过发动机或电动机行驶。",
]
query = "汽车用什么动力？"

# 中文特征仅用于让示例零依赖、可重复运行；生产中替换为已记录 revision 的模型 encoder。
# ❌ 让并列分数依赖稳定排序；✅ 分开「汽车」与「轿车」特征，并断言预期顺序。
FEATURES = (("猫", "宠物"), ("汽车",), ("轿车",), ("车",),
            ("燃油", "电力", "发动机", "电动机", "动力", "行驶"))

def encode_zh_toy(text):
    raw = [sum(token in text for token in feature) for feature in FEATURES]
    norm = sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]

q = encode_zh_toy(query)
scores = [sum(qj * dj for qj, dj in zip(q, encode_zh_toy(doc))) for doc in docs]
top2 = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:2]
for rank, i in enumerate(top2, start=1):
    print(rank, docs[i], round(scores[i], 3))

assert [docs[i] for i in top2] == [
    "汽车使用燃油或电力提供动力。",
    "轿车通过发动机或电动机行驶。",
]
```

生产替换时,保存上面的实验卡，再由支持该 artifact 的 encoder 生成 doc/query 向量；ANN 索引只负责近似候选搜索，不能替代质量评测。

## 何时用 / 坑

- **embedding 选型不能只看某次 MTEB 快照**:语言不对、领域漂移、长度超限都会翻车。在自己数据上按实验卡评。
- **建库与查询的模型/归一化必须一致**:换了 embedding 模型必须**全量重建索引**;query 和 doc 用不同模型或一个归一化一个没归一化 → 相似度全错。
- **相似度度量要跟模型匹配**:模型按 cosine 训的,你拿 L2 硬搜会退化。
- **HNSW 内存不能凭倍数估**:先算向量本体,再按目标实现和参数测量图、metadata、allocator 的常驻内存和磁盘;预算不够才比较 IVF-PQ、量化或分片。
- **ANN 会漏召回**:它是「近似」。nprobe/efSearch 调太小,边界文档永远召不回。召回不全先查这两个参数。
- **纯向量召回有盲区**:精确实体、ID、罕见专名、代码符号,dense 容易召不准——这正是要上 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]](dense+BM25)和 [[10 重排序 Reranking|重排序 Reranking]] 的原因。
- **向量本身是「记忆」**:同一套 embedding+ANN 也是 [[19 Agent 记忆系统|Agent 记忆系统]] 的检索底座,RAG 和 agent 长期记忆共用这层基础设施。

## 工业界实践

向量库这层是 RAG 里最「运维重」的部分,生产决策围绕三件事:**索引选型、过滤策略、量化压缩**。

**索引选型(HNSW vs IVF vs IVF-PQ)必须是受约束实验**:
- **HNSW**:控制 `M`、`efConstruction`、`efSearch`，记录其对图内存、构建时间、Recall@k/nDCG@k 和 p95 的联合影响。`M` 不应脱离实现及数据分布给出固定推荐值。
- **IVF**:控制 `nlist`、`nprobe` 和训练样本。`nprobe` 小会漏召回；是否符合预算要看 IVF-flat 的实测内存与 p95，而不是假定它一定更省。
- **IVF-PQ**:控制子空间数 $m$、码本大小 $K$、`nprobe` 和候选重打分数。量化误差与候选不足都可能损失 Recall@k/nDCG@k；保留全精度向量作重打分会改变总内存公式，必须单列。

**ANN 选型卡**（先填约束，再比较）：

| 要回答的问题 | 必须比较的配置 | 放行条件 |
|---|---|---|
| 召回是否足够？ | HNSW 的 `M/efSearch`、IVF 的 `nlist/nprobe`、IVF-PQ 的 `m/K/nprobe` | 同一 query 集的 Recall@k、nDCG@k 达标 |
| 延迟是否可接受？ | 同一并发、同一 top-k、同一 filter 与 rerank 开关 | p50/p95、吞吐和冷/热缓存条件达标 |
| 容量是否可接受？ | 原始向量、图/列表、码本、ID、metadata、可选全精度重打分副本 | 进程 RSS、磁盘和构建峰值在预算内 |
| 是否要重排？ | 记录 `rerank=false` 的召回基线；再记录 reranker artifact/revision 与候选数 | 把重排后的 nDCG 与额外 p95 分开报告 |

**量化的公式而非承诺**:
- **scalar / int8**:仅代码部分近似为 $B_{\text{codes}}=N\cdot d\cdot1$，另加每向量/每块 scale 与 zero-point、索引、metadata；若保留 float32 重打分副本，还要再加 $N\cdot d\cdot4$。对比同一实验的 Recall@k/nDCG@k、p95 与 RSS 后再决定。
- **binary**:位打包代码的下界为 $B_{\text{codes}}=N\cdot\lceil d/8\rceil$，仍要加索引和 metadata。它的损失、oversampling 数和是否需要全精度重打分只能由本语料的实验确定；不要把它归因成某个模型「天然友好」。
- **降维**:只有 artifact 的官方接口或模型卡明确支持且实验卡记录了输出维度时才启用；将不同维度作为独立配置，比较质量、延迟与总内存，不能把未验证的截断当作免费优化。

**embedding 选型实操**:以语言、领域、文本长度、数据驻留约束和服务形态筛出候选；为每个候选填同一张实验卡、跑同一标注 query 集，才比较结果。MTEB 可作外部参考快照，但不能代替本库数据的验证([[18 RAG 评估|18]])。

**踩坑速记**:
- **换 embedding 模型 = 全量重建索引**,且要灰度。建库和查询的模型、归一化必须完全一致,否则相似度全错。
- **相似度度量要跟模型训练方式匹配**:模型按 cosine 训的,拿 L2 硬搜会退化。
- **ANN 是近似,会漏召回**:召回不全先查 `efSearch`/`nprobe` 是不是调太小,而不是先怪 embedding。
- **HNSW 内存爆**:先按 $N\cdot d\cdot b$ 算向量本体，再把图边、ID、metadata、allocator 和可选重打分副本逐项测进 RSS/磁盘；不要沿用固定图开销倍数。
- **dense 有盲区**:精确实体、ID、罕见专名、代码符号 dense 召不准,这正是要叠 [[08 混合检索 Hybrid Search|混合检索]](dense+BM25)+ [[10 重排序 Reranking|重排]] 的原因。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1:RAG 检索为什么用 bi-encoder 而不是 cross-encoder?**
标准答:bi-encoder(双塔)让 query/doc **各自独立编码**,doc 向量可**离线预建库**,查询时只编码 query 一次再做 ANN,能 scale 到亿级。cross-encoder 把 (query,doc) 拼一起算,精度高得多但**无法预建库**(每对都现算),只能用在召回后对 top-k 精排([[10 重排序 Reranking|重排]])。标准两段式 = **双塔召回 + 交叉精排**。

**Q2:讲讲三大 ANN 索引及其取舍。**
标准答:**HNSW**用多层近邻图，`M/efConstruction/efSearch` 共同影响图内存、构建成本、Recall@k 与 p95；**IVF**先聚类、只搜 `nprobe` 个列表，需评估 `nlist/nprobe` 与训练样本；**PQ**把每段替换成码字，有损且要加上码本/ID/可选重打分副本。常见组合是 IVF-PQ，但不能由名称推断固定规模、压缩倍数或召回。三者应用同一 query 集和预算，以 Recall@k/nDCG@k、p95、RSS/磁盘作决策。HNSW 出处 Malkov & Yashunin 2016。

**Q3:cosine、dot、L2 怎么选?**
标准答:**cosine 最常用**(看方向夹角,对模长不敏感);向量 L2 归一化后 **dot = cosine**(很多库默认归一化后用点积,快);L2 在归一化向量下与 cosine 单调等价。关键:**度量要跟 embedding 模型怎么训的匹配,别乱换**。

**Q4:HNSW 为什么快?efSearch 是什么?**
标准答:HNSW 建多层图,上层稀疏像「高速公路」、底层 L0 全量稠密。查询从顶层进入,贪心跳到当前层离 q 最近的点再降一层,逐层逼近,L0 精搜,复杂度近 O(log N)。`efSearch` 控制查询时候选队列大小:越大召回越高、越慢,在线可调,是召回/延迟的旋钮。

**Q5(陷阱):换了更强的 embedding 模型,只把新文档用新模型编码加进去就行了吗?**
标准答:**不行,必须全量重建索引**。旧文档是老模型的向量、新 query 是新模型的向量,两者在不同语义空间,相似度计算无意义。这题专抓「以为 embedding 可增量混用」的人。

**Q6:为什么纯向量检索还不够,要上混合检索?**
标准答:dense embedding 在精确实体、ID、罕见专名、代码符号上区分度低(语义泛化反而成了缺点),容易召不准;BM25 这类稀疏检索擅长精确字面匹配。两者互补,融合(如 RRF)能同时覆盖语义相似和精确匹配,见 [[08 混合检索 Hybrid Search|08]]。

## 知识拓展

- **多向量 / 后期交互(ColBERT)**:bi-encoder 把整段压成一个向量必然损失细粒度;**ColBERT**(Khattab & Zaharia 2020)给每个 token 留一个向量,检索时做 token 级 MaxSim「后期交互」,精度逼近 cross-encoder 而仍可建库。bge-m3 的 multi-vector 表示即此思路。这是「召回精度」和「可 scale」之间的折中前沿。
- **bge-m3 的三合一**:一个模型可产 dense + sparse + multi-vector 三种**召回表示**(arXiv:2402.03216)，可分别检索或用于 [[08 混合检索 Hybrid Search|混合检索]] 融合；它们不等同于 reranker，精排仍需单独的 cross-encoder 或其他重排模型。
- **数学根基**:整个「语义相近=向量相近」建立在内积/余弦几何上,严格定义见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]];embedding 模型本质是把 [[LLM/054 词嵌入层与权重绑定|词嵌入]] 的思想从词级提升到句/段级(对比学习拉近正例、推远负例)。
- **检索基础设施被复用**:同一套 embedding + ANN 索引也是 [[19 Agent 记忆系统|Agent 记忆系统]] 的检索底座——RAG 和 Agent 长期记忆共用这层。
- **前沿与边界**:① **ScaNN**(Google)是另一条 ANN 路线，是否适合也应纳入同一实验卡。② MTEB(Muennighoff et al. 2022,arXiv:2210.07316)及其 leaderboard 只是一版评测快照，务必自建数据小评测。③ 反模式:盲信排行榜、跨语言/领域硬套、忽视长度上限、换模型不重建索引、未记录 ANN 参数就判断召回差。

## 关键事实

- RAG 检索用 **bi-encoder(双塔)**:query/doc 独立编码 → doc 可离线建库 → 能 scale;高精度的 cross-encoder 只用于 [[10 重排序 Reranking|重排序 Reranking]]。
- 相似度:**cosine 最常用**;归一化后 cosine = dot;度量要跟模型训练方式匹配。
- ANN 三族:**HNSW(图)**、**IVF(倒排分桶)**、**PQ(量化编码)**；以参数、数据和实现实测 Recall@k/nDCG@k、p95、RSS/磁盘，常见组合为 IVF-PQ。
- HNSW 出处:**Malkov & Yashunin 2016, arXiv:1603.09320**。
- 向量库定位:FAISS=算法库;pgvector=Postgres 就地;Qdrant/Weaviate=生产中规模;Milvus=超大规模;Pinecone=全托管。
- 候选必须以可追溯 artifact/revision、instruction、维度/量化和官方文档记录；[text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small)、[text-embedding-3-large](https://developers.openai.com/api/docs/models/text-embedding-3-large)、[Create Embeddings API 的 `dimensions`](https://developers.openai.com/api/reference/resources/embeddings/methods/create)、[Cohere Embed 模型表](https://docs.cohere.com/docs/models#embed)、[Cohere Embeddings 指南](https://docs.cohere.com/docs/embeddings)与 `BAAI/bge-m3` 都应在同一实验卡中比较，**不做市场排序**。
- **MTEB(Muennighoff et al. 2022, arXiv:2210.07316)** leaderboard 是随任务/数据/代码版本变化的快照；需在自己的语料、切分、top-k、重排开关和硬件上评([[18 RAG 评估|RAG 评估]])。
- 换 embedding 模型必须**全量重建索引**;dense 有盲区,生产要叠 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] + 重排。

## 来源

- Malkov & Yashunin 2016.《Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs》arXiv:1603.09320。
- Muennighoff et al. 2022.《MTEB: Massive Text Embedding Benchmark》arXiv:2210.07316。
- Chen, Xiao et al. 2024.《M3-Embedding (BGE-M3): Multi-Linguality, Multi-Functionality, Multi-Granularity ... 》arXiv:2402.03216。
- Reimers & Gurevych.《Sentence-BERT》(sentence-transformers)。
- Khattab, O. & Zaharia, M. 2020.《ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT》SIGIR 2020, arXiv:2004.12832。token 级多向量 + MaxSim 后期交互。
- [OpenAI `text-embedding-3-small` 模型页（2026-07-17 查阅）](https://developers.openai.com/api/docs/models/text-embedding-3-small)。记录 API model ID；若服务未公开不可变 revision，实验卡必须如实标注。
- [OpenAI `text-embedding-3-large` 模型页（2026-07-17 查阅）](https://developers.openai.com/api/docs/models/text-embedding-3-large)。记录 API model ID；若服务未公开不可变 revision，实验卡必须如实标注。
- [OpenAI Create Embeddings API 参考：`dimensions`（2026-07-17 查阅）](https://developers.openai.com/api/reference/resources/embeddings/methods/create)。该可选请求字段仅支持 `text-embedding-3` 及后续模型。
- [Cohere Embed 模型表（2026-07-17 查阅）](https://docs.cohere.com/docs/models#embed)。用于核对 `embed-v4.0` 的模型条目与其列出的输出维度。
- [Cohere Embeddings 指南（2026-07-17 查阅）](https://docs.cohere.com/docs/embeddings)。用于核对检索 query/document 的 `input_type`，以及 `output_dimension`、`embedding_types` 与响应 embedding 类型的记录方式。
- [BAAI `bge-m3` 模型卡（2026-07-17 查阅）](https://huggingface.co/BAAI/bge-m3)。复现实验须 pin 到完整 commit SHA。
- [MTEB 官方仓库（2026-07-17 查阅）](https://github.com/embeddings-benchmark/mteb)。运行时记录包/仓库 revision、任务与数据集版本。
