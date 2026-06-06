[[04 Embedding 与向量数据库|Embedding 与向量数据库]] 是 RAG 检索的引擎室:**embedding** 把文本压成一串浮点向量,语义相近的文本在向量空间里靠得近;**向量数据库**则用 ANN 索引在亿级向量里毫秒级找出与 query 最近的几个。前者决定「编码得准不准」,后者决定「找得快不快」。它直接承接 [[03 分块策略 Chunking|分块策略 Chunking]] 切出来的块,是 [[01 什么是 RAG|什么是 RAG]] 里 query→doc 匹配的物理实现。

## 本质:语义检索 = 把「找相似含义」变成「找近向量」

关键词检索(BM25)只会匹配字面;但「汽车」和「轿车」、"car" 和 "automobile" 字面不同、含义相同。**dense embedding** 的核心是:把文本映射到一个高维向量空间(常 384–1536 维),让训练目标逼着**语义相近→向量相近**。于是「找语义相关文档」这个模糊问题,被还原成一个干脆的几何问题:**在向量空间里找离 query 向量最近的点**。

这就引出两个必须分别解决的子问题,本篇两半各管一个:
1. **怎么把文本编码成好向量?** → embedding 模型(bi-encoder)。
2. **怎么在海量向量里快速找最近邻?** → 向量数据库的 ANN 索引。

## 机制 A:bi-encoder 双塔与相似度

### bi-encoder(双塔)
RAG 检索用的是 **bi-encoder**:query 和 doc **各自独立**过同一个编码器,各得一个向量,再算相似度。关键好处——**doc 向量可以离线预先算好建库**,查询时只需编码 query 一次,然后做向量最近邻。这让它能 scale 到亿级。

对照 **cross-encoder**:把 (query, doc) 拼一起喂进模型输出一个相关度分,精度高得多但**无法预建库**(每个 query-doc 对都要现算),只能用在召回后的 [[10 重排序 Reranking|重排序 Reranking]] 阶段对 top-k 精排。**双塔召回 + 交叉精排** 是标准两段式。

`sentence-transformers`(SBERT)是 bi-encoder 的代表实现:在 BERT 上做对比学习,让正例对(相关句)向量拉近、负例推远,输出句向量。

![[Embedding 与向量数据库.svg]]

### 相似度度量
- **cosine 余弦**:看方向夹角,忽略模长。**最常用**,对向量缩放不敏感。
- **dot product 点积**:方向 + 模长都算。若向量已 L2 归一化,点积 = cosine。许多库默认要求归一化后用点积(快)。
- **欧氏距离 L2**:几何直线距离。归一化向量下与 cosine 单调等价。
实践:**大多数 embedding 模型设计为配 cosine**;用哪个取决于模型怎么训的,别乱换。

## 机制 B:ANN 近似最近邻索引

精确最近邻要拿 query 和**每个** doc 算相似度,O(N)——亿级库下慢到不可用。**ANN(Approximate Nearest Neighbor)** 牺牲一点点召回(可能漏掉个别真最近邻)换数量级的提速。三大族系:

### HNSW(图,默认主力)
**Hierarchical Navigable Small World**,源自 **Malkov & Yashunin 2016(arXiv:1603.09320)**。构建一个**多层近邻图**:底层 L0 含全量节点、连边稠密;往上每层节点指数减少、像「高速公路」。查询时从顶层稀疏图进入,**贪心跳到当前层离 q 最近的点,再降一层**,逐层逼近,最后在 L0 精搜。复杂度近 **O(log N)**。优点:召回高、查询快;缺点:**内存占用大**(要存图的边)、建索引慢、增删麻烦。绝大多数向量库的默认索引。

### IVF(倒排,省内存)
**Inverted File**:先把所有向量 **k-means 聚类**成若干桶(质心)。查询时只算 q 到各质心的距离,**只搜最近的 nprobe 个桶**,跳过其余。省内存、建索引快;但 nprobe 调小会**漏召回**(真最近邻落在没搜的桶里),边界样本尤甚。

### PQ(乘积量化,压内存)
**Product Quantization**:把每个向量切成 m 段,每段用一个小**码本**(k-means 得到的码字集合)近似,只存「每段最近码字的 id」。一个 768 维 float 向量可从 3KB 压到几十字节,**内存压几十倍**;代价是量化有损、精度下降。常与 IVF 组合成 **IVF-PQ**,用于十亿级、内存极度受限场景。

此外 **ScaNN**(Google)用各向异性量化 + 部分重排,在精度/速度上很强,是另一条工业级路线。

![[向量索引-ANN.svg]]

## 向量库谱(各自定位)

| 库 | 形态 | 定位 / 一句话 |
|---|---|---|
| **FAISS** | 库(Meta) | 不是数据库,是 ANN 算法库。要自己管持久化/过滤,做原型/嵌入式首选 |
| **pgvector** | Postgres 扩展 | 数据已在 Postgres 就地加向量列,**最省运维**,中小规模够用 |
| **Chroma** | 轻量嵌入式 | 开发友好、几行起步,本地原型 / 小项目 |
| **Qdrant** | 独立向量库(Rust) | 性能强、**payload 过滤**好,云原生,生产常选 |
| **Weaviate** | 独立(含 GraphQL) | 自带模块化向量化、混合检索、schema,功能全 |
| **Milvus** | 独立(分布式) | **超大规模**、多索引(IVF/HNSW/PQ)、分布式,十亿级首选 |
| **Pinecone** | 全托管 SaaS | 不想运维就交给它,按用量付费,上手快 |

**选型直觉**:数据在 Postgres → pgvector;本地原型 → Chroma / FAISS;生产中等规模要过滤 → Qdrant / Weaviate;超大规模 → Milvus;全托管省心 → Pinecone。

## 2026 主流 embedding 模型 + MTEB

- **OpenAI text-embedding-3**(small/large):API 易用,large 维度可裁剪(Matryoshka),通用强。
- **Cohere embed v3**:含多语言版,支持检索/分类等不同用途的 input type,生产口碑好。
- **bge-m3**(BAAI):**Multi-Lingual / Multi-Functionality / Multi-Granularity**——一个模型同时给 dense / sparse / multi-vector(ColBERT 式)三种表示,100+ 语言,长达 8192 token(**Chen et al. 2024, arXiv:2402.03216**)。做 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 特别顺手。
- **E5**(Microsoft):对比学习开源系列,需加 `query:`/`passage:` 前缀,性价比高。
- **Voyage**:专做检索 embedding,含领域版(code/finance/law),质量靠前。
- **Nomic Embed**:开源、可本地跑、长上下文,数据可控场景友好。

**MTEB(Massive Text Embedding Benchmark,Muennighoff et al. 2022,arXiv:2210.07316)** 是选型的事实标准:涵盖 8 类任务、58 数据集、112 种语言,公开 leaderboard。**别只看排行榜分数**——MTEB 自己的结论就是「没有一个模型在所有任务上通吃」;要按你的语言、领域、是否长文档、延迟/成本来挑,最好在自己的数据上小评测一轮([[18 RAG 评估|RAG 评估]])。

## 可跑最小代码

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")   # bi-encoder
docs  = ["猫是一种宠物。", "汽车需要燃料。", "轿车靠引擎驱动。"]

# 1) 离线:把 doc 编码成向量并归一化(配 cosine→用内积索引)
emb = model.encode(docs, normalize_embeddings=True).astype("float32")
index = faiss.IndexHNSWFlat(emb.shape[1], 32)           # HNSW,M=32
index.add(emb)                                          # 建库(doc 向量预存)

# 2) 在线:只编码 query,做 ANN 最近邻
q = model.encode(["车用什么动力?"], normalize_embeddings=True).astype("float32")
D, I = index.search(q, k=2)                             # 近似最近邻 top-2
for rank, i in enumerate(I[0]):
    print(rank, docs[i], float(D[0][rank]))             # 命中「轿车靠引擎」「汽车需要燃料」
# 真用别手搓索引:Qdrant / Milvus / pgvector 管持久化、过滤、增删
```

## 何时用 / 坑

- **embedding 选型不能只看 MTEB 总分**:语言不对、领域漂移、长度超限都会翻车。在自己数据上评。
- **建库与查询的模型/归一化必须一致**:换了 embedding 模型必须**全量重建索引**;query 和 doc 用不同模型或一个归一化一个没归一化 → 相似度全错。
- **相似度度量要跟模型匹配**:模型按 cosine 训的,你拿 L2 硬搜会退化。
- **HNSW 内存爆**:亿级库 HNSW 吃内存极猛,先估容量;扛不住转 IVF-PQ 或分片(Milvus)。
- **ANN 会漏召回**:它是「近似」。nprobe/efSearch 调太小,边界文档永远召不回。召回不全先查这两个参数。
- **纯向量召回有盲区**:精确实体、ID、罕见专名、代码符号,dense 容易召不准——这正是要上 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]](dense+BM25)和 [[10 重排序 Reranking|重排序 Reranking]] 的原因。
- **向量本身是「记忆」**:同一套 embedding+ANN 也是 [[19 Agent 记忆系统|Agent 记忆系统]] 的检索底座,RAG 和 agent 长期记忆共用这层基础设施。

## 工业界实践

向量库这层是 RAG 里最「运维重」的部分,生产决策围绕三件事:**索引选型、过滤策略、量化压缩**。

**索引选型(HNSW vs IVF vs IVF-PQ)的真实取舍**:
- **HNSW**:绝大多数库的默认,召回高、查询快(O(log N)),但**内存吃得猛**。两个核心参数:`M`(每节点边数,常 16–64,越大越准越占内存)、`efConstruction`(建索引精度)/ `efSearch`(查询精度,在线可调,越大召回越高越慢)。中小规模(千万级以内)闭眼选 HNSW。
- **IVF**:省内存、建索引快,靠 `nlist`(聚类桶数)和 `nprobe`(查询搜几个桶)调召回。`nprobe` 调小会漏召回(真最近邻落在没搜的桶)。适合内存受限、能接受微调召回的场景。
- **IVF-PQ**:十亿级、内存极度受限时的选择。PQ 把向量量化压几十倍,有损,精度降,通常配重排或重排序兜底。

**量化(生产降本主力)**:
- **Scalar / int8 量化**:float32→int8,内存降 4 倍,精度损失小,几乎免费的优化。Qdrant/Milvus 一行配置开。
- **二值量化(binary)**:1 bit/维,内存降 32 倍,配 oversampling + 重排可保住召回,OpenAI v3 / Cohere v3 等模型对二值化友好。
- **Matryoshka 维度裁剪**:OpenAI v3-large(3072 维)、bge-m3 等支持按需截断维度(如取前 256/512 维),用精度换存储/速度,无需重训。

**生产配置片段**(Qdrant 示例,展示量化 + 过滤):

```python
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://qdrant:6333")
client.create_collection(
    collection_name="kb",
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    # int8 标量量化:内存降 4 倍,查询时按需从原始向量重打分(rescore)保召回
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(type="int8", always_ram=True)),
)
# 检索:元数据过滤先行(acl/tenant/时间),再向量搜 → 又快又对又安全
hits = client.query_points(
    "kb", query=qvec, limit=50,
    query_filter=models.Filter(must=[
        models.FieldCondition(key="tenant", match=models.MatchValue(value="acme")),
        models.FieldCondition(key="acl", match=models.MatchAny(any=user_groups)),
    ]),
)  # payload 过滤是 Qdrant/Weaviate 的强项,见 16 检索安全与访问控制
```

**embedding 选型实操**:别只信 MTEB 总分(MTEB 自己的结论就是「没有通吃模型」)。流程是——按语言(中文优先 bge/Cohere multilingual)、领域(代码/法律/金融有专用 Voyage 版)、长度(长文档要 8192 上下文如 bge-m3)、延迟成本(本地可控选开源 bge/E5/Nomic,省心选 OpenAI/Cohere API)圈出 3–5 个候选,**在自己数据上跑小评测**([[18 RAG 评估|18]])定夺。

**踩坑速记**:
- **换 embedding 模型 = 全量重建索引**,且要灰度。建库和查询的模型、归一化必须完全一致,否则相似度全错。
- **相似度度量要跟模型训练方式匹配**:模型按 cosine 训的,拿 L2 硬搜会退化。
- **ANN 是近似,会漏召回**:召回不全先查 `efSearch`/`nprobe` 是不是调太小,而不是先怪 embedding。
- **HNSW 内存爆**:亿级库先估容量(向量数 × 维度 × 4B × ~1.5 图开销)。扛不住转 IVF-PQ 或分片(Milvus)。
- **dense 有盲区**:精确实体、ID、罕见专名、代码符号 dense 召不准,这正是要叠 [[08 混合检索 Hybrid Search|混合检索]](dense+BM25)+ [[10 重排序 Reranking|重排]] 的原因。

## 面试高频

**Q1:RAG 检索为什么用 bi-encoder 而不是 cross-encoder?**
标准答:bi-encoder(双塔)让 query/doc **各自独立编码**,doc 向量可**离线预建库**,查询时只编码 query 一次再做 ANN,能 scale 到亿级。cross-encoder 把 (query,doc) 拼一起算,精度高得多但**无法预建库**(每对都现算),只能用在召回后对 top-k 精排([[10 重排序 Reranking|重排]])。标准两段式 = **双塔召回 + 交叉精排**。

**Q2:讲讲三大 ANN 索引及其取舍。**
标准答:**HNSW**(多层近邻图,贪心逐层逼近,O(log N),召回高查询快,但内存大、增删麻烦,默认主力)/ **IVF**(k-means 聚类成桶,只搜最近 nprobe 个桶,省内存,nprobe 小漏召回)/ **PQ**(乘积量化,每段用码本近似,内存压几十倍,有损)。常组合 **IVF-PQ** 用于十亿级内存受限场景。HNSW 出处 Malkov & Yashunin 2016。

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
- **bge-m3 的三合一**:一个模型同时产 dense + sparse + multi-vector 三种表示(arXiv:2402.03216),做 [[08 混合检索 Hybrid Search|混合检索]] 特别顺手——一次编码拿齐稠密召回、稀疏召回、精排所需的全部信号。
- **数学根基**:整个「语义相近=向量相近」建立在内积/余弦几何上,严格定义见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]];embedding 模型本质是把 [[LLM/054 词嵌入层与权重绑定|词嵌入]] 的思想从词级提升到句/段级(对比学习拉近正例、推远负例)。
- **检索基础设施被复用**:同一套 embedding + ANN 索引也是 [[19 Agent 记忆系统|Agent 记忆系统]] 的检索底座——RAG 和 Agent 长期记忆共用这层。
- **前沿与边界**:① **ScaNN**(Google)用各向异性量化 + 部分重排,精度/速度俱佳的另一条工业路线。② MTEB(Muennighoff et al. 2022,arXiv:2210.07316)是选型事实标准但**没有通吃模型**,务必自建数据小评测。③ 反模式:盲信排行榜分数、跨语言/领域硬套、忽视长度上限、换模型不重建索引、ANN 参数默认不调就怪召回差。

## 关键事实

- RAG 检索用 **bi-encoder(双塔)**:query/doc 独立编码 → doc 可离线建库 → 能 scale;高精度的 cross-encoder 只用于 [[10 重排序 Reranking|重排序 Reranking]]。
- 相似度:**cosine 最常用**;归一化后 cosine = dot;度量要跟模型训练方式匹配。
- ANN 三族:**HNSW(图,默认,O(log N),内存大)**、**IVF(倒排,省内存,nprobe 小漏召回)**、**PQ(量化压内存,有损)**,常组合 IVF-PQ。
- HNSW 出处:**Malkov & Yashunin 2016, arXiv:1603.09320**。
- 向量库定位:FAISS=算法库;pgvector=Postgres 就地;Qdrant/Weaviate=生产中规模;Milvus=超大规模;Pinecone=全托管。
- 2026 模型:OpenAI v3 / Cohere v3 / **bge-m3(dense+sparse+多向量三合一, arXiv:2402.03216)** / E5 / Voyage / Nomic。
- 选型基准:**MTEB(Muennighoff et al. 2022, arXiv:2210.07316)**,但没有通吃模型,需在自己数据上评([[18 RAG 评估|RAG 评估]])。
- 换 embedding 模型必须**全量重建索引**;dense 有盲区,生产要叠 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] + 重排。

## 来源

- Malkov & Yashunin 2016.《Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs》arXiv:1603.09320。
- Muennighoff et al. 2022.《MTEB: Massive Text Embedding Benchmark》arXiv:2210.07316。
- Chen, Xiao et al. 2024.《M3-Embedding (BGE-M3): Multi-Linguality, Multi-Functionality, Multi-Granularity ... 》arXiv:2402.03216。
- Reimers & Gurevych.《Sentence-BERT》(sentence-transformers)。
- Khattab, O. & Zaharia, M. 2020.《ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT》SIGIR 2020, arXiv:2004.12832。token 级多向量 + MaxSim 后期交互。
