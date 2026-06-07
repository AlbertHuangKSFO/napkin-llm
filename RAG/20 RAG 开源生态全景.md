[[20 RAG 开源生态全景|RAG 开源生态全景]]不是一份"框架排行榜",而是一张**按职责分层的地图**:一个 [[01 什么是 RAG|RAG]] 管线 = 文档解析 → 索引/Embedding → 向量库 → 检索 → 重排 → 生成 → 评估,外加一层**横跨全栈的框架/编排**把它们黏起来。地图的价值在于:告诉你某个需求该去哪层找轮子、同层有哪些互斥竞品、**owner 又改名搬到哪了**(这事在 RAG 圈一样频繁)。它与 [[13 Modular RAG|Modular RAG]] 是一体两面——模块化让每层可插拔,生态就是每层的候选池。

## 本质:层稳定,库流动
和 [[39 Agent 开源生态全景|Agent 开源生态全景]] 同一个心智:**碎**(一个 RAG 被拆成七八层正交职责,每层三五个竞品)且**快**(库每周一版,org 还改名搬家)。所以"罗列一百个库"三个月就过期;有用的是把库**摁进固定的层**——层是稳定的(职责不变),库是流动的(谁火换谁)。记住"文档解析这层 2026 主流是 Docling/unstructured",比记某个版本号有用得多。

读图三句话:**同层多个库 ≈ 互斥竞品**(选一个就够);**跨层多个库 ≈ 互补可叠**;**框架/编排横跨各层**把它们串成管道。

![[RAG 开源生态全景.png]]

## 分层拆解(逐一核验 owner / 2026 活跃度)

### ① 框架 / 编排——把各层串成管道(横跨全栈)
**定位**:提供 RAG 的高层抽象(loader→splitter→index→retriever→synthesizer 一条龙),让你不必手搓每层。锁定最深、选错重写成本最高。
- **`run-llama/llama_index`**(`llama-index`,核心 `llama-index-core`)——RAG 出身的事实标杆,数据连接器/索引/查询引擎最全,如今 Workflows 1.0 是事件驱动编排。文档密集、要 [[36 Agentic RAG|Agentic RAG]] 时首选。
- **`langchain-ai/langchain` + `langchain-ai/langgraph`**——LangChain 是组件胶水(retriever/向量库/LLM 适配最广),LangGraph 把流程建成有状态图;要可控分支、Human-in-the-Loop、[[12 Self-RAG、CRAG 与 Adaptive RAG|自适应路由]]时上 LangGraph,2026 已到 1.x。
- **`deepset-ai/haystack`**(`haystack-ai`)——管线式(Pipeline + Component),生产派偏爱其显式 DAG 与可观测性,deepset 商业支撑。
- **`infiniflow/ragflow`**(`ragflow`)——**深度文档理解**起家的开箱即用 RAG 引擎(自带解析+切块+检索+UI),2026 融合 Agent 能力、加 Browser 组件、RAPTOR 引入数据集级 AHC(Ψ-RAG)。要"装上就能用"的一体机时选它。
- **`neuml/txtai`**(`txtai`)——embeddings-first 的语义搜索框架,多模态(文本/音频/视频)、轻量 RAG 管线一把梭,单文件可跑。
- **`weaviate/Verba`**(无独立 pip,应用形态)——Weaviate 官方出品的开箱即用 RAG 聊天 UI,社区驱动,适合快速起一个带界面的 demo。
- **`truefoundry/cognita`**——模块化**生产级** RAG 框架,带 UI、增量索引、企业合规,TrueFoundry 出品,主打"从原型到生产"。
- **`IntelLabs/fastRAG`**(`fastrag`)——Intel Labs 的**高效** RAG 研究框架,主攻在 Intel 硬件上做优化检索+生成管线,建在 Haystack 之上。
- **`stanfordnlp/dspy`**(`dspy`)——不是检索器,而是把 RAG/检索管线**当可优化程序**:你写 signature,DSPy 自动编译出 prompt/权重(见 [[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]])。要让 RAG 各模块"自动调参"而非手写 prompt 时上它。

**这一层选型逻辑**:文档密集/Agentic → LlamaIndex;要可控图编排 → LangGraph;要显式管线+生产可观测 → Haystack;要开箱一体机 → RAGFlow;要自动优化 → DSPy。

### ② 文档解析(parsing/loading)——脏文档变干净文本
**定位**:把 PDF/网页/扫描件/表格变成 LLM 能吃的结构化文本,**这层质量直接决定下游一切**(解析烂 = 分块烂 = 检索烂)。
- **`Unstructured-IO/unstructured`**(`unstructured`)——老牌万能预处理库,支持几十种格式,基础场景广覆盖;复杂表格抽取精度近年被 Docling 超过。
- **LlamaParse**(LlamaIndex 旗下托管服务,`llama-cloud-services`)——擅长复杂版式/表格的托管解析,准但收费。
- **`docling-project/docling`**(`docling`,**原 `DS4SD/docling`,IBM Research 出品,已捐给 Linux Foundation / LF AI & Data**)——2026 文档解析黑马,Granite-Docling 模型表格抽取约 97.9% 精度,3.7 万+ star,推 agentic AI 方向。复杂表格/版式优先它。

### ③ Embedding / Rerank——把文本变向量、把候选排序
**定位**:索引侧的语义引擎,详见 [[04 Embedding 与向量数据库|Embedding 与向量数据库]] 与 [[10 重排序 Reranking|重排序 Reranking]]。
- **`UKPLab/sentence-transformers`**(`sentence-transformers`,**org 已迁到 `huggingface/sentence-transformers`,旧址 301 跳转,pip 名不变**)——本地跑 embedding/rerank 的事实标准库,封装 SBERT/BGE/cross-encoder。
- **`FlagOpen/FlagEmbedding`**(`FlagEmbedding`,北京智源 BAAI 出品)——**bge** 系列(bge-m3 多语言/多粒度、bge-reranker)的官方仓库,开源 embedding/rerank 第一梯队。
- **Cohere**(闭源 API)——Embed v3 与 **Rerank** 托管服务,Rerank 是工业界重排的常用即插即用选项。

### ④ 向量库(vector store)——存向量、做相似度检索
**定位**:non-parametric 记忆的物理载体,选型详见 [[04 Embedding 与向量数据库|Embedding 与向量数据库]]。
- **`facebookresearch/faiss`**(`faiss-cpu`/`faiss-gpu`)——Meta 出品的 ANN 算法库(非数据库),嵌入式、极快,做原型/库内检索的底座。
- **`milvus-io/milvus`**(`pymilvus`)——开源向量库里 star 最多,十亿级分布式,Zilliz 背书。
- **`qdrant/qdrant`**(`qdrant-client`)——Rust 写的向量库,过滤检索性能一流,可自托管/云。
- **`weaviate/weaviate`**(`weaviate-client`)——对象+向量一体,原生混合检索与模块化,见 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]]。
- **`chroma-core/chroma`**(`chromadb`)——最易上手,原型/小规模生产首选,与 LangChain/LlamaIndex 深度集成。
- **`pgvector/pgvector`**(Postgres 扩展)——给 Postgres 加向量列,**复用现有 PG 运维**;HNSW 索引后 100 万级可比肩专用库。
- **Pinecone**(闭源托管)——全托管、自动扩缩、亚 100ms 延迟,免运维的生产首选(收费)。

### ⑤ 评估(evaluation)——贯穿全链给系统打分
**定位**:详见 [[18 RAG 评估|RAG 评估]];这层不进管线但贯穿全链,改了任一层都靠它度量升降。
- **`explodinggradients/ragas`**(`ragas`)——RAG 评估事实标准,reference-free 指标 + 合成测试集。
- **`truera/trulens`**(`trulens`)——RAG Triad 三角 + 反馈函数;TruEra 2024-05 被 Snowflake 收购。
- **`confident-ai/deepeval`**(`deepeval`)——pytest 原生、最适合塞进 CI,指标库最广。
- **`stanford-futuredata/ARES`**(`ares-ai`)——微调轻量裁判 + PPI 纠偏(Saad-Falcon et al. 2023)。
- **`Arize-ai/phoenix`**(`arize-phoenix`)——OpenTelemetry 追踪+评估,偏线上可观测,与 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 同源。

## 可跑最小代码
```python
# 一条最小 RAG 管线"贴标签"地展示各层用了谁
# ① 解析   → docling / unstructured
from docling.document_converter import DocumentConverter
text = DocumentConverter().convert("doc.pdf").document.export_to_markdown()

# ② Embedding → sentence-transformers(底层可换 bge)
from sentence_transformers import SentenceTransformer
embed = SentenceTransformer("BAAI/bge-small-en-v1.5")     # FlagOpen 的 bge

# ③ 向量库 → chroma(原型);生产可换 milvus/qdrant/pgvector
import chromadb
col = chromadb.Client().create_collection("docs")
col.add(documents=[text], embeddings=[embed.encode(text).tolist()], ids=["d1"])

# ④ 检索 → 向量库自带;⑤ 重排 → cross-encoder/cohere(此处略)
hits = col.query(query_embeddings=[embed.encode("问题").tolist()], n_results=3)

# ⑥ 生成 → 任意 LLM;⑦ 评估 → ragas/deepeval(离线批量跑)
# 框架层(llama-index/langchain)会把 ①~⑥ 这串胶水替你写好
```

## 对比表(每层一句话定位 + 当前 owner)
| 层 | 代表库 · owner | 一句话 |
|---|---|---|
| 框架/编排 | LlamaIndex(`run-llama`)/LangGraph(`langchain-ai`)/Haystack(`deepset-ai`)/RAGFlow(`infiniflow`)/DSPy(`stanfordnlp`) | 串起全栈;选错重写最贵 |
| 文档解析 | unstructured(`Unstructured-IO`)/LlamaParse/Docling(`docling-project`,IBM→LF) | 决定下游上限 |
| Embedding/Rerank | sentence-transformers(`huggingface`,原 UKPLab)/bge(`FlagOpen`)/Cohere | 语义引擎 |
| 向量库 | FAISS(`facebookresearch`)/Milvus(`milvus-io`)/Qdrant/Weaviate/Chroma/pgvector/Pinecone | non-parametric 记忆载体 |
| 评估 | Ragas/TruLens(`truera`)/DeepEval(`confident-ai`)/ARES(`stanford-futuredata`)/Phoenix(`Arize-ai`) | 贯穿全链打分 |

## 何时用 / 坑
- **先选框架还是先攒组件?** 要快出原型用框架(LlamaIndex/LangChain 全包);要极致可控就直接拼 retriever+向量库+LLM,别让框架黑盒挡视线。
- **owner 会搬家**:`sentence-transformers` 已从 UKPLab 迁到 huggingface、`docling` 从 DS4SD 迁到 docling-project 并入 LF;照旧 URL star 可能 301。认 **org 现名**别认旧链接。
- **托管 vs 自托管**:Pinecone/LlamaParse/Cohere/Phoenix Cloud 省运维但收费且锁定;Milvus/Qdrant/Chroma/Ragas 自托管可控但要自己背运维。
- **别每层都装满**:一个真实 RAG 通常横跨 3~5 层(如 Docling + bge + Qdrant + Cohere Rerank + LlamaIndex + Ragas),不是把某层所有库都用上。
- **评估这层最常被跳过却最该先建**:没有 [[18 RAG 评估|RAG 评估]] 当标尺,换库换参全凭感觉。
- **与 Agent 生态重叠**:LangGraph/LlamaIndex/DSPy/Phoenix 同时出现在 [[39 Agent 开源生态全景|Agent 开源生态全景]]——RAG 进化到 [[36 Agentic RAG|Agentic RAG]] 时,两张生态图在编排/评估层合流。

## 关键事实
- **Docling** 原 `DS4SD/docling`(IBM Research),现 `docling-project/docling`,已捐给 Linux Foundation(LF AI & Data),2026 表格抽取约 97.9%、3.7 万+ star。
- **sentence-transformers** org 从 `UKPLab` 迁到 `huggingface`(旧址 301,pip 名 `sentence-transformers` 不变)。
- **bge** 系列出自 `FlagOpen/FlagEmbedding`(BAAI 智源),非个人项目。
- **TruLens** 母公司 TruEra 2024-05 被 Snowflake 收购;**Phoenix** 属 `Arize-ai`,基于 OpenTelemetry。
- **Milvus** 由 Zilliz 背书;**FAISS** 是算法库不是数据库,常作其他库的底座。

## 工业界实践

生态图的工业价值在于:**真实 RAG 横跨 3~5 层**,你要会按需求在每层挑一个、把它们黏起来,而不是把某层装满。

**1)一套典型生产选型(参考组合)**
```
解析   Docling(复杂表格/版式) 或 unstructured(格式广)
Embed  bge-m3 / sentence-transformers(自托管) 或 OpenAI/Cohere(托管)
向量库  Qdrant / Milvus(规模化) | pgvector(复用 PG 运维) | Pinecone(免运维托管)
重排   Cohere Rerank v3 或 bge-reranker(自托管 cross-encoder)
编排   LlamaIndex(文档密集/Agentic) 或 LangGraph(可控图编排) 或 Haystack(生产显式 DAG)
评估   Ragas + DeepEval(进 CI) | Phoenix/Langfuse(线上追踪)
```
不是把每层所有库都用上;一个真实管线挑一条路径即可。

**2)规模化:召回质量 × 延迟 × 成本的三角**
- **召回**:单路向量召回不够,叠 [[08 混合检索 Hybrid Search|混合检索]](BM25 稀疏 + 稠密)兜底关键词,再 [[10 重排序 Reranking|重排序]] 精排——这是生产标配,不是可选项。
- **延迟**:ANN 索引选 **HNSW**(查询快、内存大)或 **IVF-PQ**(省内存、略损精度);rerank 是延迟大头(cross-encoder 逐对打分),只对 top-50~100 候选 rerank,别对全量。Embedding 和 rerank 都要**批处理 + GPU**。
- **成本**:embedding 缓存(同文本不重复 embed)、向量量化(PQ/SQ 压显存)、rerank 用小模型 + 只精排头部。pgvector 复用现有 Postgres 运维能省一套独立向量库的成本。

**3)多模态 RAG 的生态(2024–2026 新增重点)**
传统多模态 RAG 要 OCR + 版面分析 + 分块,管线长且易丢图表信息。新范式**直接把文档页当图片检索**(见 [[15 多模态 RAG|多模态 RAG]]):
- **ColPali**(arXiv:2407.01449)/ **ColQwen2**——**视觉 late interaction** 检索:用 VLM(PaliGemma / Qwen2-VL)对**文档页图像**生成多向量,query 与每个 patch 做 ColBERT 式 MaxSim 匹配,**免 OCR、免复杂分块**,对表格/图表/扫描件鲁棒。`vidore/colpali` 是参考实现,**vidore benchmark** 是视觉文档检索的事实标准。
- 配套向量库:多向量 late interaction 需要支持 multi-vector 的库——**Vespa、Qdrant(multivector)、Weaviate** 已原生支持。
- 多模态 embedding:**CLIP / SigLIP / Cohere multimodal embed / Voyage multimodal** 做图文同空间检索。

**4)框架锁定与「先框架还是先组件」**
- 要快出原型 → 框架全包(LlamaIndex/LangChain);要极致可控 → 直接拼 retriever + 向量库 + LLM,别让框架黑盒挡视线。
- **编排层锁定最深、选错重写最贵**,优先稳定;向量库/embedding 层相对易换。

**5)owner 会搬家(认 org 现名别认旧链接)**
- `sentence-transformers`:UKPLab → **huggingface**(旧址 301,pip 名不变)。
- `docling`:DS4SD(IBM)→ **docling-project**,已捐 **Linux Foundation / LF AI & Data**。
- TruLens 母公司 TruEra 2024-05 被 **Snowflake** 收购。

**6)托管 vs 自托管权衡**
Pinecone / LlamaParse / Cohere / Phoenix Cloud 省运维但**收费 + 锁定 + 数据出境**(注意 [[17 检索数据治理|数据驻留]] 合规);Milvus / Qdrant / Chroma / Ragas 自托管可控但自背运维。

## 面试高频

**Q1:画一个 RAG 管线,每层用什么开源工具?**
标准答:**解析**(Docling/unstructured)→ **Embedding/Rerank**(sentence-transformers/bge/Cohere)→ **向量库**(Qdrant/Milvus/pgvector/Chroma/Pinecone)→ **检索**(向量库自带 + 混合检索)→ **重排**(cross-encoder/Cohere Rerank)→ **生成**(任意 LLM)→ **评估**(Ragas/DeepEval/Phoenix),外加**框架/编排**(LlamaIndex/LangGraph/Haystack)横跨全栈黏起来。强调**层稳定、库流动**:记「这层主流是谁」比记版本号有用。

**Q2:LlamaIndex 和 LangChain/LangGraph 怎么选?**
标准答:**LlamaIndex** RAG 出身,数据连接器/索引/查询引擎最全,文档密集和 Agentic RAG 首选;**LangChain** 是组件胶水(适配最广),**LangGraph** 把流程建成有状态图,要可控分支、Human-in-the-Loop、自适应路由时上 LangGraph。要显式 DAG + 生产可观测则 Haystack。

**Q3:向量库怎么选?FAISS、pgvector、Milvus、Pinecone 区别?**
标准答:**FAISS** 是算法库(非数据库),嵌入式极快,做原型/底座;**pgvector** 给 Postgres 加向量列,复用现有 PG 运维,百万级够用;**Milvus/Qdrant** 分布式十亿级、过滤检索强,要规模化自托管选它;**Pinecone** 全托管免运维但收费锁定。选型看规模、是否复用现有栈、托管 vs 自托管(见 [[04 Embedding 与向量数据库|Embedding 与向量数据库]])。

**Q4:复杂 PDF(表格、扫描件)解析,2026 用什么?**
标准答:**Docling**(IBM→LF,Granite-Docling 表格抽取约 97.9%)是黑马,复杂表格/版式优先;**unstructured** 格式覆盖广做基础场景;要极致复杂版式可用托管的 **LlamaParse**。或直接走视觉检索 **ColPali/ColQwen2** 免 OCR。解析这层质量直接决定下游上限。

**Q5:这么多评估框架怎么选?**
标准答:**Ragas** 事实标准(reference-free + 合成集);**DeepEval** pytest 原生最适合进 CI(G-Eval/DAG);**TruLens** RAG Triad;**ARES** 微调裁判 + PPI;**Phoenix/Langfuse** 偏线上追踪。评估这层最常被跳过却最该先建——没标尺换库换参全凭感觉(见 [[18 RAG 评估|RAG 评估]])。

## 知识拓展

- **「层稳定、库流动」是核心方法论**:RAG 被拆成七八层正交职责(解析/embed/向量库/检索/重排/生成/评估 + 编排),每层三五个竞品、每周一版还改名。罗列一百个库三个月就过期;把库**摁进固定的层**才耐用。这与 [[39 Agent 开源生态全景|Agent 开源生态全景]] 同一心智。
- **与 Agent 生态合流**:LangGraph / LlamaIndex / DSPy / Phoenix 同时出现在两张生态图——RAG 进化到 [[36 Agentic RAG|Agentic RAG]] 时,**编排层和评估层合流**(检索变多轮自主、评估变多步轨迹)。
- **DSPy 这条特殊路线**:不是检索器,而是把 RAG 管线**当可优化程序**(写 signature,自动编译 prompt/权重,见 [[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]]),让各模块「自动调参」而非手搓 prompt——和「手拼组件」是正交的两种构建哲学。
- **前沿(2024–2026)**:① **视觉 late interaction**(ColPali/ColQwen2)把多模态 RAG 从「OCR + 分块」简化为「页面当图检索」,vidore 成新基准;② **late interaction / multi-vector** 检索(ColBERT 系)在 Vespa/Qdrant 等库原生化;③ 文档解析向 **agentic** 方向走(Docling 路线图);④ RAGFlow 把 RAPTOR/AHC、Browser 组件、Agent 能力打进开箱一体机。
- **反模式**:① 每层都装满(真实管线只横跨 3~5 层);② 认旧 GitHub 链接(owner 搬家导致 301);③ 跳过评估层(没标尺优化全凭感觉);④ 盲目托管图省事却踩数据驻留合规(见 [[17 检索数据治理|检索数据治理]])。本篇与 [[13 Modular RAG|Modular RAG]] 是一体两面:模块化让每层可插拔,生态就是每层的候选池。

## 来源
- 各仓库官方 GitHub(2026 核验):`run-llama/llama_index`、`langchain-ai/langgraph`、`deepset-ai/haystack`、`infiniflow/ragflow`、`neuml/txtai`、`weaviate/Verba`、`truefoundry/cognita`、`IntelLabs/fastRAG`、`stanfordnlp/dspy`。
- 多模态/视觉检索:Faysse et al. (2024). **ColPali: Efficient Document Retrieval with Vision Language Models**. arXiv:2407.01449;`vidore/colpali`、ColQwen2、vidore benchmark;multi-vector late interaction 支持见 Vespa/Qdrant/Weaviate 官方文档(2026)。
- 文档解析:`Unstructured-IO/unstructured`、`docling-project/docling`(IBM,donated to Linux Foundation)、LlamaParse(LlamaCloud)。
- Embedding/向量库:`huggingface/sentence-transformers`(原 UKPLab)、`FlagOpen/FlagEmbedding`(bge)、`facebookresearch/faiss`、`milvus-io/milvus`、`qdrant/qdrant`、`weaviate/weaviate`、`chroma-core/chroma`、`pgvector/pgvector`、Pinecone。
- 评估:`explodinggradients/ragas`、`truera/trulens`、`confident-ai/deepeval`、`stanford-futuredata/ARES`、`Arize-ai/phoenix`。
