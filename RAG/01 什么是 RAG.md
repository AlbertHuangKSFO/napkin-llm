[[01 什么是 RAG|什么是 RAG]](Retrieval-Augmented Generation,检索增强生成)是一种把**外部知识检索**接到**语言模型生成**之前的范式:回答前先从知识库捞出相关片段,拼进上下文,再让 LLM 据此作答。一句话点本质——**让模型在"开卷"状态下答题**,而不是仅凭训练时压进权重里的记忆硬背。它是整个 [[RAG]] 体系的总入口。

## 本质:parametric 与 non-parametric 记忆的拼接
LLM 训练时把知识压成权重,这叫 **parametric memory(参数化记忆)**——容量大但模糊、过时、改不动、说不出处。RAG 额外挂一份 **non-parametric memory(非参数化记忆)**:一个可检索的外部知识库(通常是向量库)。生成时两者协作:参数化记忆提供语言能力与常识,非参数化记忆提供**精确、可更新、可溯源**的事实片段。

这个 parametric/non-parametric 二分正是 RAG 起源论文的核心提法(见[[#来源]])。理解它就能想清楚 RAG 到底解决什么、和微调差在哪。

### RAG 解决的四类问题
- **知识截止(knowledge cutoff)**:模型训练有时间下限,2023 训练的模型不知道 2025 的事;RAG 把新文档塞进库即可,无需重训。
- **幻觉(hallucination)**:无据时模型编造;给定检索证据后,生成被"锚"在真实片段上,幻觉率显著下降(但不归零,见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]])。
- **私有 & 实时数据**:企业内网文档、个人笔记、刚发布的数据——这些从不在公开预训练语料里,只能靠检索喂进去。
- **可更新、可溯源**:库里换一篇文档,知识立刻更新;答案能附上来源片段,做[[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]]里的引用归因。

### RAG vs 微调:改知识 vs 改权重
关键对照:**RAG 改的是外部库,微调改的是模型权重**。要让模型"知道一件新事实",RAG 只需往库里加文档(分钟级、可回滚、可溯源);微调要准备数据、跑训练、还可能灾难性遗忘且学到的事实仍不可溯源。经验法则:**注入/更新知识用 RAG,改变行为风格/输出格式/领域语感用微调**,二者常叠加。更深一层的"RAG 还是干脆把全部文档塞进长上下文"的权衡见 [[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]]。

## 机制:离线索引 + 在线检索→生成两阶段

![[什么是 RAG.png]]

### 阶段一 离线索引(Offline Indexing)
把原始语料变成可检索的索引,通常一次性构建、之后增量更新:
1. **加载 & 清洗**:PDF/网页/数据库 → 纯文本。
2. **分块 Chunking**:长文切成片段,粒度直接决定检索质量,见 [[03 分块策略 Chunking|分块策略 Chunking]] 与 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]]。
3. **Embedding**:每个片段过嵌入模型变成稠密向量,见 [[04 Embedding 与向量数据库|Embedding 与向量数据库]]。
4. **入库**:向量 + 原文 + 元数据写进向量数据库(non-parametric 记忆的物理载体)。

### 阶段二 在线问答(Online)
每次用户提问触发:
1. **检索 top-k**:Query 也向量化,在向量库里做相似度搜索,取回最相关的 k 个片段。可叠加 [[07 查询变换 Query Transformation|查询变换 Query Transformation]] 改写问句、[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 融合稀疏+稠密、[[10 重排序 Reranking|重排序 Reranking]] 精排候选。
2. **拼装上下文**:把检索片段和原始 Query 按模板拼成 Prompt(`已知以下资料:{chunks}\n问题:{query}`)。
3. **生成**:LLM 据上下文作答,理想情况下**只用给定证据**,并标注来源做引用归因。

三阶段数据流的纵向视角:

![[什么是 RAG-三阶段.png]]

## 可跑最小代码
```python
# 极简 RAG:展示离线索引 + 在线检索→生成两阶段的骨架(伪实现)
from sentence_transformers import SentenceTransformer
import numpy as np

embed = SentenceTransformer("all-MiniLM-L6-v2")

# ---- 离线:索引 ----
docs = [
    "RAG 由 Lewis 等人在 2020 年 NeurIPS 论文中提出。",
    "向量数据库存储片段的稠密向量,支持相似度检索。",
    "微调改的是模型权重,RAG 改的是外部知识库。",
]
doc_vecs = embed.encode(docs)  # 每个片段 → 向量,实际写入向量库

# ---- 在线:检索 top-k ----
def retrieve(query, k=2):
    q = embed.encode([query])[0]
    sims = doc_vecs @ q / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(q))
    idx = sims.argsort()[::-1][:k]      # 余弦相似度取 top-k
    return [docs[i] for i in idx]

# ---- 在线:拼上下文 → 生成 ----
def rag_answer(query, llm):
    ctx = "\n".join(f"- {c}" for c in retrieve(query))
    prompt = f"仅根据以下资料回答,并标注依据:\n{ctx}\n\n问题:{query}"
    return llm(prompt)                  # 交给任意 LLM 客户端

# print(rag_answer("RAG 是谁提出的?", my_llm))
```

## 对比表
| 维度 | parametric(模型权重) | non-parametric(外部库 / RAG) |
|---|---|---|
| 知识载体 | 训练压进权重 | 可检索向量库 / 文档 |
| 更新知识 | 重训 / 微调 | 改库即可,分钟级 |
| 时效 | 卡在训练截止 | 实时,加文档即更新 |
| 可溯源 | 几乎不可 | 可附来源片段 |
| 私有数据 | 需训练时见过 | 检索时喂入即可 |
| 出错样式 | 幻觉、过时 | 检索不准 / 证据未用好 |

## 何时用 / 坑
- **该用**:需要私有/实时/可溯源知识,事实易变,答案要附依据。
- **别迷信**:RAG 不消除幻觉,只是降低;检索捞错或 LLM 不忠实证据照样错——失败模式与各自解法见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]。
- **全默认就翻车**:固定分块 + 单稠密检索 + 直接塞 + 直接生成的 Naive 配置极易出问题,真实系统几乎都要进阶到 [[13 Modular RAG|Modular RAG]]。
- **与 Agent 的关系**:把"要不要检索、检索几次、怎么改写"交给模型自主决策,就升级成 [[36 Agentic RAG|Agentic RAG]];RAG 也常作为一个工具被 [[15 Function Calling 工具调用|Function Calling 工具调用]] / [[09 ReAct|ReAct]] 循环调用。

## 工业界实践

真实生产里没人写「极简 RAG」那种手搓循环。一条**典型企业 RAG 数据流**长这样:

```
文档源(S3/SharePoint/Confluence/DB)
  → 解析(Unstructured.io / LlamaParse / Azure Doc Intelligence)
  → 分块([[03 分块策略 Chunking|分块]] + 元数据:source/title/page/acl)
  → embedding(OpenAI v3 / Cohere v3 / bge-m3,批量调用)
  → 写入向量库(Qdrant/Milvus/pgvector/Pinecone)+ 对象存储原文
  ─────────────── 以上离线,定时/事件触发增量 ───────────────
查询
  → 改写/扩展([[07 查询变换 Query Transformation|Query Transformation]])
  → 混合召回([[08 混合检索 Hybrid Search|dense+BM25]],各取 top-50)
  → 重排([[10 重排序 Reranking|Cohere Rerank / bge-reranker]],精排到 top-5)
  → 拼 prompt(含引用模板)→ LLM 生成 → 引用归因 + 后处理
  → 全链路 trace 落到可观测平台(LangSmith / Langfuse / Arize Phoenix)
```

**框架选型**(三大主流,选一个别混搭):
- **LangChain / LangGraph**:生态最大、集成最全,LangGraph 适合做有环的 [[13 Modular RAG|Modular RAG]] / [[36 Agentic RAG|Agentic RAG]]。缺点是抽象层厚、版本动荡。
- **LlamaIndex**:专为「数据→索引→查询」而生,`Node`/`Retriever`/`QueryEngine` 抽象贴 RAG,small-to-big、路由、子问题开箱即用。做纯检索问答最顺手。
- **Haystack**(deepset):Pipeline 显式声明、组件化清晰,偏工程化/企业部署,可观测性好。

**最小生产骨架**(LangChain 风格,展示真实组件而非手搓):

```python
# 生产 RAG 不手搓:用框架把「检索器 + 重排 + 生成」串成可观测管线
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

emb = OpenAIEmbeddings(model="text-embedding-3-large")
store = QdrantVectorStore.from_existing_collection(  # 持久化向量库
    embedding=emb, collection_name="kb", url="http://qdrant:6333")

base = store.as_retriever(search_kwargs={"k": 50})    # 粗召回 top-50
reranker = CohereRerank(model="rerank-v3.5", top_n=5)  # 交叉编码器精排到 5
retriever = ContextualCompressionRetriever(            # 召回→重排两段式
    base_compressor=reranker, base_retriever=base)

PROMPT = "仅根据资料回答,每句话末尾用 [n] 标注来源编号。无法回答就说不知道。\n资料:\n{ctx}\n问题:{q}"
def answer(q, llm=ChatOpenAI(model="gpt-4o")):
    docs = retriever.invoke(q)
    ctx = "\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    return llm.invoke(PROMPT.format(ctx=ctx, q=q))      # 全程被 LangSmith 自动 trace
```

**规模化的三角权衡**——召回率 / 延迟 / 成本,无法同时拉满:
- 召回率↑:候选 k 加大、上混合检索 + 重排、用更强 embedding。但 k 大→重排慢、塞进 prompt 的 token 多→贵且慢。
- 延迟↓:HNSW 换更激进的 `efSearch`、重排只跑前 N、embedding 缓存、prompt 缓存(Anthropic/OpenAI 都支持)。
- 成本↓:向量量化(int8 / 二值化,Qdrant/Milvus 支持,内存降 4–32 倍)、用小 embedding 维度(Matryoshka 裁剪)、便宜的生成模型 + 重排兜底质量。

**评估与可观测**:上线前用 [[18 RAG 评估|RAG 评估]] 里的 **Ragas**(faithfulness / context precision / context recall / answer relevancy 四件套,见 [[#来源]])在自建测试集上跑分;线上用 **Langfuse / LangSmith / Arize Phoenix** 抓每条 trace 的检索片段、token、延迟、用户反馈,做离线回归。

**踩坑速记**(生产高频):
- **检索片段不进 prompt 也白搭**:k 设太大被截断、lost-in-the-middle,关键证据排中间被忽视。重排把它顶到首尾。
- **embedding 换版 = 全量重建索引**,且要灰度。线上悄悄换模型会让旧库与新 query 向量不对齐。
- **元数据过滤先行**:按 `acl`/`tenant`/`时间` 先过滤再向量搜,既对又快又安全(见 [[16 检索安全与访问控制|检索安全与访问控制]])。
- **prompt 缓存**:把固定的系统提示 + 高频文档前缀缓存,长 RAG prompt 成本可降一大截。

## 面试高频

**Q1:RAG 和微调怎么选?能不能都不用,直接长上下文塞进去?**
标准答:注入/更新**知识**用 RAG(改库分钟级、可溯源、可回滚),改**行为/风格/格式/领域语感**用微调,二者常叠加。长上下文(把全部文档塞进窗口)适合文档量小、一次性、要全局推理的场景,但 token 成本随长度线性涨、有 lost-in-the-middle、且无法溯源到具体片段。详见 [[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]]。
*追问「RAG 能消除幻觉吗?」* → 不能,只降低。检索捞错或 LLM 不忠实证据照样错(陷阱:别说「消除」,会被抓)。

**Q2:画一下 RAG 的完整数据流,离线和在线各做什么?**
标准答:离线 = 加载清洗 → 分块 → embedding → 入库(向量+原文+元数据);在线 = query 向量化 → ANN 检索 top-k →(可选 改写/混合/重排)→ 拼 prompt → 生成 → 引用归因。能主动补「生产里 query 侧还有改写、混合、重排三道工序」是加分项。

**Q3:RAG 起源论文提出了什么核心概念?**
标准答:Lewis et al. 2020(NeurIPS)首创 "RAG" 术语,提出 **parametric memory(权重里的知识)/ non-parametric memory(可检索外部库)** 二分;原始架构 = **DPR 检索器 + BART 生成器** 端到端微调;两种解码变体 **RAG-Sequence**(整段答案用同一篇文档)/ **RAG-Token**(逐 token 可换文档)。

**Q4:为什么 RAG 比单纯把知识微调进模型更适合「实时/私有」场景?**
标准答:权重更新慢且不可溯源,微调一条新事实要重训、可能灾难性遗忘、还说不出处;RAG 加一篇文档即生效、可回滚、能附来源。
*陷阱*:面试官常追问「那为什么不全用 RAG?」→ 因为 RAG 救不了「需要内化的能力/风格/推理模式」,这些得靠微调。

**Q5(陷阱题):给你一个 RAG 系统答错了,你怎么定位是检索的锅还是生成的锅?**
标准答:先用 [[18 RAG 评估|评估]] 拆两侧——看 **context recall/precision**(检索侧:对的片段进 top-k 了吗)和 **faithfulness**(生成侧:答案有没有忠实于检索到的证据)。检索侧低 → 补混合检索/重排/查询改写;生成侧低 → 上引用约束 / [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG/CRAG]]。别一上来就换 embedding 模型——很多失败在「检到了但没用好」的衔接处(见 [[02 Naive RAG 与失败模式|七个失败点]] FP3/FP4)。

## 知识拓展

- **RAG 的演进谱系**:Naive([[02 Naive RAG 与失败模式|02]])→ Advanced(检索前后打补丁:[[07 查询变换 Query Transformation|查询变换]]、[[10 重排序 Reranking|重排序]])→ [[13 Modular RAG|Modular RAG]](可重排/循环/路由)→ [[36 Agentic RAG|Agentic RAG]](把「要不要检索、检几次」交模型自主)。这是面试讲「RAG 怎么从玩具到生产」的主线。
- **检索粒度的前沿**:[[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|Contextual Retrieval(Anthropic 2024)]] 给每个 chunk 注入文档级上下文再 embedding,combined Contextual Embeddings + Contextual BM25 把 top-20 检索失败率降 49%(5.7%→2.9%),叠重排降 67%(见 [[#来源]])。RAPTOR(2024)做层次化摘要树。
- **数学根基**:RAG 检索的「语义相近 = 向量相近」依赖余弦/点积度量,几何直觉见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]];embedding 模型本身是 [[LLM/054 词嵌入层与权重绑定|词嵌入]] 思想在句/段级的延伸。
- **边界与反模式**:① 数据量小、更新少、要全局推理 → 别硬上 RAG,长上下文更简单(见 [[19 RAG vs 长上下文 vs Agentic Search|19]])。② 把 RAG 当万能补丁——知识根本不在库里(FP1 Missing Content)时,再强的检索器也救不了,那是数据治理问题([[17 检索数据治理|17]])。③ 不评估就堆模块,盲目加重排/多跳只增延迟不增准。
- **前沿方向(带年份)**:**Self-RAG**(Asai et al. 2023)让模型自评要不要检索、证据够不够;**CRAG**(2024)给检索结果打分并触发纠错检索;**GraphRAG**(微软 2024,见 [[14 GraphRAG 知识图谱检索|14]])用知识图谱做全局性问答;**Agentic / 多跳检索**(IRCoT、FLARE,见 [[09 多跳检索：IRCoT、Self-Ask、FLARE|09]])处理需多步推理的复杂问题。

## 关键事实
- 术语 "RAG" 由 Lewis et al. 2020(NeurIPS)首创,该文同时提出 parametric/non-parametric 记忆框架。
- 原始架构 = **DPR 检索器 + BART 生成器**,端到端微调;两种解码变体:**RAG-Sequence**(整段答案用同一篇检索文档)与 **RAG-Token**(逐 token 可换不同文档)。
- 三段式演进 Naive → Advanced → Modular 由 Gao et al. 2023 综述总结,见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]。
- 工程化常见失败点的权威清单见 Barnett et al. 2024(七个失败点)。

## 来源
- Lewis, P., Perez, E., Piktus, A., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. NeurIPS 2020. arXiv:2005.11401. — FAIR 出品,提出 RAG 术语、parametric/non-parametric 记忆、DPR+BART、RAG-Sequence / RAG-Token。
- Gao, Y., Xiong, Y., Gao, X., et al. (2023). **Retrieval-Augmented Generation for Large Language Models: A Survey**. arXiv:2312.10997. — Naive / Advanced / Modular 三范式综述。
- Es, S., et al. (2023). **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217. — faithfulness / answer relevancy / context precision / context recall 四件套,无参考评估。
- Anthropic (2024). **Introducing Contextual Retrieval**. anthropic.com/news/contextual-retrieval. — Contextual Embeddings + Contextual BM25 将 top-20 检索失败率降 49%,叠重排降 67%。
