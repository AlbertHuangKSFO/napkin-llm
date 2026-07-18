[[01 什么是 RAG|什么是 RAG]](Retrieval-Augmented Generation,检索增强生成)是一种把**外部知识检索**接到**语言模型生成**之前的范式:回答前先从知识库捞出相关片段,拼进上下文,再让 LLM 据此作答。一句话点本质——**让模型在"开卷"状态下答题**,而不是仅凭训练时压进权重里的记忆硬背。它是整个 [[RAG]] 体系的总入口。

## 本质:parametric 与 non-parametric 记忆的拼接
Lewis et al.(2020)把预训练 seq2seq 模型称作 **parametric memory(参数化记忆)**,把由神经检索器访问的 Wikipedia 稠密向量索引称作 **non-parametric memory(非参数化记忆)**。后来的工程实践可用文档库、关键词索引或向量索引实现后者；能否更新、控制权限和附上来源，取决于索引、版本与引用链是否被实现，而非这一术语本身自动保证。

这个 parametric/non-parametric 二分正是 RAG 起源论文的核心提法(见[[#来源]])。理解它就能想清楚 RAG 到底解决什么、和微调差在哪。

### RAG 能提供什么——先看前提
- **训练后知识**:若新文档已完成解析、索引，并在该问题中被召回且放入上下文，模型就有机会依据训练截止日之后的内容回答；端到端更新时延由 ingestion、索引和缓存策略决定。
- **有证据的回答**:把检索片段交给生成器，为逐条核对与引用归因创造条件；是否更少幻觉必须在指定语料、模型、提示词和指标下评估，见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]。
- **受控访问的数据**:RAG 可把企业文档或个人笔记作为查询时的证据来源；它不是唯一途径，且仍须在检索前后执行 ACL、租户隔离和敏感数据控制。
- **可更新、可审计的知识面**:保留文档版本、chunk ID 和检索 trace 时，可以追溯某次答案使用的证据；仅替换库中文档不会自动使所有缓存、索引或既有答案同步更新。

### RAG vs 微调:四条轴，而不是二选一

| 轴 | RAG 的典型取舍 | 微调的典型取舍 | 决策时要验证 |
|---|---|---|---|
| 能力 | 让模型在回答时读取外部证据；不能替代稳定的任务行为或推理能力 | 可强化指令遵循、格式和领域行为；不保证精确记住每条新事实 | 同一组任务上的正确性、格式合规和拒答行为 |
| 知识更新 | 改文档、索引和缓存策略即可改变可检索知识面 | 需准备数据并重新训练或适配权重 | 更新 SLA、回滚路径、过期内容是否仍可被召回 |
| 成本与延迟 | 有解析、embedding、存储和每次检索/上下文 token 的成本 | 有数据标注、训练和模型托管成本；推理成本取决于部署模型 | 用真实流量测 p50/p95 延迟、单位请求成本和质量 |
| 可审计性 | 若保存 chunk、版本和 trace，可把回答关联到当次证据 | 通常难把某条输出归因到某一训练样本 | 是否满足引用、复现、访问控制和审计要求 |

实践中常把两者叠加：微调负责所需行为，RAG 负责可变的外部证据；是否值得叠加，仍以离线金标集和线上 trace 验证。更深一层的“RAG 还是把全部文档塞进长上下文”的取舍见 [[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]]。

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

### 小数字手算:检索相似度只是排序信号
设 query 向量 $q=(1,1)$，两个 chunk 向量分别为 $d_1=(1,0)$、$d_2=(1,1)$。用余弦相似度排序：

$$
\cos(q,d)=\frac{q\cdot d}{\lVert q\rVert\lVert d\rVert}
$$

$$
\cos(q,d_1)=\frac{1}{\sqrt2\cdot1}\approx0.707,\qquad
\cos(q,d_2)=\frac{2}{\sqrt2\cdot\sqrt2}=1
$$

所以此 toy 例中 $d_2$ 排在前面。相似度高并不证明它足以支持最终回答；仍要检查证据内容和生成忠实度。

三阶段数据流的纵向视角:

![[什么是 RAG-三阶段.png]]

## 可跑最小代码
```python
# 可直接运行：只用标准库复现「离线建索引 → top-k 检索 → 带证据回答」的骨架。
# 这不是生产级 embedding；替换为向量模型前，应保留 source_id / version 等元数据。
from collections import Counter
from math import sqrt
import re

docs = [
    {"id": "lewis-2020", "version": "2020-05-22", "acl": "public",
     "text": "Lewis 等人在 2020 年提出 RAG。"},
    {"id": "vector-db", "version": "2026-07-17", "acl": "public",
     "text": "向量索引可按相似度检索文档片段。"},
    {"id": "audit", "version": "2026-07-17", "acl": "internal",
     "text": "可审计回答需要保存所用文档版本和检索记录。"},
]

def terms(text):
    return Counter(re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]+", text.lower()))

def cosine(a, b):
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    norm = sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
    return dot / norm if norm else 0.0

# ---- 离线：保存可检索表示，同时保留证据 ID ----
index = [(doc, terms(doc["text"])) for doc in docs]

def retrieve(query, k=2):
    q = terms(query)
    ranked = sorted(index, key=lambda pair: cosine(q, pair[1]), reverse=True)
    return [doc for doc, _ in ranked[:k]]

# ❌ 朴素：静态答案既不检查当前证据，也没有来源、版本或访问条件。
def naive_answer(query):
    return f"问题：{query}\n回答：RAG 由 Lewis 等人在 2020 年提出。"

# ✅ 高效：先检索并计算分数，再按访问范围过滤；只从最高正分证据抽取答案。
def grounded_answer(query, allowed_acls):
    q = terms(query)
    scored = sorted(
        ((doc, cosine(q, vector)) for doc, vector in index if doc["acl"] in allowed_acls),
        key=lambda pair: pair[1], reverse=True)
    evidence = [doc for doc, score in scored if score > 0]
    if not evidence:
        return f"问题：{query}\n回答：没有足够已授权的证据，需澄清或检索。"
    answer = evidence[0]["text"]  # toy 例：抽取最匹配证据，不能补写证据外的内容
    body = "\n".join(f"[{d['id']}@{d['version']}] {d['text']}" for d in evidence)
    return f"问题：{query}\n回答：{answer}\n依据：\n{body}"

query = "RAG 是谁提出的？"
print(naive_answer(query))
grounded = grounded_answer(query, {"public"})
assert "Lewis" in grounded and "[lewis-2020@2020-05-22]" in grounded
print(grounded)

unanswerable = grounded_answer("审计性如何实现？", {"public"})
assert "没有足够已授权的证据，需澄清或检索。" in unanswerable
print(unanswerable)
```

## 对比表
| 维度 | parametric(模型权重) | non-parametric(外部库 / RAG) |
|---|---|---|
| 知识载体 | 训练压进权重 | 可检索向量库 / 文档 |
| 更新知识 | 通过训练/适配改变，粒度与时延由训练流程决定 | 通过文档、索引和缓存流程改变，时延由该流程决定 |
| 时效 | 不直接读取请求时的新文档 | 可在请求时读取已索引的较新文档，但需验证更新链路 |
| 可溯源 | 通常不能把单次输出映射到一个训练样本 | 可保留来源片段；是否可审计取决于版本与 trace 设计 |
| 受控数据 | 需审慎决定是否进入训练数据 | 可在受控检索与上下文注入下使用，仍须执行访问控制 |
| 出错样式 | 幻觉、过时 | 检索不准 / 证据未用好 |

## 何时用 / 坑
- **考虑 RAG**:答案需要引用具体文档、知识变化速度快于可接受的训练周期，或访问控制要求把数据留在受管知识库时；先用代表性 query 测召回、答案质量、权限与成本。
- **别迷信**:RAG 不保证消除幻觉；检索错误、上下文截断或生成器未忠实使用证据都可能答错——失败模式与验证方法见 [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]。
- **从基线开始而非从默认结束**:固定分块、单一稠密检索和直接拼接可以作为对照组；是否需要查询改写、混合检索、重排或 [[13 Modular RAG|Modular RAG]]，要由离线评测和 trace 决定。
- **与 Agent 的关系**:把"要不要检索、检索几次、怎么改写"交给模型自主决策,就升级成 [[36 Agentic RAG|Agentic RAG]];RAG 也常作为一个工具被 [[15 Function Calling 工具调用|Function Calling 工具调用]] / [[09 ReAct|ReAct]] 循环调用。

## 工业界实践

生产系统可按下列组件划分职责；具体产品、版本、候选数和部署形态应记录在配置中，并在自己的语料和流量上验证：

```
文档源(S3/SharePoint/Confluence/DB)
  → 解析(Unstructured.io / LlamaParse / Azure Doc Intelligence)
  → 分块([[03 分块策略 Chunking|分块]] + 元数据:source/title/page/acl)
  → embedding(选定且版本固定的模型，批量调用)
  → 写入向量库(Qdrant/Milvus/pgvector/Pinecone)+ 对象存储原文
  ─────────────── 以上离线,定时/事件触发增量 ───────────────
查询
  → 改写/扩展([[07 查询变换 Query Transformation|Query Transformation]])
  → 混合召回([[08 混合检索 Hybrid Search|dense+BM25]]，候选数由评测选定)
  → 重排([[10 重排序 Reranking|重排序]]，保留数由延迟、成本和质量约束选定)
  → 拼 prompt(含引用模板)→ LLM 生成 → 引用归因 + 后处理
  → 全链路 trace 落到可观测平台(LangSmith / Langfuse / Arize Phoenix)
```

**框架不是架构结论**:LangChain/LangGraph、LlamaIndex、Haystack 等都能封装部分组件；选择依据应是当前版本的 API 稳定性、团队可观测性接入、部署要求与升级成本。模块是否需要循环、路由或 agent 调度，应在需求和评测证明后再引入，见 [[13 Modular RAG|Modular RAG]] 与 [[36 Agentic RAG|Agentic RAG]]。

**规模化的三角权衡**——召回率 / 延迟 / 成本,无法同时拉满:
- 召回率↑:候选 k 加大、上混合检索 + 重排、用更强 embedding。但 k 大→重排慢、塞进 prompt 的 token 多→贵且慢。
- 延迟↓:调整索引搜索参数、重排候选数与缓存策略；应同时报告召回质量，避免把“更快”误当成更好。
- 成本↓:评估量化、向量维度、模型规格和缓存策略；节省幅度依赖数据、硬件、索引和供应商版本，不能从别处的数字直接外推。

**评估与可观测**:上线前在自建测试集上报告 [[18 RAG 评估|RAG 评估]] 指标及其实现版本。Ragas 原论文提出的是无人工金标也可用的评估框架；关键事实正确性和权限合规仍建议抽样人工标注核验。线上可记录每条 trace 的语料/索引版本、检索片段、token、延迟与用户反馈，再做离线回归。

**踩坑速记**(生产高频):
- **检索片段不进 prompt 也白搭**:k 设太大被截断、lost-in-the-middle,关键证据排中间被忽视。重排把它顶到首尾。
- **embedding 换版通常需要重建或并行索引**:先检查维度、归一化和距离度量兼容性；以版本化的查询/文档向量和灰度评测确认，不要混用不兼容的表示。
- **元数据过滤先行**:按 `acl`/`tenant`/`时间` 先过滤再向量搜,既对又快又安全(见 [[16 检索安全与访问控制|检索安全与访问控制]])。
- **prompt 缓存**:若所用模型与计费版本支持，固定提示和稳定文档前缀可能降低成本或延迟；以供应商当期文档和实际账单验证命中条件与收益。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1:RAG 和微调怎么选?能不能都不用,直接长上下文塞进去?**
标准答:按四轴回答：能力、知识更新、成本/延迟、可审计性。RAG 让模型在请求时读取已索引证据，适合需要引用、版本和受控访问的知识面；微调更适合强化行为、格式或领域任务表现；两者可以叠加。长上下文适用于文档规模、时延和成本都经评测可接受且需要全局推理的任务。不要承诺固定更新时延或默认质量优势，详见 [[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]]。
*追问「RAG 能消除幻觉吗?」* → 不能保证。只有在指定语料、模型、提示与指标的评测中，才能判断它是否降低了无依据回答；检索错误或 LLM 未忠实使用证据仍会出错。

**Q2:画一下 RAG 的完整数据流,离线和在线各做什么?**
标准答:离线 = 加载清洗 → 分块 → embedding → 入库(向量+原文+元数据);在线 = query 向量化 → ANN 检索 top-k →(可选 改写/混合/重排)→ 拼 prompt → 生成 → 引用归因。能主动补「生产里 query 侧还有改写、混合、重排三道工序」是加分项。

**Q3:RAG 起源论文提出了什么核心概念?**
标准答:Lewis et al. 2020(NeurIPS)首创 "RAG" 术语,提出 **parametric memory(权重里的知识)/ non-parametric memory(可检索外部库)** 二分;原始架构 = **DPR 检索器 + BART 生成器** 端到端微调;两种解码变体 **RAG-Sequence**(整段答案用同一篇文档)/ **RAG-Token**(逐 token 可换文档)。

**Q4:为什么 RAG 常被考虑用于更新快或受控的数据?**
标准答:因为它能把回答关联到请求时检索到的、带版本的证据，而不用把每次文档更新都写进权重。但“更新快”和“可审计”是系统性质：ingestion、索引、缓存、ACL 和 trace 都要实测与实现。微调仍可能更适合所需行为/格式；不要把二者说成互斥。
*陷阱*:面试官追问“为什么不全用 RAG?” → 检索质量、上下文预算、延迟和成本都有限；需要评估证据覆盖、引用忠实度和任务行为。

**Q5(陷阱题):给你一个 RAG 系统答错了,你怎么定位是检索的锅还是生成的锅?**
标准答:先用 [[18 RAG 评估|评估]] 形成诊断假设，再查看同一条样本的语料/索引版本、召回列表、上下文拼装和生成引用。context recall/precision、faithfulness 等分数不能直接证明根因；要用人工标注或带 gold evidence 的测试集验证。确认是召回、拼装还是生成问题后，才 A/B 测混合检索、重排、查询改写或引用约束。

## 知识拓展

- **RAG 的演进谱系**:Naive([[02 Naive RAG 与失败模式|02]])→ Advanced(检索前后打补丁:[[07 查询变换 Query Transformation|查询变换]]、[[10 重排序 Reranking|重排序]])→ [[13 Modular RAG|Modular RAG]](可重排/循环/路由)→ [[36 Agentic RAG|Agentic RAG]](把「要不要检索、检几次」交模型自主)。这是面试讲「RAG 怎么从玩具到生产」的主线。
- **检索粒度的前沿**:[[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|Contextual Retrieval(Anthropic，2024-09-19)]] 在每个 chunk 前加入文档级上下文，再为 embedding 与 BM25 建索引。Anthropic 在其代码、小说、论文和科学论文等实验中，用 Gemini Text 004、top-20 和 $1-\mathrm{recall@20}$ 报告组合方案 $5.7\%\to2.9\%$（49%）及加入 Cohere 重排后的 $5.7\%\to1.9\%$（67%）；这是该实验条件下的结果，不是固定收益。RAPTOR(2024)做层次化摘要树。
- **数学根基**:RAG 检索的「语义相近 = 向量相近」依赖余弦/点积度量,几何直觉见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]];embedding 模型本身是 [[LLM/054 词嵌入层与权重绑定|词嵌入]] 思想在句/段级的延伸。
- **边界与反模式**:① 数据量小、更新少、要全局推理 → 别硬上 RAG,长上下文更简单(见 [[19 RAG vs 长上下文 vs Agentic Search|19]])。② 把 RAG 当万能补丁——知识根本不在库里(FP1 Missing Content)时,再强的检索器也救不了,那是数据治理问题([[17 检索数据治理|17]])。③ 不评估就堆模块,盲目加重排/多跳只增延迟不增准。
- **前沿方向(带年份)**:**Self-RAG**(Asai et al. 2023)让模型自评要不要检索、证据够不够;**CRAG**(2024)给检索结果打分并触发纠错检索;**GraphRAG**(微软 2024,见 [[14 GraphRAG 知识图谱检索|14]])用知识图谱做全局性问答;**Agentic / 多跳检索**(IRCoT、FLARE,见 [[09 多跳检索：IRCoT、Self-Ask、FLARE|09]])处理需多步推理的复杂问题。

## 关键事实
- Lewis et al. 的 RAG 论文 arXiv v1 发布于 **2020-05-22**，NeurIPS 2020 接收；其模型把预训练 seq2seq 的参数记忆与由神经检索器访问的 Wikipedia 稠密向量索引结合。
- 原始架构 = **DPR 检索器 + BART 生成器**，端到端微调；两种解码变体：**RAG-Sequence**（整段答案用同一篇检索文档）与 **RAG-Token**（逐 token 可换不同文档）。
- Gao et al. 综述 arXiv v1 发布于 **2023-12-18**，以 Naive / Advanced / Modular 组织 RAG 演进；这是综述的分类框架，不是所有系统必须经历的路线。
- Ragas 论文 arXiv v1 发布于 **2023-09-26**，提出不依赖人工 ground truth 的多维评估框架；使用哪个指标及其含义要以固定的 Ragas 版本为准。

## 来源
- Lewis, P., Perez, E., Piktus, A., et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. NeurIPS 2020. arXiv:2005.11401. <https://arxiv.org/abs/2005.11401> — 原始 RAG 机制与发布日期。
- Gao, Y., Xiong, Y., Gao, X., et al. (2023). **Retrieval-Augmented Generation for Large Language Models: A Survey**. arXiv:2312.10997. <https://arxiv.org/abs/2312.10997> — Naive / Advanced / Modular 分类。
- Es, S., et al. (2023). **Ragas: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217. <https://arxiv.org/abs/2309.15217> — reference-free 多维评估框架。
- Anthropic (2024-09-19). **Introducing Contextual Retrieval**. <https://www.anthropic.com/engineering/contextual-retrieval> — 给出了模型、top-k、语料和失败率定义的实验条件。
