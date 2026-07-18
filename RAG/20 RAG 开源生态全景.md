[[20 RAG 开源生态全景|RAG 开源生态全景]]是一张按职责分层、按约束选型的地图，而不是排行榜。一个 [[01 什么是 RAG|RAG]] 系统通常由文档解析 → 表示/索引 → 检索与重排 → 生成与引用 → 评估/可观测组成，框架/编排横跨其间。选型应从文档类型、语言、ACL、是否多模态和实测 SLO 出发；仓库热度、单点宣传数字和默认 branch 都不是生产证据。

## 本质：层稳定，源码与版本流动

生态的层职责相对稳定，具体仓库、owner、许可证、维护状态和 API 却会变。因此每次采用一个项目都要同时记录**官方 source manifest**与本地 benchmark：

- source manifest 回答“谁维护、从哪里取源码、锁到哪个 tag/commit、何时核验、它的许可/托管/数据约束是什么”；
- local benchmark 回答“在**我们的**文档、语言、ACL、硬件和 SLO 下，质量、P95、索引体积与成本是否达标”。

同层可以互换，跨层可以叠加；但不能把“README 能跑”推成“适合受限数据或低延迟服务”。这与 [[13 Modular RAG|Modular RAG]] 的可插拔思想配套：模块可换，评价标准也必须可换且可复现。

![[RAG 开源生态全景.png]]

## 直觉与手算：先翻目录卡，再用放大镜看候选页

把页面图像 late interaction 想成图书馆查一份合同：先用**目录卡**（metadata、OCR 文本的精确词、稀疏/单向量检索）从全馆找出可能的 20 页，再用**放大镜**逐页看印章、表格单元和布局。若一开始就对全馆每个图像 patch 用放大镜，细节虽然保留，时间、显存和索引空间都会失控；若只看目录卡，又可能漏掉表格位置和图示语义。序列号、金额和条款原文这类逐字问题，应优先在 OCR/text 中 exact match，并保留页面回跳作为核对证据。

令 query 有 $m$ 个归一化 token 向量 $q_i$，页面有 $n$ 个归一化 patch 向量 $p_j$。late interaction 对每个 query token 选择最匹配的 page patch：

$$
\begin{aligned}
s_{ij} &= q_i^\top p_j && \text{（归一化后为 cosine similarity）}\\
a_i &= \max_{1\le j\le n} s_{ij} && \text{（第 }i\text{ 个 query token 的最佳 patch）}\\
S(q,d) &= \sum_{i=1}^{m} a_i && \text{（页面的 MaxSim 分数）}\\
-m &\le S(q,d) \le m && \text{（因为 }-1\le s_{ij}\le1\text{）}
\end{aligned}
$$

**小数字手算。**若 $m=2,n=3$，归一化点积矩阵为

$$
\begin{bmatrix}
0.9 & 0.2 & 0.1\\
0.3 & 0.8 & 0.4
\end{bmatrix}
$$

则 $a_1=\max(0.9,0.2,0.1)=0.9$，$a_2=\max(0.3,0.8,0.4)=0.8$，所以 $S(q,d)=0.9+0.8=1.7$。对 $|C|$ 个候选页，patch 比较量近似为

$$
C_{\text{late}}=|C|\cdot m\cdot n.
$$

若先从 1,000 页粗召回到 20 页，同样的 $m=2,n=3$ 下，比较量由 $1000\times2\times3=6000$ 降为 $20\times2\times3=120$，是 $50\times$ 的缩减；真实系统仍须实测编码、传输、索引、缓存与 P95，不能把这笔算术当成延迟承诺。

## 机制：先按约束切层，再做本地基准

### ① 框架 / 编排

- **LlamaIndex**：数据连接、索引、query engine 等 RAG 原语较集中；适合希望显式组织文档/索引/检索边界的团队。
- **LangChain + LangGraph**：前者偏组件适配，后者偏有状态图与人工介入；适合有分支、重试、授权和 [[12 Self-RAG、CRAG 与 Adaptive RAG|自适应路由]] 的流程。
- **Haystack**：组件和 pipeline 明确，适合希望将检索与生成链路以显式图配置、单测和观测管理的团队。
- **RAGFlow**：一体化文档处理与检索应用，适合优先验证端到端文档体验、能接受其部署边界的场景。
- **DSPy**：不是向量库或 retriever，而是将模块签名、示例与指标联动优化；适合已有可评估任务、要系统化调 prompt/模块行为时使用，见 [[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]]。

### ② 文档解析与表示

- **unstructured**：多格式接入与内容分区；先用自己的 PDF、网页、扫描件和表格样本检查结构丢失、OCR、语言覆盖和数据出境路径。
- **Docling**：本地文档转换与结构化输出选项；不要把单论文或厂商 benchmark 的表格数字搬成通用结论。应在相同文档类型、标注规则、硬件和版本条件下复现，或只写“论文在其条件下报告的结果”。
- **sentence-transformers / FlagEmbedding（BGE）**：本地 embedding、cross-encoder rerank 与多语模型的常用接口/模型来源。模型语言与 query 语言必须一致或明确采用多语模型，不能用英文 embedding 却拿中文 query 作“最小示例”。

### ③ 索引、检索与重排

- **FAISS**：近邻检索算法库，不是带 ACL、持久化、备份和租户隔离的完整数据库。
- **Qdrant / Milvus / Weaviate / pgvector**：选择前先验证 metadata filter、命名空间/租户隔离、备份恢复、混合检索、客户端与服务端兼容性，以及 ACL 过滤是否发生在召回之前。[[08 混合检索 Hybrid Search|混合检索]] 与 [[10 重排序 Reranking|重排序]] 是质量/延迟权衡中的独立阶段。
- 重排以 cross-encoder 或模型 API 为独立步骤，必须按候选量、批量、设备和 query 长度报告本地 P50/P95，不能从别人演示的单次时延推导 SLO。

### ④ 评估与可观测

- **Ragas / DeepEval**：离线集、回归门禁与 judge 校准，字段与版本细节见 [[18 RAG 评估|RAG 评估]]。
- **Phoenix**：以 trace、span 与评估结果关联线上观测；线上 trace 仍需要脱敏、留存期限、访问控制和采样策略。

## 可运行的最小代码：语言匹配的 embedding 检查

❌ 反例：选择英文单语 embedding，却把“中文问题”作为示例 query；即使代码运行，语言失配也会污染演示结论。

✅ 下例选择多语的 `BAAI/bge-m3`，对中文文档和中文 query 做可执行的余弦召回。首次运行会下载模型；需在受控 Python 环境安装 `sentence-transformers`、`torch`，并按模型许可证/网络策略处理缓存。本文未在当前工作区安装依赖或下载模型，故不宣称已实跑。

```python
# embed_zh.py
# pip install "sentence-transformers" "torch"
from sentence_transformers import SentenceTransformer
import numpy as np

documents = [
    "检索增强生成先从知识库召回证据，再由模型基于证据回答。",
    "向量索引需要结合元数据过滤和访问控制，避免越权召回。",
]
query = "如何防止检索到没有权限看的文档？"

model = SentenceTransformer("BAAI/bge-m3")  # 多语模型；模型 revision 应写入 run manifest
vectors = model.encode(documents + [query], normalize_embeddings=True)
scores = vectors[:-1] @ vectors[-1]          # 已归一化，点积即 cosine similarity
best = int(np.argmax(scores))
print({"doc": documents[best], "score": float(scores[best])})
```

这段代码只验证“表示空间和语言样本是否能被调用”，不验证 ACL 正确性、长文召回、P95 或业务效果。将它替换成真实语料、加上 `doc_version`、ACL filter、gold evidence 和 [[18 RAG 评估|五条评估线]]，才是可比较的系统基准。

## 选型卡：用输入约束而非库名决策

| 输入约束 | 优先验证的层与能力 | 基准样本 / 指标 | 典型风险 |
|---|---|---|---|
| 文本 PDF、网页、办公文档 | parser 的结构、标题、表格、页码 | 结构 F1、evidence 可定位率、失败率 | 将解析文本当原文，引用不能回页 |
| 扫描件、图表、复杂布局 | OCR/版面或视觉检索 | 图表问题 Recall@k、人工 claim→证据审查 | OCR 丢表格关系、图页无可追溯版本 |
| 中文/多语 query 与语料 | 多语 embedding、语言分桶 rerank | 按语言的 nDCG、Recall@k、拒答率 | 英文模型+中文 query 的假阳性演示 |
| 每用户/部门不同权限 | 预过滤 ACL、审计 trace、租户隔离 | 越权召回率必须为 0、授权回放 | 先召回再过滤导致泄露 |
| 图片/页面级文档检索 | multi-vector / late interaction | page Recall@k、索引体积、端到端 P95 | 把多向量成本当成普通单向量成本 |
| 明确 SLO（例如查询 P95、预算） | 召回、重排、缓存、索引参数 | 冷/热 P50/P95、QPS、GPU/CPU、每 query 成本 | 只报平均或单 query demo |

### 视觉 multi-vector：需要版本门槛与两阶段基准

ColPali / ColQwen2 一类视觉 late interaction 会为一页产生多个向量，以 query 与 patch 向量的 MaxSim 做匹配；它能保留图表与布局信号，但不等于“免费免 OCR”。部署前应满足三个门槛：

1. **版本门槛**：锁定视觉模型 revision、tokenizer、向量库**服务端与 client** release tag/commit；用同一容器实际写入并查询 multi-vector，不能仅根据 README 的“support”字样判定可用。
2. **两阶段检索**：先用单向量/BM25/metadata 做粗召回（降低候选页数），只对候选页做 late interaction 精排；全库逐 patch MaxSim 通常不满足可控 SLO。
3. **成本门槛**：在同一语料、页分辨率、chunk/page 策略与硬件下，记录索引构建耗时、向量数/磁盘、query P50/**P95**、候选数、显存/CPU 和每 query 成本；质量同时以 page Recall@k 与 citation 可回溯率验证。

**资源与回退 caveat（有来源）**：ColPali 官方实现以页面 patch 的 multi-vector 表示和 CUDA/Apple Silicon 的 MaxSim kernel 评分；Qdrant 官方文档明确指出一个逻辑文档的 token-level vectors 可达数百，逐向量建索引会造成高 RAM 与慢写入，因此将它作为 first-pass 之后的 rerank。由此，页面图像 late interaction 应当是一条**两阶段支路**：先用 metadata、文本/OCR 稀疏精确匹配或单向量粗召回缩小候选，再对候选页做多向量精排；同时保留可检索的文本/OCR 与页图映射。对序列号、条款原文、金额等要求 exact-string 的问题，先走文本/OCR 的精确查找，失败或版面语义关键时再回退到页面图像检索/人工核对，不能把视觉向量结果伪装成逐字证据。

视觉路线与 [[15 多模态 RAG|多模态 RAG]] 互补：文本/OCR 流程仍可能更适合需要精确文字 span、低资源或严控索引成本的场景。

## source manifest（官方来源，而非热度榜）

下表覆盖本篇实际提到的项目。`tag / pin` 不以会变化的“最新”冒充固定版本：采用时应把官方 release tag **及其 commit SHA** 写入 `uv.lock`、容器 digest 或部署 manifest；没有 release 的项目固定审计过的 commit。核验日均为 2026-07-17。

| 项目 | 官方 URL | owner / repo | tag / pin | 核验日 | 维护状态 | 约束（必须另验） |
|---|---|---|---|---|---|---|
| LlamaIndex | [GitHub](https://github.com/run-llama/llama_index) | `run-llama/llama_index` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | connector 依赖、模型/数据出境 |
| LangChain | [GitHub](https://github.com/langchain-ai/langchain) | `langchain-ai/langchain` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 集成包版本组合、provider 数据路径 |
| LangGraph | [GitHub](https://github.com/langchain-ai/langgraph) | `langchain-ai/langgraph` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 状态持久化、人审与授权边界 |
| Haystack | [GitHub](https://github.com/deepset-ai/haystack) | `deepset-ai/haystack` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | pipeline 组件/许可/遥测 |
| RAGFlow | [GitHub](https://github.com/infiniflow/ragflow) | `infiniflow/ragflow` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 部署面、模型与存储依赖 |
| DSPy | [GitHub](https://github.com/stanfordnlp/dspy) | `stanfordnlp/dspy` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 优化集泄漏、评估过拟合 |
| unstructured | [GitHub](https://github.com/Unstructured-IO/unstructured) | `Unstructured-IO/unstructured` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 文档上传、OCR/格式额外依赖 |
| Docling | [GitHub](https://github.com/docling-project/docling) | `docling-project/docling` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 模型下载、硬件、论文条件不可外推 |
| sentence-transformers | [GitHub](https://github.com/huggingface/sentence-transformers) | `huggingface/sentence-transformers` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 模型 revision、语言与许可证 |
| FlagEmbedding / BGE | [GitHub](https://github.com/FlagOpen/FlagEmbedding) | `FlagOpen/FlagEmbedding` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 模型 revision、语言/长文本设置 |
| FAISS | [GitHub](https://github.com/facebookresearch/faiss) | `facebookresearch/faiss` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | 非数据库；ACL/持久化由上层负责 |
| Qdrant | [GitHub](https://github.com/qdrant/qdrant) | `qdrant/qdrant` | server/client 同时 pin | 2026-07-17 | 活跃；发布前复核 | filter、备份、multi-vector 的实际版本能力 |
| Milvus | [GitHub](https://github.com/milvus-io/milvus) | `milvus-io/milvus` | server/client 同时 pin | 2026-07-17 | 活跃；发布前复核 | 运维、索引参数、隔离与恢复 |
| Weaviate | [GitHub](https://github.com/weaviate/weaviate) | `weaviate/weaviate` | server/client 同时 pin | 2026-07-17 | 活跃；发布前复核 | 模块、向量化路径、multi-vector 版本门槛 |
| pgvector | [GitHub](https://github.com/pgvector/pgvector) | `pgvector/pgvector` | extension + PG version pin | 2026-07-17 | 活跃；发布前复核 | 数据库升级、HNSW/过滤基准 |
| Ragas | [releases](https://github.com/vibrantlabsai/ragas/releases) | `vibrantlabsai/ragas` | release tag + lock hash | 2026-07-17 | 活跃；发布前复核 | judge/model/prompt 漂移，见 [[18 RAG 评估\|RAG 评估]] |
| DeepEval | [GitHub](https://github.com/confident-ai/deepeval) | `confident-ai/deepeval` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | judge 费用、数据发送与 CI 稳定性 |
| Phoenix | [GitHub](https://github.com/Arize-ai/phoenix) | `Arize-ai/phoenix` | release tag + SHA | 2026-07-17 | 活跃；发布前复核 | trace 脱敏、留存和访问控制 |
| ColPali | [GitHub](https://github.com/illuin-tech/colpali) | `illuin-tech/colpali` | model/repo revision + SHA | 2026-07-17 | 活跃；发布前复核 | VLM 许可、页图像成本、patch 索引 |
| ColQwen2 | [GitHub](https://github.com/vidore/colqwen2) | `vidore/colqwen2` | model/repo revision + SHA | 2026-07-17 | 活跃；发布前复核 | 同上；tokenizer/图像预处理也要锁 |
| Cognita（历史归档） | [GitHub](https://github.com/truefoundry/cognita) | `truefoundry/cognita` | 仅历史 commit | 2026-07-17 | **2026-07-17 核验：2026-03-13 已归档** | 只作迁移/历史参考，不作新项目候选 |
| fastRAG（历史） | [GitHub](https://github.com/IntelLabs/fastRAG) | `IntelLabs/fastRAG` | 仅历史 commit | 2026-07-17 | **2026-01-12 已归档** | 官方声明不再维护/接受补丁；不作新项目候选 |

## 关键事实

- `Cognita` 与 `fastRAG` 的官方 GitHub 仓库均显示 archived/read-only；前者归档日为 2026-03-13，后者为 2026-01-12。它们仍可解释历史架构，不能被描述为当前活跃推荐。
- Docling 的项目归属、实现和 benchmark 会更新；若引用任何精度数值，必须同时给出论文/报告、数据集、任务定义、版本、硬件和评测日期。没有这些条件时不报数字。
- multi-vector / late interaction 的“支持”是具体 server、client、模型与索引配置的组合能力，不是产品名级承诺。升级任何一项都要重跑质量和 P95 基准。
- source manifest 与 benchmark 互补：前者防止拿错来源/版本，后者防止把别人的 workload 结论移植到自己的 ACL、语言和 SLO。

## 工业界实践

**1) 先定义验收集，再挑组件。** 将真实文档按类型、语言、权限和难度分桶，每题存 gold evidence、文档 revision、预期拒答和（如有）业务 receipt。选 parser、embedding、向量库或 reranker 时都跑同一份冻结集，见 [[18 RAG 评估|RAG 评估]]。

**2) ACL 进入检索路径。** 用户/租户过滤条件应在候选生成阶段就生效，并把授权 policy revision、filter、命中文档和拒绝原因留进 trace。仅在生成前隐藏一部分文字，不能证明没有越权召回。

**3) SLO 必须在本地 workload 测。** 对每个候选组合记录数据量、chunk/page 策略、索引参数、机器/加速器、并发、冷/热缓存、P50/P95、失败率和每 query 成本。平均延迟、单次 notebook 或供应商宣传都不是承诺。

**4) 发布配置而不是“库名”。** 发布的最小单元是 `parser revision + model revision + index build revision + vector server/client + prompt + ACL policy + evaluation manifest`。这样 rollback 时能回到可重建状态，而不是只说“我们用了某框架”。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1：RAG 生态怎么选？**
先按层定位：解析、表示/索引、召回/重排、生成/引用、评估/可观测、编排。再用文档类型、语言、ACL、多模态和 SLO 把候选缩小，最后在冻结的本地集上比较 Recall@k、citation、终态、P95 与成本；不按仓库热度或功能堆叠选。

**Q2：为什么 embedding 示例不能用英文模型配中文问题？**
模型训练语言/分词与 query 语言会直接影响表示空间。应使用经验证的多语模型或语言一致模型，并按语言分桶报告检索指标；能输出一个向量不等于检索质量合格。

**Q3：视觉 late interaction 如何兼顾质量和延迟？**
锁模型、tokenizer、client/server 版本后，先用廉价粗召回收窄候选，再对候选页做 patch 级 MaxSim 精排；同时记录索引向量数/磁盘、构建时间和 query P95。只跑全库多向量精排无法证明可控的 SLO。

**Q4：为什么 source manifest 和 benchmark 都需要？**
manifest 解决来源、owner、tag/commit、维护与约束；benchmark 解决自己的文档、权限、硬件和流量下的质量/成本。只做其中一个，要么不可复现，要么无法证明适用。

## 知识拓展

- 解析质量会沿链路放大：标题/表格/页码丢失会破坏分块、召回和 citation 回链，因此 parser 也要以 evidence 可定位率验收，而不只看文本是否“读出来”。
- 多语言长文检索可用 MLDR 一类基准暴露长度和语言差异，但上线仍要加入自己的术语、混合语言、权限拒答与版本更新样本。
- RAG 与 [[36 Agentic RAG|Agentic RAG]] 相遇后，框架选型还要评价工具 schema、授权、幂等和 receipt；答案看似正确仍不足以证明任务完成。

## 来源

- 本篇项目的官方 URL、owner/repo、pin 规则、核验日、维护状态与约束见上方 source manifest。维护状态以官方仓库页面为准，采用前需再次核验。
- [Cognita archive](https://github.com/truefoundry/cognita)（2026-03-13）与 [fastRAG archive](https://github.com/IntelLabs/fastRAG)（2026-01-12）：均为历史条目，不作为新项目推荐。
- [ColPali 官方实现](https://github.com/illuin-tech/colpali)（页面 patch multi-vector、MaxSim 与设备/内存相关实现）及 [Qdrant 官方 multivector / late-interaction 文档](https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/)（多向量的 RAM/写入成本与 dense-first、late-rerank 两阶段模式）；[ColPali 论文](https://arxiv.org/abs/2407.01449)、[MLDR / M3-Embedding 论文](https://aclanthology.org/2024.findings-acl.137/)。
