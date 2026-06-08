[[10 重排序 Reranking|重排序 Reranking]] 的本质是:召回之后,用一个**更准但更慢**的 cross-encoder 模型把候选文档重新精排,只把 top-5 喂给生成。它和召回是分工:召回(bi-encoder/向量/BM25)负责**从百万级里捞出几十上百个候选**(要快、要召回率高),重排负责**在这几十个里挑出真正最相关的几个**(要准、可以慢)。这就是 **两阶段 retrieve-then-rerank** 架构。

它接在 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 之后、[[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] 之前,是把检索质量"收口"的关键一环:召回阶段为了不漏召得宽,难免带噪音;重排用 query 和 doc 的**深度交互**把噪音压下去,直接决定喂进生成的上下文有多干净。

## 本质:为什么要两阶段

[[04 Embedding 与向量数据库|Embedding 与向量数据库]] 的向量检索用 **bi-encoder(双塔)**:query 和 doc **各自独立**编码成向量,事后算余弦相似度。好处是 doc 向量可**离线预算 + 建 ANN 索引**,在线只编码 query,毫秒级扫百万库。代价:query 和 doc 在编码时**互相看不到**,丢了 token 级交互,精度有上限。

**cross-encoder** 反过来:把 `[query | SEP | doc]` **拼成一条**喂进同一个 Transformer,query 的每个 token 能和 doc 的每个 token 做**交叉注意力**,直接输出一个相关性分。精度高得多,但**每个 query-doc 对都得现场过一遍模型**,无法预算、无法建索引——扫百万库不可能。

![[bi-vs-cross-encoder.png]]

结论是显而易见的分工:**别让 cross-encoder 扫全库,只让它精排 bi-encoder 海选出的小候选集**。

像招聘:**HR 海选**靠简历关键词从上千份里快速筛出 100 份(快、宁可多放进来也别漏掉好苗子,对应召回);**面试官精排**对这 100 人逐个深聊、当面交互,挑出最终 5 个(慢、准,对应 cross-encoder 重排)。没人会让面试官把上千人全部面一遍(cross-encoder 扫全库不可行),也没人敢只看简历就直接发 offer(跳过重排直接喂向量 top-k)。

![[重排序 Reranking.png]]

典型配置:bi-encoder/混合检索海选 **top-100** → cross-encoder 精排 → 取 **top-5** 喂生成。100 次 cross-encoder 前向是可接受的延迟,百万次则不可能。

## 机制

### 两阶段漏斗
1. **召回(recall-oriented)**:[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 或纯向量,从全库捞 top-50~200。指标是**召回率**——宁可多召也别漏掉正确答案,因为漏了重排也救不回。
2. **精排(precision-oriented)**:cross-encoder 对每个候选打分,重排,取 top-k。指标是**精度/排序质量**(nDCG)——把真正相关的顶到最前。

为什么不一步到位:召回要快要全(只能用 bi-encoder),精排要准(只能用 cross-encoder),两个目标的最优模型不同,所以分两阶段各用各的。

### 2026 主流重排模型
- **Cohere Rerank**:闭源 API,工业界最常用的开箱即用 reranker,多语言强。
- **bge-reranker(BAAI)**:开源,`bge-reranker-base/large/v2-m3`,中文社区默认首选,可自托管。
- **Jina reranker**:`jina-reranker-v2`,开源 + API,长文档、多语言。
- **mxbai-rerank(Mixedbread)**:开源 cross-encoder reranker,轻量。
- **LLM-as-reranker / RankGPT**:直接让 LLM 给候选排序。**RankGPT** 让 GPT 输出候选的**排列(permutation)**而非逐个打分,出处 Sun, Yan, Ma, Wang, Ren, Chen, Yin, Ren《Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents》(arXiv:2304.09542, EMNLP 2023 Outstanding Paper);GPT-4 在 TREC 上平均超过全量微调的 monoT5-3B 约 2.7 nDCG,且可把排序能力**蒸馏**进小模型。LLM reranker 最准也最贵最慢,适合候选少、质量要求极高的场景。

### MMR 去冗余
重排只保证"每篇都相关",不保证"几篇之间不重复"。若 top-5 是 5 篇近重复文档,信息密度极低还挤占上下文。**MMR(Maximal Marginal Relevance)**在选下一篇时同时奖励"与 query 相关"、惩罚"与已选文档相似",`MMR = λ·rel(d,q) - (1-λ)·max sim(d, 已选)`,在相关性和多样性间权衡。常作为重排后的一道去冗余工序,尤其和 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 配合时(避免选进同一父文档的多个相邻句子)。

**MMR 手算(λ=0.7,3 候选选 top-2)**。设相关性 $\mathrm{rel}(A)=0.90,\ \mathrm{rel}(B)=0.85,\ \mathrm{rel}(C)=0.60$,候选间相似度 $\mathrm{sim}(A,B)=0.95$(A、B 近重复)、$\mathrm{sim}(A,C)=0.30$。第一轮无已选,只剩相关项 $\lambda\cdot\mathrm{rel}$:$A=0.63,\ B=0.595,\ C=0.42$,**A 当选**。第二轮对 A 算多样性惩罚:

$$\mathrm{MMR}(B)=0.7\times0.85-0.3\times0.95=0.595-0.285=0.31$$
$$\mathrm{MMR}(C)=0.7\times0.60-0.3\times0.30=0.42-0.09=0.33$$

B 虽然比 C 更相关(0.85 vs 0.60),但和已选的 A 几乎重复(sim 0.95)被狠扣,**最终选 C 而非 B**——这正是 MMR 压近重复、换来覆盖面的体现。

## 可跑最小代码

```python
# 两阶段:bi-encoder 海选 top-N → cross-encoder 精排 top-k →(可选)MMR 去冗余
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")     # cross-encoder:拼接打分

def retrieve_then_rerank(query, vstore, embed, top_n=100, top_k=5):
    # ① 召回:bi-encoder 向量海选,要快要全(召回率优先)
    candidates = vstore.search(embed(query), k=top_n)  # 100 个候选,带噪音没关系

    # ② 精排:cross-encoder 对每个 (query, doc) 现场打分(精度优先)
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)                   # 100 次前向,可接受
    ranked = [c for _, c in sorted(zip(scores, candidates),
                                   key=lambda x: x[0], reverse=True)]
    return ranked[:top_k]                              # 只把 top-5 喂生成

def mmr(query_vec, cand_vecs, candidates, k=5, lam=0.7):
    # MMR 去冗余:相关性 vs 多样性权衡,避免 top-k 全是近重复
    selected, remaining = [], list(range(len(candidates)))
    while remaining and len(selected) < k:
        best = max(remaining, key=lambda i:
            lam * cos(query_vec, cand_vecs[i])
            - (1 - lam) * max([cos(cand_vecs[i], cand_vecs[j]) for j in selected], default=0))
        selected.append(best); remaining.remove(best)
    return [candidates[i] for i in selected]
```

要点:① 召回 `top_n=100` 大、精排 `top_k=5` 小,这是漏斗的灵魂;② cross-encoder 输入是 **(query, doc) 对**,现场算,不能预存;③ cross-encoder 只跑 100 次而非全库,延迟可控;④ MMR 在"已经相关"的候选里再去重复,提升上下文信息密度。

## 对比表

| 维度 | bi-encoder(召回) | cross-encoder(重排) | LLM/RankGPT(重排) |
|---|---|---|---|
| 编码方式 | query/doc 各自独立 | 拼接联合 | 拼接,输出排列 |
| query-doc 交互 | 无(事后算余弦) | token 级交叉注意力 | 全注意力 + 推理 |
| 精度 | 中 | 高 | 最高 |
| 速度 | 快(可预算/索引) | 慢(每对现算) | 最慢 |
| 能扫全库? | 能 | 不能 | 不能 |
| 用在 | 召回海选 top-100 | 精排 top-5 | 候选极少、质量极高 |

## 何时用 / 坑

**该上重排**:几乎所有讲究质量的生产 RAG 默认配置。召回带噪、top-k 里相关文档没排在最前、或 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] / [[07 查询变换 Query Transformation|查询变换 Query Transformation]] 扩大了候选集后,都需要重排收口。它是把"召得多"变成"喂得准"的标准手段。

**坑**:
- **跳过重排直接喂 top-k 向量结果**:向量召回的 top-5 排序常不够准,真正最相关的可能排在第 8、第 15。不重排,生成层就吃到次优上下文。
- **召回 top_n 设太小**:重排只能在候选集内重排,候选里就没正确文档,重排无力回天。召回要给够(top-50~100),重排才有发挥空间。
- **重排延迟**:cross-encoder 对 100 个候选打分有实打实的延迟,LLM reranker 更甚。延迟敏感场景控制 top_n,或用小 reranker(bge-base / mxbai)。
- **只相关不多样**:不接 MMR,top-5 可能是 5 篇近重复,信息量低。需要覆盖面时加 MMR。
- **长文档截断**:cross-encoder 有最大长度,长 doc 被截会丢关键段。配合 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]]——重排在小粒度(句/段)上做,再回扩到父文档喂生成。
- **域不匹配**:通用 reranker 在专业域(法律/医疗/代码)可能不准,必要时微调或换域内模型。

## 关键事实

- 重排 = 召回后用 **cross-encoder 精排**:query+doc 拼一起过 Transformer 打分,比 bi-encoder 准但慢。
- **两阶段 retrieve-then-rerank**:bi-encoder 海选 top-100(快、召回率优先)→ cross-encoder 精排 top-5(准、精度优先)。分两阶段因为召回和精排的最优模型不同。
- bi-encoder **独立编码**可预算/建索引但丢交互;cross-encoder **拼接联合编码**保留 token 级交互但每对现算、不能扫全库。
- 2026 主流:**Cohere Rerank**(API)、**bge-reranker**(BAAI,中文默认)、**Jina reranker**、**mxbai-rerank**;**RankGPT**(arXiv:2304.09542, EMNLP 2023 Outstanding)= LLM 输出候选排列,最准最贵。
- **MMR** 在重排后去冗余:`λ·相关 - (1-λ)·与已选最大相似`,避免 top-k 全近重复。
- 位置:[[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 召回 → 重排收口 → [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]];与 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 配合(小粒度重排 + 回扩父文档)。

## 工业界实践

重排是生产 RAG 里**性价比最高的单点改进**之一:不动召回、不动索引,加一个 reranker 就能把 nDCG/命中率拉一截,因此几乎是"讲究质量就默认上"的标配。

**主流 reranker(2026 现状,具体名 + 定位)**
- **Cohere Rerank 3.5**(2024-12 发布):闭源 API,工业界最常用的开箱即用 reranker;上下文 4096,支持 100+ 语言,在 BEIR 及金融/电商等域 SOTA,Cohere 内测称比混合检索好 ~23%、比 BM25 好 ~31%。已上 Amazon Bedrock / Azure AI Foundry / Oracle Cloud,企业无需自托管。
- **Qwen3-Reranker**(2025-06,阿里):开源,0.6B / 4B / 8B 三档,4B 拿到 MTEB-R 69.76、8B 中文 CMTEB-R 77.45,**中文 + 多语言场景 2025 年的新首选**,可自托管。
- **bge-reranker(BAAI)**:`bge-reranker-base/large/v2-m3`,开源、轻量、易微调,中文社区长期默认基线,延迟敏感场景常用 base。
- **jina-reranker-v2**:开源 + API,长文档、多语言、还支持代码与函数调用排序。
- **mxbai-rerank(Mixedbread)**:开源 cross-encoder,主打轻量低延迟。
- **LLM-as-reranker / RankGPT**:让 LLM 输出候选**排列**(见正文 arXiv:2304.09542),最准最贵最慢,适合候选极少、质量要求极高(如法律/医疗终审)。
- **ColBERT / 后期交互(late interaction)**:介于 bi/cross 之间——doc 的 token 级向量可**预存建索引**,在线只算 query token 与 doc token 的 MaxSim,兼顾精度与可扩展;生产里 ColBERTv2 / PLAID、以及向量库内置的多向量检索(Qdrant multivector、Vespa)是"想要 cross-encoder 精度又怕它扫不动"的折中。

**reranker 选型卡**(按场景挑具体模型):

| 场景 | 选什么 | 为什么 |
|---|---|---|
| 不想自托管、要开箱即用 | **Cohere Rerank 3.5**(API) | 100+ 语言、BEIR/金融电商 SOTA,已上 Bedrock/Azure,企业零运维 |
| 中文 / 多语言 + 可自托管 | **Qwen3-Reranker**(0.6/4/8B) | 2025 中文新首选,CMTEB-R 77.45,可微调,数据可控 |
| 延迟敏感 / 轻量基线 | **bge-reranker-base** 或 **mxbai** | 小、快、易微调,延迟敏感场景常用 base |
| 候选极少 + 质量要求极高 | **LLM-as-reranker / RankGPT** | listwise 输出排列最准,但最贵最慢,法律/医疗终审才值 |
| 想要 cross-encoder 精度又怕扫不动 | **ColBERT / 后期交互** | doc token 向量可预存建索引,在线只算 MaxSim,精度与可扩展折中 |
| 专业域(法律/医疗/代码)不准 | 域内 (query,pos,neg) 三元组**微调** bge/Qwen3 | 通用 reranker 语义偏移大,微调比换模型更划算 |

**典型架构与配置**
```
混合检索(BM25 + 向量,RRF 融合)→ top-100~200 候选
   → cross-encoder rerank(Cohere / bge / Qwen3)→ top-5~10
   → (可选) MMR 去冗余 → grounded 生成
```
- 框架里几乎都是一行接入:LlamaIndex `SentenceTransformerRerank` / `CohereRerank` 作为 `node_postprocessor`;LangChain `ContextualCompressionRetriever` 套 `CohereRerank`;Haystack `Ranker` 组件。

**规模化(召回/延迟/成本/索引)**
- **延迟模型**:cross-encoder 延迟 ≈ `候选数 × 单次前向`,与 `top_n` 线性。控延迟的杠杆:① 砍 `top_n`(100→50);② 换小 reranker(bge-base / mxbai);③ batch 推理 + GPU;④ ONNX/TensorRT 加速;⑤ 长候选截断到 reranker 上下文内。
- **索引选型(召回阶段)**:召回要"宁多勿漏"给重排留空间。**HNSW** 高召回、`efSearch` 可在线调,延迟敏感场景主选;**IVF / IVF-PQ** 内存省但召回受量化损伤,大库省成本时用,需把 `nprobe` 调大保召回;**标量/二值量化 + 全精度重排(rescore)** 是省内存又保精度的常用组合。无论哪种,**召回的损失重排救不回**,所以索引参数偏召回侧。
- **成本**:闭源 API(Cohere)按调用计费,大流量下重排成本可观,常对热 query 做**语义缓存**;开源自托管(bge/Qwen3)换 GPU 成本,但可控、可微调。

**评估与可观测**
- 离线:用 **nDCG@k / MRR / Recall@k / Hit@k** 对比"重排前后"在自建标注集上的提升,这是证明 reranker 值不值得加的直接证据。
- 端到端:**Ragas** 的 `context_precision`(召回的料里相关比例,直接受重排影响)、`context_recall`;**TruLens / LangSmith / Phoenix** trace 每个候选的 rerank 分,定位"正确文档排第几"。
- 常驻基准:**BEIR**(零样本检索/重排)、**MTEB-Reranking**、中文 **C-MTEB**。

**踩坑与最佳实践**
- **召回 top_n 给够**:正确文档不在候选集,重排回天乏术。先保 Recall@top_n,再谈 rerank。
- **重排粒度用小、喂生成回扩父块**:在句/段级 rerank(精度高、不超长),命中后回扩到父文档喂生成(信息全),配 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]]。
- **接 MMR 去冗余**:top-5 全近重复会浪费上下文,需覆盖面时加 MMR。
- **域不匹配就微调**:通用 reranker 在法律/医疗/代码可能不准,bge/Qwen3 可用域内 (query, pos, neg) 三元组微调。
- **别同时多个 reranker 串联**:边际收益递减、延迟叠加,通常一个够。

## 面试高频

**Q1:bi-encoder 和 cross-encoder 有什么区别?为什么 RAG 要两阶段?**
标准答:bi-encoder(双塔)把 query 和 doc **各自独立编码**成向量,事后算余弦,doc 向量可离线预算 + 建 ANN 索引,毫秒级扫百万库,但编码时 query/doc 互相看不到、丢 token 级交互,精度有上限;cross-encoder 把 `[query|SEP|doc]` **拼成一条**过 Transformer,token 级交叉注意力直接出相关分,精度高得多但**每对都得现场过模型、无法预存建索引**,扫不动全库。所以分两阶段:bi-encoder 海选 top-100(快、召回率优先)→ cross-encoder 精排 top-5(准、精度优先),各用各最优的模型。
- 追问"cross-encoder 为什么不能建索引?":它的分依赖 query 和 doc 的**联合表示**,没有"独立的 doc 向量"可预存;换一个 query,所有分都要重算。
- 陷阱:有人答"cross-encoder 更准所以全用它"——错,扫百万库要百万次前向,延迟不可接受。

**Q2:重排放在 RAG 流程哪一步?为什么不一步到位?**
标准答:召回([[08 混合检索 Hybrid Search|混合检索 Hybrid Search]])之后、生成([[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]])之前,把检索质量"收口"。不一步到位是因为召回(要快要全,只能 bi-encoder)和精排(要准,只能 cross-encoder)的最优模型不同。

**Q3:重排只保证相关,不保证什么?怎么补?**
标准答:不保证候选之间**不重复**——top-5 可能是 5 篇近重复,信息密度低还挤占上下文。用 **MMR**(`λ·相关 - (1-λ)·与已选最大相似`)在相关性和多样性间权衡去冗余。
- 追问"λ 怎么取?":λ→1 纯相关(可能冗余),λ→0 纯多样(可能跑题),通常 0.5~0.7。

**Q4:召回 top_n 设多大?设太小会怎样?**
标准答:典型 50~200。太小则正确文档可能根本不在候选集里,重排只能在候选内重排、无力回天——**重排救不回召回的漏召**。所以 top_n 偏召回侧,精度交给 reranker。
- 追问"那为什么不无限大?":每个候选都要过一次 cross-encoder,top_n 越大延迟越高,需在召回率和延迟间权衡。

**Q5:ColBERT 这种"后期交互"和 bi/cross 是什么关系?**
标准答:介于两者之间。bi-encoder 把 doc 压成**单个向量**(交互全丢),cross-encoder **全 token 联合**(算不动全库),ColBERT 折中:doc 保留**每个 token 的向量**可预存建索引,在线只算 query token 与 doc token 的 MaxSim——保住部分 token 级交互又能扩展。
- 陷阱:问"ColBERT 算 bi 还是 cross"——都不是,是 late interaction 第三类。

**Q6(场景题):向量检索 top-5 里正确答案排第 8,怎么办?**
标准答:① 扩大召回 top_n 让第 8 进候选;② 加 cross-encoder 重排把它顶到前面;③ 检查是否该上混合检索补关键词召回。能指出"先保召回进候选、再靠重排提排序"是关键。

## 知识拓展

**进阶与前沿(带年份)**
- **RankGPT(2023,EMNLP Outstanding)**:LLM 输出候选**排列**而非逐个打分,GPT-4 在 TREC 上超全量微调的 monoT5-3B,且可**蒸馏**进小模型(见正文)。开启了 LLM listwise 重排一脉。
- **后期交互演进**:**ColBERT(2020)→ ColBERTv2(2021,残差压缩省存储)→ PLAID(2022,加速检索)**;2024–2025 多向量检索被 Qdrant / Vespa / Weaviate 原生支持,late interaction 从"研究"走进"向量库一等公民"。
- **重排与 embedding 的统一**:**Qwen3-Embedding/Reranker(2025-06)**、**Cohere Rerank 3.5(2024-12)** 都把多语言/推理能力做进同一基座,embedding 与 rerank 同源,工程上更一致。
- **listwise / setwise LLM 重排(2024–2025)**:相比 pointwise(逐对打分),listwise(一次看一组排序,如 RankGPT)、setwise 更贴近排序本质,准确率更高但更贵;滑动窗口 + 二分等技巧降其延迟。
- **重排即压缩**:CRAG 的 decompose-then-recompose、LongLLMLingua 等把"重排 + 过滤无关句"合并,既排序又压上下文,服务 [[20 上下文工程|上下文工程]]。

**底层联系**:bi-encoder 的余弦打分本质是**向量点积/归一化相似度**,见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]]——理解"为什么独立编码 + 点积会丢交互",就理解了为什么需要 cross-encoder。

**边界与反模式**
- **反模式一:跳过重排直接喂向量 top-k**——向量召回排序常不够准,真正最相关的可能排第 8/15,生成层吃到次优上下文。
- **反模式二:重排但召回 top_n 太小**——候选里没正确文档,重排是无米之炊。两个坑常一起犯。
- **反模式三:用通用 reranker 硬套专业域**——法律/医疗/代码语义偏移大,不微调会跑偏。
- **边界**:候选已经很干净(强混合检索 + 精确过滤)时重排边际收益小;延迟极敏感(<50ms)的实时场景,cross-encoder 的逐对前向可能就是预算瓶颈,得退到小 reranker 或 late interaction。

**相关链接**:上游 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] 提供候选、[[07 查询变换 Query Transformation|查询变换 Query Transformation]] 扩大候选后更需重排收口;下游接 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]];与 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 配合(小粒度重排 + 回扩父文档);也是 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 中"相关性过滤"的工程化对应物。

## 来源
- Sun, Yan, Ma, Wang, Ren, Chen, Yin, Ren.《Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents》(RankGPT). arXiv:2304.09542, EMNLP 2023(Outstanding Paper).
- Khattab, Zaharia.《ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT》(后期交互,介于 bi/cross 之间). arXiv:2004.12832, SIGIR 2020.
