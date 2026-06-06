[[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|进阶索引：Contextual Retrieval、RAPTOR、Late Chunking]] 是 2024 年三项针对 [[03 分块策略 Chunking|分块策略 Chunking]] 根本病的索引改良:**一个 chunk 被孤立切出来后,就丢掉了它在全文里的上下文**——「它」指代谁、这段属于哪一章、放在什么背景下,统统没了,导致 embedding 编不准、召回失败。三者用三条不同路子给 chunk **补回上下文**:Contextual Retrieval 用 LLM 写前缀、RAPTOR 建摘要树、Late Chunking 改 embedding 顺序。它们都发生在「建索引」阶段,是 [[04 Embedding 与向量数据库|Embedding 与向量数据库]] 之上的增强层。

## 本质:共同的敌人是「孤立 chunk 丢上下文」

[[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 里最隐蔽的一类失败:文档说「2023 年公司营收增长 3%」,但「公司」是哪家、相对哪个基期,写在上一段——切块后这个 chunk 单独看**信息不全**。query「ACME 公司 2023 营收增速」来了,这个 chunk 因为没出现「ACME」,相似度不够,**召不回**。问题不在 query、不在模型,在于**切块销毁了上下文**。

三种进阶索引是对这个共同敌人的三种解法,正交、可叠加:

| 方法 | 怎么补上下文 | 何时补 | 额外成本 |
|---|---|---|---|
| Contextual Retrieval | LLM 给每块生成「该块在全文中的位置/背景」前缀 | 建库时,逐块 | 每块一次 LLM(可缓存) |
| RAPTOR | 递归聚类+摘要,建一棵从细节到全局的树 | 建库时,递归 | 多轮聚类+LLM 摘要 |
| Late Chunking | 先整篇 embedding 让每 token 看过全文,再切 | embedding 阶段改顺序 | 几乎为零(只改流程) |

## 机制一:Contextual Retrieval(Anthropic, 2024-09)

来自 **Anthropic 2024 年 9 月工程博客《Contextual Retrieval》**。做法朴素而有效:**建库时,对每个 chunk,把「整篇文档 + 这个 chunk」喂给一个便宜 LLM(如 Claude Haiku),让它生成一两句话说明「这个 chunk 在全文里讲的是什么背景」,把这段上下文前缀拼到 chunk 前面,再做 embedding 和 BM25 索引。** 于是上例的 chunk 被改写成「[本段出自 ACME 公司 2023 年报,讨论营收] 2023 年公司营收增长 3%」——「ACME」「2023 年报」进了向量,召回就成了。

两个子技术:**Contextual Embeddings**(带前缀的块去 embedding)+ **Contextual BM25**(带前缀的块也进 BM25 关键词索引)。

**核验数字**(Anthropic 官方博客):
- 仅 Contextual Embeddings:top-20 检索失败率降 **35%**。
- Contextual Embeddings + Contextual BM25:降 **49%**。
- 再加 reranking([[10 重排序 Reranking|重排序 Reranking]]):降 **67%**(失败率从 5.7% → 1.9%)。

关键工程点:**prompt caching**——整篇文档作为前缀对该文档的所有 chunk 复用,缓存命中后逐块生成上下文的成本被压到很低。这是它能落地的前提。本质上它把 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]](dense+BM25)和 rerank 当作搭档,三者叠加才到 67%。

## 机制二:RAPTOR(Sarthi et al., ICLR 2024)

**RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**(**Sarthi et al. 2024, arXiv:2401.18059, ICLR 2024**,作者含 Christopher D. Manning)。解决的是另一个层面的上下文缺失:**普通 RAG 只能召回零散的短 chunk,无法回答需要跨越整篇文档、综合全局的问题**(「这本书的主旨」「全文论证的演进」)。

做法是**自底向上递归建树**:
1. 把文档切成 chunk(叶子,L0),各自 embedding。
2. 对这些向量**聚类**(论文用软聚类 GMM,允许一个块属于多簇)。
3. 每个簇用 LLM 生成一段**摘要**,这些摘要成为上一层节点(L1),再各自 embedding。
4. 对 L1 摘要再聚类、再摘要 → L2 …… **递归直到收敛成一个根**。

得到一棵树:底层是细节,越往上越抽象、越全局。**检索时跨层一起检索**(论文的 collapsed-tree:把所有层的节点放进同一个池子做最近邻)——query 既能命中 L0 的具体事实,也能命中 L1/L2 的主题/全局摘要,一次拿到「局部细节 + 全局脉络」。论文报告:RAPTOR + GPT-4 在 QuALITY 基准上比此前最佳**绝对准确率提升约 20%**,在需要多步推理的长文 QA 上是 SOTA。

![[进阶索引-RAPTOR树.svg]]

## 机制三:Late Chunking(Jina AI, 2024)

**Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models**(Jina AI,**Günther et al. 2024, arXiv:2409.04701**)。它最优雅——**不改数据、不加 LLM,只改 embedding 和切块的先后顺序**。

对比 naive:
- **Naive Chunking**:先把文档**切**成块,再把每块**各自独立**喂进 embedding 模型 → 每块的向量是「盲编码」,不知道别的块,跨块指代/上下文丢失。
- **Late Chunking**:先把**整篇文档**(在 embedding 模型的长上下文窗口内,如 8192 token)一次性过 Transformer,得到**每个 token 的向量**——因为有自注意力,**每个 token 的向量都「看过」全文**;**然后才**按 chunk 边界把 token 序列切开,对每块的 token 向量做 **mean-pooling** 得到块向量。于是每个块向量天然携带全文上下文,而它仍是一个个独立的、可建库的块向量。

「late」就是指:**pooling(切块)发生在 token embedding 之后**,而非之前。免训练、几乎零额外成本,直接换上长上下文 bi-encoder(如 jina-embeddings-v2/v3、bge-m3 这类长窗口模型)即可。局限:受 embedding 模型上下文窗口约束(超长文档仍需分段),且要求模型本身支持输出 token 级向量。

![[Late Chunking 对比.svg]]

## 可跑最小代码

```python
# ① Contextual Retrieval:逐块用便宜 LLM 加上下文前缀(prompt cache 整篇文档)
def contextualize(doc, chunk, llm):
    ctx = llm(f"<document>{doc}</document>\n"
              f"这是其中一个片段:<chunk>{chunk}</chunk>\n"
              "用一两句话说明该片段在全文中的背景,只输出这段上下文。")
    return ctx.strip() + "\n\n" + chunk           # 前缀拼回,再去 embed + BM25

# ③ Late Chunking:整篇过模型拿 token 向量,再按块边界池化
import numpy as np
def late_chunking(text, spans, model_token_emb):
    # spans: 每块在 token 序列中的 [start, end);token_emb: 整篇一次前向得到
    token_emb = model_token_emb(text)             # (L, d),每行已含全文上下文
    return [token_emb[s:e].mean(axis=0) for s, e in spans]  # 块向量=token 向量均值
# 对照 naive:[model_token_emb(text[s:e]).mean(0) for s,e in spans] —— 每块单独前向,丢上下文
# RAPTOR 见 github.com/parthsarthi03/raptor;Late Chunking 见 github.com/jina-ai/late-chunking
```

## 对比表

| 维度 | Contextual Retrieval | RAPTOR | Late Chunking |
|---|---|---|---|
| 补的是什么上下文 | 该块在全文的背景说明 | 跨层的全局/主题摘要 | 每块的全文 token 级上下文 |
| 建库开销 | 逐块一次 LLM(可缓存) | 递归聚类+多轮 LLM 摘要 | 几乎为零(只改顺序) |
| 是否引入新内容 | 是(LLM 生成前缀) | 是(摘要节点) | 否(仍是原块) |
| 最擅长 | 召回失败、关键词盲点 | 全局/综合性长文问答 | 跨块指代、上下文连续性 |
| 依赖 | 便宜 LLM + prompt cache | LLM + 聚类 | 长上下文 token-level 模型 |
| 出处 | Anthropic 2024-09 博客 | arXiv:2401.18059 ICLR24 | arXiv:2409.04701 Jina |

## 何时用 / 坑

- **三者正交,可叠加**:Contextual Retrieval 官方就是配 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]] + [[10 重排序 Reranking|重排序 Reranking]] 才到 67%;Late Chunking 可作为 RAPTOR 叶子块的 embedding 方式。
- **Contextual Retrieval 不开 prompt cache 会很贵**:逐块带整篇文档进 LLM,不缓存成本爆炸。这是落地前提,务必确认。
- **RAPTOR 建索引重、增量更新难**:文档一变,聚类和摘要树要重建;摘要质量受 LLM 限制(摘错=污染整层)。适合相对静态、需要全局问答的语料(书、报告);高频更新的语料慎用。
- **Late Chunking 受窗口限制**:文档超过模型上下文(如 >8192 token)仍要先分大段,跨大段的上下文又丢了;且要求模型能输出 token 级向量,不是所有 embedding API 都给。
- **别拿进阶索引救烂分块**:这些是「锦上添花」,基础的 [[03 分块策略 Chunking|分块策略 Chunking]]、[[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 没做好,先补那些。
- **评估要量化**:每项都声称提升,但在你的语料上未必。上之前先用 [[18 RAG 评估|RAG 评估]] 测召回基线,叠加后对比。
- 这些索引增强的最终目标和 [[36 Agentic RAG|Agentic RAG]] 一致——让检索拿到的证据更全、更准,只是 Agentic 在「检索时」补救,进阶索引在「建库时」预防。

## 工业界实践

进阶索引在生产里**不是三选一,而是按语料形态挑一条主线、必要时叠加**。下面按三种方法各自的落地形态、成本结构与踩坑展开。

### Contextual Retrieval 的工业落地

这是三者里**最容易上生产**的:它只在 ingestion 管线里加一步「逐块 LLM 加前缀」,不改检索器、不改索引结构,产出的仍是普通带前缀的 chunk,任何向量库 + BM25 栈都能直接吃。

**典型架构(ingestion 阶段)**:
```
原始文档 → 切块(03 分块策略)
        → 逐块: prompt = [整篇文档(命中 cache) + 该 chunk] → 便宜 LLM(Claude Haiku / GPT-4o-mini)生成 50~100 token 上下文前缀
        → 前缀 + 原 chunk 一起去 ① embedding 入向量库 ② 分词入 BM25 倒排
检索阶段:dense + sparse 两路召回(08 混合检索)→ RRF 融合 → reranker 收口(10 重排序)
```

**成本是它的命门,prompt caching 是落地前提**。不开 cache,每个 chunk 都要把整篇文档重新喂一遍 LLM,一篇 50 chunk 的长文就是 50 次「整篇文档」的输入 token,成本爆炸。开了 cache(Anthropic prompt caching / OpenAI 自动 cache),整篇文档作为公共前缀只算一次写入价,后续每块命中缓存读取价(Anthropic 缓存读取约为基础价 1/10)。Anthropic 官方给的量级:**用 Haiku + cache,生成每百万 chunk 上下文的成本约 1 美元级别**,这才让它对中大型语料可行。

**工程化要点**:
- **前缀长度要控**:50~100 token 足够,过长会稀释原 chunk 的语义、抬高 embedding 噪声,还增加 BM25 的词项膨胀。
- **缓存 TTL 与并发**:同一文档的所有 chunk 应在缓存有效窗口内连续处理,否则缓存过期重新计费;批处理时按文档分组、文档内串行/小并发,跨文档并行。
- **增量更新友好**:文档级前缀只依赖整篇文档,单篇文档更新只需重跑这一篇的前缀生成,比 RAPTOR 的全树重建轻得多。
- **生产框架**:LangChain / LlamaIndex 都有社区实现;也可以纯手写一个 ingestion 脚本(就是上面那个 `contextualize`)。AWS、Databricks 的 RAG 参考架构里已把它列为「召回增强」推荐步骤。

### RAPTOR 的工业落地

RAPTOR 在生产里**定位偏窄但不可替代**:专攻「需要综合整篇/整库的全局性问题」(法规全文要义、研报结论、整本手册的设计哲学)。它建库重、增量难,所以**只在语料相对静态 + 确有全局问答需求**时上。

- **落地形态**:LlamaIndex 有官方 `RAPTOR` pack(`llama-index-packs-raptor`),封装了「embed → GMM 软聚类 → LLM 摘要 → 递归」和 collapsed-tree 检索;也可直接用作者仓库 `parthsarthi03/raptor`。
- **聚类与摘要成本**:每层都要跑聚类(UMAP 降维 + GMM)+ 每簇一次 LLM 摘要,层数 × 簇数次 LLM 调用。摘要模型质量直接决定上层节点质量——**摘错会污染整层及其以上**,所以摘要 prompt 要强调「忠实、不引入文档外信息」,并对关键语料抽检。
- **检索策略**:生产多用 **collapsed tree**(所有层节点塞进同一向量池做 ANN),而非逐层 traversal,因为前者一次召回就能同时命中细节叶子和全局摘要,工程更简单、延迟更低。
- **增量更新是最大痛点**:语料一变,受影响的簇及其上层摘要都要重算。常见折中:对高频更新的「热」语料用普通 small-to-big,对静态的「冷」全局语料单独建 RAPTOR 树,两套索引并存、检索时合并。

### Late Chunking 的工业落地

Late Chunking 几乎零额外成本,**只要你的 embedding 模型支持长上下文 + 输出 token 级向量就能开**——它是「换个 embedding 调用方式」而非「加一个阶段」。

- **可用模型**:Jina `jina-embeddings-v3`(原生支持 late chunking,API 有开关)、`bge-m3`、`nomic-embed-text` 等长窗口 bi-encoder。Jina 的 API 直接提供 `late_chunking=true` 参数,无需自己做 token 级池化。
- **窗口约束的工程处理**:文档超过模型窗口(如 >8192 token)时,先按「大段」切到窗口内,再在大段内做 late chunking——跨大段的上下文仍会丢,所以大段边界要切在自然语义边界(章节)上,把损失降到最低。
- **常见叠加**:Late Chunking 给 small-to-big([[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]])的**子块**做 embedding,既精准召回又携带全文上下文,再返回父块给 LLM,是两项正交增强的经典组合。

### 评估、可观测与选型

- **必须先量化基线再叠加**:每项都声称提升,但在你的语料上未必。标准做法是用 [[18 RAG 评估|RAG 评估]] 先测纯向量 baseline 的 **context recall / context precision**(Ragas 指标),再分别叠加每项、A/B 对比 top-k 召回失败率与端到端答案质量。Anthropic 官方那组「35%→49%→67%」就是这种逐层叠加的失败率对比,值得照搬方法论。
- **可观测**:ingestion 侧监控「每文档前缀生成耗时 / 缓存命中率 / 前缀平均长度」;RAPTOR 侧监控「树高 / 各层节点数 / 摘要 token 量」;检索侧用 LangSmith / Phoenix(Arize)/ Langfuse 记录每个 query 命中的层级与节点,定位「全局问答是否真的命中了摘要节点」。
- **选型口诀**:召回失败/关键词盲点 → Contextual Retrieval;全局综合长文问答 → RAPTOR;跨块指代/上下文连续性 + 已有长上下文 embedding → Late Chunking。三者正交,资源够就叠加,但永远**先把 [[03 分块策略 Chunking|分块策略 Chunking]] 和 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 做对**再上增强层。

## 面试高频

**Q1:Contextual Retrieval 到底解决什么问题?为什么不直接把块切大?**
标准答:解决「孤立 chunk 丢全文上下文导致 embedding 不准、召回失败」。不切大是因为切大会让一个向量糊多个主题、相似度不突出(块大召回不准,见 [[03 分块策略 Chunking|分块策略 Chunking]] 的根本矛盾)。Contextual Retrieval 的巧处是**保持块小(召回精准)的同时,用 LLM 生成的短前缀把全文背景补回去**,鱼和熊掌兼得。
- 追问「成本怎么控?」→ prompt caching:整篇文档作为公共前缀对该文档所有 chunk 复用,缓存命中后逐块成本被压到极低,这是落地前提。
- 陷阱:别答成「把上下文拼进去就行」而忽略成本——不开 cache 这方法在生产不可行,这是面试官最爱追的点。

**Q2:RAPTOR 和普通 chunk 检索的本质区别?它擅长什么、不擅长什么?**
标准答:普通检索只能召回零散短块,**回答不了需要综合整篇/整库的全局性问题**。RAPTOR 自底向上递归「聚类 + LLM 摘要」建树,上层节点是全局摘要,检索时跨层一起检索(collapsed tree),既命中具体事实又命中全局脉络。擅长全局/综合长文 QA;不擅长高频更新语料(增量重建树代价大)和对摘要质量敏感(摘错污染整层)。
- 追问「为什么用软聚类 GMM 而不是硬聚类?」→ 允许一个 chunk 属于多个簇,更贴合「一段话可能同时与多个主题相关」的现实。

**Q3:Late Chunking 和 Naive Chunking 差在哪?为什么它几乎零成本?**
标准答:Naive 是「先切块,再各块独立 embedding」,每块盲编码、丢跨块上下文;Late 是「先整篇过 Transformer 得每个 token 向量(因自注意力,每 token 都看过全文),再按块边界切、对块内 token 向量 mean-pooling」。「late」指**池化/切块发生在 token embedding 之后**。零成本是因为它不加 LLM、不改数据,只调换了「切块」和「embedding」的先后顺序,换个长上下文 token 级模型即可。
- 追问「它的局限?」→ 受 embedding 模型上下文窗口约束(超长文档仍要先分段,跨段上下文又丢);且要求模型能输出 token 级向量,不是所有 embedding API 都给。

**Q4(综合陷阱):这三个方法是互斥的吗?上线顺序怎么排?**
标准答:**正交、可叠加**,不是三选一。Contextual Retrieval 在「建库时」给块补背景前缀,Late Chunking 在「embedding 时」给块注入全文上下文,RAPTOR 在「索引结构上」加全局摘要层——三者作用在不同环节。上线顺序:先把基础分块/检索粒度做对,再按语料痛点选主线增强,每项都用 [[18 RAG 评估|RAG 评估]] 量化增益后再决定是否叠加。陷阱答法是说「选最强的那个」——面试官想听的是「它们正交且要先验证基线」。

## 知识拓展

- **Contextual Retrieval 的更早思想脉络**:给 chunk 补上下文不是 2024 才有,信息检索里早有 **document expansion**(doc2query / docT5query,Nogueira & Lin 2019,用生成模型给文档预测可能的 query 再索引)。Contextual Retrieval 可看作「用强 LLM 做文档/块扩展」的现代版,只是扩展的是「背景说明」而非「潜在 query」。
- **RAPTOR 之后的树/图结构检索前沿**:RAPTOR(ICLR 2024)走「摘要树」,微软 **GraphRAG**(2024)走「实体-关系知识图谱 + 社区摘要」,两者都在攻同一个「全局综合问答」难题但路线不同——RAPTOR 靠聚类摘要,GraphRAG 靠图社区检测(Leiden)+ 分层社区摘要。延伸阅读见 [[14 GraphRAG 知识图谱检索|GraphRAG 知识图谱检索]]。
- **Late Chunking 的理论根**:它本质依赖 Transformer 自注意力让每个 token 的表示融入全序列信息,与 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]] 里「向量相似度」的下游用法一脉相承——pooling 出的块向量仍用余弦相似度做 ANN。要求的「长上下文 + token 级输出」也解释了为何普通只给单一句向量的 embedding API(如早期 OpenAI text-embedding)用不了 late chunking。
- **边界与反模式**:① 别拿进阶索引救烂分块/烂粒度,基础没做好先补 [[03 分块策略 Chunking|分块策略 Chunking]]、[[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]];② RAPTOR 别用在高频更新语料(增量重建是反模式);③ Contextual Retrieval 别不开 prompt cache 就上量(成本反模式);④ 任何一项都别「不测就上」——必须在自己语料用 [[18 RAG 评估|RAG 评估]] 验证,通用 benchmark 的增益不保证迁移。
- **与召回-重排栈的关系**:进阶索引提升的是**第一阶段召回**质量,它和 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]](多路召回)、[[10 重排序 Reranking|重排序 Reranking]](精排收口)是同一条检索栈上的不同环节,Anthropic 的 67% 失败率下降正是「Contextual + Hybrid + Rerank」三者叠加的结果。

## 关键事实

- 三者解同一病:**孤立 chunk 丢全文上下文 → embedding 不准 → 召回失败**;路子分别是「LLM 前缀 / 摘要树 / 改 embedding 顺序」。
- **Contextual Retrieval**(Anthropic 2024-09):每块加 LLM 生成的上下文前缀。Contextual Embeddings 单用降召回失败 **35%**,+Contextual BM25 降 **49%**,再 +rerank 降 **67%**(5.7%→1.9%)。靠 prompt cache 压成本。
- **RAPTOR**(**Sarthi et al., arXiv:2401.18059, ICLR 2024**):递归 embed→聚类→LLM 摘要建树,跨层检索,擅长全局/综合长文 QA;RAPTOR+GPT-4 在 QuALITY 上绝对准确率 +约20%。
- **Late Chunking**(**Günther et al., Jina AI, arXiv:2409.04701**):先整篇 embedding 得 token 向量(各含全文上下文),再按块切并池化;免训练、近零成本,受模型上下文窗口限制。
- 三者**正交可叠加**,但都是增强层——基础分块/粒度没做好别先上;每项都需在自己语料上用 [[18 RAG 评估|RAG 评估]] 验证增益。

## 来源

- Anthropic. 2024-09.《Introducing Contextual Retrieval》(官方工程博客,anthropic.com/news/contextual-retrieval)。
- Sarthi et al. 2024.《RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval》arXiv:2401.18059(ICLR 2024)。
- Günther, Mohr, Wang, Xiao (Jina AI). 2024.《Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models》arXiv:2409.04701。
