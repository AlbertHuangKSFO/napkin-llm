[[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|进阶索引：Contextual Retrieval、RAPTOR、Late Chunking]] 是 2024 年三项针对 [[03 分块策略 Chunking|分块策略 Chunking]] 根本病的索引改良:**一个 chunk 被孤立切出来后,就丢掉了它在全文里的上下文**——「它」指代谁、这段属于哪一章、放在什么背景下,统统没了,导致 embedding 编不准、召回失败。三者用三条不同路子给 chunk **补回上下文**:Contextual Retrieval 用 LLM 写前缀、RAPTOR 建摘要树、Late Chunking 改 embedding 顺序。它们都发生在「建索引」阶段,是 [[04 Embedding 与向量数据库|Embedding 与向量数据库]] 之上的增强层。

## 本质:共同的敌人是「孤立 chunk 丢上下文」

[[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 里最隐蔽的一类失败:文档说「2023 年公司营收增长 3%」,但「公司」是哪家、相对哪个基期,写在上一段——切块后这个 chunk 单独看**信息不全**。query「ACME 公司 2023 营收增速」来了,这个 chunk 因为没出现「ACME」,相似度不够,**召不回**。问题不在 query、不在模型,在于**切块销毁了上下文**。

三种进阶索引从不同位置处理这个问题,但**不能据此推断任意组合都会增益**；是否组合、组合顺序和收益都应由同一语料上的消融实验决定:

| 方法 | 怎么补上下文 | 何时补 | 额外成本 |
|---|---|---|---|
| Contextual Retrieval | LLM 给每块生成「该块在全文中的位置/背景」前缀 | 建库时,逐块 | 每块一次 LLM(可缓存) |
| RAPTOR | 递归聚类+摘要,建一棵从细节到全局的树 | 建库时,递归 | 多轮聚类+LLM 摘要 |
| Late Chunking | 先整篇 embedding 让每 token 看过全文,再切 | embedding 阶段改顺序 | 不需额外训练/LLM,但长序列前向的显存与延迟须实测 |

**一句类比记住三条路**:像给一张从书里撕下来的散页补回出处——Contextual Retrieval 是**在散页顶上贴张便利贴**写「本页出自 ACME 2023 年报第三章」(LLM 写前缀);RAPTOR 是**另做一本目录/摘要册**,从章节摘要到全书主旨层层向上(摘要树);Late Chunking 是**先把整本书通读一遍再撕页**,撕下来时每页都还记得全书在讲什么(先整篇 embedding 再切)。

## 机制一:Contextual Retrieval(Anthropic, 2024-09)

来自 **Anthropic 2024 年 9 月工程博客《Contextual Retrieval》**。做法朴素而有效:**建库时,对每个 chunk,把「整篇文档 + 这个 chunk」喂给一个便宜 LLM(如 Claude Haiku),让它生成一两句话说明「这个 chunk 在全文里讲的是什么背景」,把这段上下文前缀拼到 chunk 前面,再做 embedding 和 BM25 索引。** 于是上例的 chunk 被改写成「[本段出自 ACME 公司 2023 年报,讨论营收] 2023 年公司营收增长 3%」——「ACME」「2023 年报」进了向量,召回就成了。

两个子技术:**Contextual Embeddings**(带前缀的块去 embedding)+ **Contextual BM25**(带前缀的块也进 BM25 关键词索引)。

**核验数字的适用边界**（Anthropic 官方工程博客，**2024-09-19**）：这是在 **codebases、fiction、arXiv papers、Science Papers** 等知识域取平均的实验；图中采用其测试中表现最好的 **Gemini Text 004** embedding 配置。指标是 $1-\mathrm{Recall@20}$，即相关文档未出现在 top-20 chunks 的比例，而不是端到端回答正确率，也不是任何生产语料的保证。

- 仅 Contextual Embeddings：top-20 失败率 $5.7\%\rightarrow3.7\%$，相对下降 **35%**。
- Contextual Embeddings + Contextual BM25：$5.7\%\rightarrow2.9\%$，相对下降 **49%**。
- 再接其测试的 Cohere reranker：先取 **top-150** 候选、重排后保留 **top-20**，失败率 $5.7\%\rightarrow1.9\%$，相对下降 **67%**。

这三项是同一篇实验中的不同配置，不应改写成“任何向量库都会提升 35/49/67%”。上线时应固定 query 集、chunk 规则、$k$、embedding 模型和 reranker，逐项复现 baseline → contextual embeddings → +BM25 → +reranker 的消融。

关键工程点:**prompt caching**——整篇文档作为前缀对该文档的所有 chunk 复用。Anthropic 当时给出的**一次性**成本估算是每百万**文档 token** \$1.02；前提明确为 800-token chunks、8k-token documents、50-token context instructions、每 chunk 生成 100-token context。这是按文档 token 而非 chunk 数计价的历史假设，也不是当前或通用生产成本；接入前应按当前供应商价格、实际文档长度、chunk 数、上下文长度与缓存命中率重新估算。

## 机制二:RAPTOR(Sarthi et al., ICLR 2024)

**RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**(**Sarthi et al. 2024, arXiv:2401.18059, ICLR 2024**,作者含 Christopher D. Manning)。解决的是另一个层面的上下文缺失:**普通 RAG 只能召回零散的短 chunk,无法回答需要跨越整篇文档、综合全局的问题**(「这本书的主旨」「全文论证的演进」)。

做法是**自底向上递归建树**:
1. 把文档切成 chunk(叶子,L0),各自 embedding。
2. 对这些向量**聚类**(论文用软聚类 GMM,允许一个块属于多簇)。
3. 每个簇用 LLM 生成一段**摘要**,这些摘要成为上一层节点(L1),再各自 embedding。
4. 对 L1 摘要再聚类、再摘要 → L2 …… **递归直到收敛成一个根**。

得到一棵树:底层是细节,越往上越抽象、越全局。**检索时跨层一起检索**(论文的 collapsed-tree:把所有层的节点放进同一个池子做最近邻)——query 既能命中 L0 的具体事实,也能命中 L1/L2 的主题/全局摘要,一次拿到「局部细节 + 全局脉络」。论文报告:RAPTOR + GPT-4 在 QuALITY 基准上比此前最佳**绝对准确率提升约 20%**,在需要多步推理的长文 QA 上是 SOTA。

![[进阶索引-RAPTOR树.png]]

## 机制三:Late Chunking(Jina AI, 2024)

**Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models**(Jina AI,**Günther et al. 2024, arXiv:2409.04701**)。它不需要额外训练或生成式 LLM；变化在于 **embedding 与池化/切块的先后顺序**。

对比 naive:
- **Naive Chunking**:先把文档**切**成块,再把每块**各自独立**喂进 embedding 模型 → 每块的向量是「盲编码」,不知道别的块,跨块指代/上下文丢失。
- **Late Chunking**:先把**整篇文档**(在 embedding 模型的长上下文窗口内,如 8192 token)一次性过 Transformer,得到**每个 token 的向量**——因为有自注意力,**每个 token 的向量都「看过」全文**;**然后才**按 chunk 边界把 token 序列切开,对每块的 token 向量做 **mean-pooling** 得到块向量。于是每个块向量天然携带全文上下文,而它仍是一个个独立的、可建库的块向量。

「late」就是指:**pooling(切块)发生在 token embedding 之后**,而非之前。论文的方法免额外训练，但不等于推理成本为零：将整篇（或窗口内的大段）送进长序列 encoder 可能增加显存占用和延迟，需以目标模型、长度分布、batch 与硬件实测。Jina `jina-embeddings-v3` API 提供原生 `late_chunking=true` 开关；**其他模型不能只因“长窗口”就假定可用**，需要自行导出 token hidden states，并用同一个 tokenizer 把文本 chunk 边界严格映射到 token span。局限还包括上下文窗口；超长文档仍须采用 long-late-chunking 或先分大段，跨大段的上下文并不会自动保留。

![[Late Chunking 对比.png]]

## 小数字手算：按 token span 池化

以下是**人为构造的二维 hidden states**，只演示算法，不代表模型效果。整段一次前向后有 $H=[(1,10),(2,20),(3,30),(4,40)]$，两个已对齐的 token spans 是 $[0,2)$ 与 $[2,4)$：

- 第一个块：$[(1,10)+(2,20)]/2=(1.5,15)$。
- 第二个块：$[(3,30)+(4,40)]/2=(3.5,35)$。

最终得到两个可单独入库的块向量 $(1.5,15)$、$(3.5,35)$；这也正是下方代码断言的最终值。关键不是这些玩具数值，而是 span 必须来自**同一 tokenizer、同一次整段编码**。

## 公式推导：Late Chunking 的均值池化

设整段 token 为 $x_{1:L}$，长上下文 encoder 的 token hidden states 为：

$$
H=F(x_{1:L})=[h_1,\ldots,h_L],\qquad h_i\in\mathbb{R}^d
$$

若一个 chunk 对应 token 区间 $[s,e)$，则其向量是该区间的均值：

$$
v_{s:e}=\frac{1}{e-s}\sum_{i=s}^{e-1}h_i
$$

Naive chunking 则在切片上独立前向：$\tilde v_{s:e}=\operatorname{mean}(F(x_{s:e}))$。因为 Transformer 的 $h_i$ 可依赖 $x_{1:L}$ 中的其他 token，一般 $v_{s:e}\ne\tilde v_{s:e}$；这正是 Late Chunking 可能保留跨块指代的原因。该式只描述表示如何生成，不承诺任何数据集上的召回增益。

## 可跑最小代码

```python
# ① Contextual Retrieval：把 LLM 客户端作为依赖注入；生产中按 document 分组以命中 prompt cache。
def contextualize(document, chunk, llm):
    prompt = (f"<document>{document}</document>\n"
              f"<chunk>{chunk}</chunk>\n"
              "只输出一句该 chunk 在全文中的检索背景。")
    return f"{llm(prompt).strip()}\n\n{chunk}"

# ③ Late Chunking：spans 必须是「同一 tokenizer、同一次未截断文档编码」的 token 下标。
def mean_pool(rows):
    return [sum(col) / len(rows) for col in zip(*rows)]

def late_chunking(token_ids, spans, encode_token_hidden_states):
    hidden = encode_token_hidden_states(token_ids)  # 一次长序列前向，形状 (L, d)
    if len(hidden) != len(token_ids):
        raise ValueError("hidden states 必须与同一 tokenizer 的 token_ids 一一对齐")
    vectors = []
    for start, end in spans:
        if not (0 <= start < end <= len(hidden)):
            raise ValueError(f"非法 token span: {(start, end)}")
        vectors.append(mean_pool(hidden[start:end]))
    return vectors

if __name__ == "__main__":
    contextual = contextualize(
        "ACME 2023 年报讨论 Q2 营收。", "营收环比增长 3%。",
        lambda _: "本段来自 ACME 2023 年报的 Q2 营收章节。")
    assert contextual.endswith("营收环比增长 3%。")

    # 为使示例不依赖下载模型，这里用可验证的假 hidden states 代替真实 encoder。
    # ✅ 真实实现应从模型导出 token hidden states；不可把字符区间直接当 token span。
    fake_encoder = lambda ids: [[float(i), float(i * 10)] for i in ids]
    assert late_chunking([1, 2, 3, 4], [(0, 2), (2, 4)], fake_encoder) == [
        [1.5, 15.0], [3.5, 35.0]
    ]
    print("contextualize 与 token-span pooling 通过")
```

上面代码验证的是**顺序与 span 对齐约束**，不是模型基准。Jina `jina-embeddings-v3` 的 API 可直接传 `late_chunking=true`；手写其他模型适配时，须保留 offset mapping、处理 special tokens，并对“先整段编码后池化”与 naive chunking 分别测显存、P50/P95 延迟与 Recall@k。

## 对比表

| 维度 | Contextual Retrieval | RAPTOR | Late Chunking |
|---|---|---|---|
| 补的是什么上下文 | 该块在全文的背景说明 | 跨层的全局/主题摘要 | 每块的全文 token 级上下文 |
| 建库开销 | 逐块一次 LLM（可缓存；成本取决于 token/价格） | 递归聚类+多轮 LLM 摘要 | 无额外训练/LLM；长序列编码的显存、吞吐与延迟须测 |
| 是否引入新内容 | 是(LLM 生成前缀) | 是(摘要节点) | 否(仍是原块) |
| 最擅长 | 召回失败、关键词盲点 | 全局/综合性长文问答 | 跨块指代、上下文连续性 |
| 依赖 | 生成用 LLM + prompt cache | LLM + 聚类 | 长上下文 token-level 模型 |
| 出处 | Anthropic 2024-09 博客 | arXiv:2401.18059 ICLR24 | arXiv:2409.04701 Jina |

## 何时用 / 坑

- **按问题与消融选型，不按“可叠加”想象选型**：Anthropic 的 67% 只验证了其 Contextual Embeddings + Contextual BM25 + Cohere reranker 配置；RAPTOR 与 Late Chunking 的论文并未给出这三种方法任意组合的通用消融。若候选方案会同时改 chunk 表示、索引节点和 rerank，应先单独测，再测成对组合，最后才考虑全组合。
- **Contextual Retrieval 不开 prompt cache 会很贵**:逐块带整篇文档进 LLM,不缓存成本爆炸。这是落地前提,务必确认。
- **RAPTOR 建索引重、增量更新难**:文档一变,聚类和摘要树要重建;摘要质量受 LLM 限制(摘错=污染整层)。适合相对静态、需要全局问答的语料(书、报告);高频更新的语料慎用。
- **Late Chunking 受窗口和接口限制**:文档超过模型上下文仍要采用 long-late-chunking 或先分大段，跨大段的上下文不会自动保留；Jina v3 是有原生开关的例子。其他模型须可导出 token hidden states，并确保文本 chunk 的边界与 tokenizer 得到的 token span 对齐；很多只返回单向量的 embedding API 不具备这个接口。
- **别拿进阶索引救烂分块**:这些是「锦上添花」,基础的 [[03 分块策略 Chunking|分块策略 Chunking]]、[[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 没做好,先补那些。
- **评估要量化**:每项都声称提升,但在你的语料上未必。上之前先用 [[18 RAG 评估|RAG 评估]] 测召回基线,叠加后对比。
- 这些索引增强的最终目标和 [[36 Agentic RAG|Agentic RAG]] 一致——让检索拿到的证据更全、更准,只是 Agentic 在「检索时」补救,进阶索引在「建库时」预防。

## 工业界实践

生产选型先看语料、接口和可接受的建库/查询预算；是否叠加不是默认答案。下面的建议是实施检查表，增益需用自己的标注 query 集验证。

### Contextual Retrieval 的工业落地

它在 ingestion 管线增加「逐块 LLM 加前缀」，产物仍是普通文本 chunk，因此可以接入现有的向量与 BM25 索引；但是否值得引入 LLM 成本、缓存依赖和前缀质量控制，应由评估决定。

**典型架构(ingestion 阶段)**:
```
原始文档 → 切块(03 分块策略)
        → 逐块: prompt = [整篇文档(命中 cache) + 该 chunk] → LLM 生成上下文前缀
        → 前缀 + 原 chunk 一起去 ① embedding 入向量库 ② 分词入 BM25 倒排
检索阶段:dense + sparse 两路召回(08 混合检索)→ RRF 融合 → reranker 收口(10 重排序)
```

**成本要按 token 账本核算**。没有缓存时，同一文档会随每个 chunk 重复输入；缓存策略、TTL、供应商价格和调用排程决定实际账单。可复核的历史锚点只有 Anthropic 2024-09-19 的一次性 **\$1.02 / 百万文档 token** 估算（800-token chunk、8k-token 文档、50-token 指令、100-token 前缀），不可改写为按 chunk 数的成本。上线前以当前合同价格和真实 token 日志重新计算。

**工程化要点**:
- **前缀长度以消融定**:Anthropic 示例通常生成 50--100 token；对自己的 corpus 至少比较前缀长度、Recall@k、索引大小与 BM25 词项膨胀。
- **缓存 TTL 与并发**:按文档分组，在缓存有效窗口内处理其 chunks；记录命中/未命中 token、重试和前缀版本。
- **更新策略**:文档变更后，必须让该文档的前缀、dense 向量和 BM25 条目以同一版本重建；是否比摘要树便宜取决于文档大小、树的共享范围和实现。

### RAPTOR 的工业落地

当失败 query 确实需要综合长文而不是定位单段事实时，RAPTOR 的摘要节点才是候选；它的建树和更新成本较高，适合优先在相对静态的语料上做小规模验证。

- **聚类与摘要成本**:每层都要跑聚类(UMAP 降维 + GMM)+ 每簇一次 LLM 摘要,层数 × 簇数次 LLM 调用。摘要模型质量直接决定上层节点质量——**摘错会污染整层及其以上**,所以摘要 prompt 要强调「忠实、不引入文档外信息」,并对关键语料抽检。
- **检索策略须比较**:论文给出 collapsed-tree 与 tree traversal；在自己的问题集上分别记录叶子命中、摘要命中、Recall@k 和延迟，避免预设一种必然更快或更准。
- **增量更新**:语料变更会影响相关簇及祖先摘要；需要把树版本、源 chunk 和摘要的来源关系持久化，才能界定重算范围。

### Late Chunking 的工业落地

Late Chunking 不要求额外训练或 LLM，但它不是免费开关：长序列 encoder 的显存、吞吐、尾延迟和批大小可能改变。先确认模型/服务是否提供正确的接口，再跑与 naive chunking 同条件的基准。

- **接口判断**:Jina `jina-embeddings-v3` API 有原生 `late_chunking=true`；这是本文唯一作为“开关”列出的模型。其他候选须自行取得 token hidden states，复用同一 tokenizer/offset mapping，并明确 padding、special token 与截断规则，才可按 span 池化。
- **窗口与资源基准**:文档超过窗口时按自然语义边界切为大段，再在段内处理；报告跨段丢失比例、GPU 峰值显存、tokens/s、P50/P95 和 Recall@k；“无额外训练”不等于没有编码资源开销。
- **与父文档检索组合**:可把它作为 [[06 检索粒度：父文档与句子窗口检索|父文档与句子窗口检索]] 子块表示的候选实现；需比较父文档返回前后、以及有/无 late chunking 的 2×2 消融。

### 评估、可观测与选型

- **必须先量化基线再组合**:用 [[18 RAG 评估|RAG 评估]] 先测纯向量 baseline 的 context recall/context precision，再做单项与成对消融，记录 top-k 失败率和端到端答案质量。Anthropic 的 35/49/67% 可以借鉴“固定配置逐步比较”的方法，不能当作目标数值。
- **可观测**:ingestion 侧记录每文档前缀生成耗时、缓存命中率和前缀长度；RAPTOR 侧记录树高、各层节点数和摘要 token；检索侧记录命中的层级、节点和版本。
- **选型起点**:关键词/背景缺失可先试 Contextual Retrieval；确有全局综合问题再试 RAPTOR；跨块指代且接口可导出 token states 时试 Late Chunking。始终先保证 [[03 分块策略 Chunking|分块策略 Chunking]] 与 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 可评估，再决定保留哪一种。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1:Contextual Retrieval 到底解决什么问题?为什么不直接把块切大?**
标准答:解决「孤立 chunk 丢全文上下文导致 embedding 不准、召回失败」。不切大是因为切大会让一个向量糊多个主题、相似度不突出(块大召回不准,见 [[03 分块策略 Chunking|分块策略 Chunking]] 的根本矛盾)。Contextual Retrieval 的巧处是**保持块小(召回精准)的同时,用 LLM 生成的短前缀把全文背景补回去**,鱼和熊掌兼得。
- 追问「成本怎么控?」→ prompt caching:整篇文档作为公共前缀对该文档所有 chunk 复用,缓存命中后逐块成本被压到极低,这是落地前提。
- 陷阱:别答成「把上下文拼进去就行」而忽略成本——不开 cache 这方法在生产不可行,这是面试官最爱追的点。

**Q2:RAPTOR 和普通 chunk 检索的本质区别?它擅长什么、不擅长什么?**
标准答:普通检索只能召回零散短块,**回答不了需要综合整篇/整库的全局性问题**。RAPTOR 自底向上递归「聚类 + LLM 摘要」建树,上层节点是全局摘要,检索时跨层一起检索(collapsed tree),既命中具体事实又命中全局脉络。擅长全局/综合长文 QA;不擅长高频更新语料(增量重建树代价大)和对摘要质量敏感(摘错污染整层)。
- 追问「为什么用软聚类 GMM 而不是硬聚类?」→ 允许一个 chunk 属于多个簇,更贴合「一段话可能同时与多个主题相关」的现实。

**Q3:Late Chunking 和 Naive Chunking 差在哪?它真的零成本吗?**
标准答:Naive 是「先切块,再各块独立 embedding」,每块盲编码、丢跨块上下文;Late 是「先整篇过 Transformer 得每个 token 向量(因自注意力,每 token 都看过全文),再按块边界切、对块内 token 向量 mean-pooling」。「late」指**池化/切块发生在 token embedding 之后**。它免额外训练和生成式 LLM，但整段长序列前向可能增加显存和延迟；不能把“少一个训练阶段”答成“零成本”。
- 追问「它的局限?」→ 受 embedding 模型上下文窗口约束(超长文档须处理大段边界);Jina v3 API 有 `late_chunking=true`，其他模型须可导出 token hidden states 且 token span 与原文 chunk 对齐；对每个目标模型实测资源与 Recall@k。

**Q4(综合陷阱):这三个方法是互斥的吗?上线顺序怎么排?**
标准答:它们改变的是不同环节，**但没有“三者任意组合必然有效”的通用消融证据**。Contextual Retrieval 改块文本，Late Chunking 改块向量生成，RAPTOR 加摘要节点；可能组合，也可能因成本、上下文噪声或数据分布无收益。上线顺序是先把基础分块/检索粒度做对，再针对失败模式选一项，固定评测集做单项与成对消融，收益大于成本才保留。Anthropic 的 67% 只属于其 Contextual Embeddings + Contextual BM25 + reranker 实验，不可外推为三者叠加结果。

## 知识拓展

- **Contextual Retrieval 的更早思想脉络**:给 chunk 补上下文不是 2024 才有,信息检索里早有 **document expansion**(doc2query / docT5query,Nogueira & Lin 2019,用生成模型给文档预测可能的 query 再索引)。Contextual Retrieval 可看作「用强 LLM 做文档/块扩展」的现代版,只是扩展的是「背景说明」而非「潜在 query」。
- **RAPTOR 之后的树/图结构检索前沿**:RAPTOR(ICLR 2024)走「摘要树」,微软 **GraphRAG**(2024)走「实体-关系知识图谱 + 社区摘要」,两者都在攻同一个「全局综合问答」难题但路线不同——RAPTOR 靠聚类摘要,GraphRAG 靠图社区检测(Leiden)+ 分层社区摘要。延伸阅读见 [[14 GraphRAG 知识图谱检索|GraphRAG 知识图谱检索]]。
- **Late Chunking 的理论根**:它本质依赖 Transformer 自注意力让每个 token 的表示融入全序列信息,与 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]] 里「向量相似度」的下游用法一脉相承——pooling 出的块向量仍用余弦相似度做 ANN。要求的「长上下文 + token 级输出」也解释了为何普通只给单一句向量的 embedding API(如早期 OpenAI text-embedding)用不了 late chunking。
- **边界与反模式**:① 别拿进阶索引救烂分块/烂粒度,基础没做好先补 [[03 分块策略 Chunking|分块策略 Chunking]]、[[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]];② RAPTOR 别用在高频更新语料(增量重建是反模式);③ Contextual Retrieval 别不开 prompt cache 就上量(成本反模式);④ 任何一项都别「不测就上」——必须在自己语料用 [[18 RAG 评估|RAG 评估]] 验证,通用 benchmark 的增益不保证迁移。
- **与召回-重排栈的关系**:进阶索引主要改第一阶段召回表示或候选结构；它可与 [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]]、[[10 重排序 Reranking|重排序 Reranking]] 一同评测。Anthropic 的 67% 属于其 Contextual Embeddings + Contextual BM25 + Cohere reranker、top-150→top-20、$1-\mathrm{Recall@20}$ 的固定实验配置。

## 关键事实

- 三者解同一病:**孤立 chunk 丢全文上下文 → embedding 不准 → 召回失败**;路子分别是「LLM 前缀 / 摘要树 / 改 embedding 顺序」。
- **Contextual Retrieval**（Anthropic **2024-09-19**）：每块加 LLM 生成的上下文前缀。35/49/67% 是跨 codebases、fiction、arXiv papers、Science Papers 等域的 **Gemini Text 004** 实验，指标为 $1-\mathrm{Recall@20}$：$5.7\%\rightarrow3.7\%$、$2.9\%$、$1.9\%$；67% 还特指 Cohere reranker 的 top-150→top-20。其 \$1.02 / 百万**文档 token** 一次性成本假设为 800-token chunks、8k-token documents、50-token 指令、100-token 前缀，须按当前价格重算。
- **RAPTOR**（Sarthi et al., arXiv:2401.18059，2024-01-31；ICLR 2024）：递归 embed→聚类→LLM 摘要建树、跨层检索。论文报告在 QuALITY 上与 GPT-4 配合时，相比此前最佳结果绝对准确率约 +20%，是该基准/配置结果。
- **Late Chunking**（Günther et al., arXiv:2409.04701，2024-09-07）：先整篇编码出 token hidden states，再按 token span 池化；免额外训练，但长序列资源成本须实测。Jina `jina-embeddings-v3` API 有原生开关；其他模型必须能导出 token hidden states 并保证 span 对齐。
- 三者都是增强候选而非必然可叠加的配方；以 [[18 RAG 评估|RAG 评估]] 先做 baseline、单项和组合消融，再决定是否上线。

## 来源

- Anthropic. **2024-09-19**. [《Contextual Retrieval》官方工程博客](https://www.anthropic.com/engineering/contextual-retrieval)：实验域、Gemini Text 004、$1-\mathrm{Recall@20}$、top-150→top-20 与 \$1.02 / 百万文档 token 的假设。
- Sarthi et al. **2024-01-31**. [《RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval》](https://arxiv.org/abs/2401.18059)，arXiv:2401.18059，ICLR 2024。
- Günther, Mohr, Williams, Wang, Xiao. **2024-09-07**. [《Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models》](https://arxiv.org/abs/2409.04701)，arXiv:2409.04701。
- Jina AI. [《Migration From Jina Embeddings v2 to v3》](https://jina.ai/news/migration-from-jina-embeddings-v2-to-v3/)：`jina-embeddings-v3` API 的 `late_chunking=true`、8192-token 请求约束（产品行为会变化，使用时复核当前文档）。
