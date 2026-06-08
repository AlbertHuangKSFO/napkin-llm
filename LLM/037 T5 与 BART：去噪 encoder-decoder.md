[[037 T5 与 BART：去噪 encoder-decoder]]:两个 2019—2020 年的 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|encoder-decoder]] 模型——T5 把所有 NLP 任务统一成“文本进、文本出”,BART 用“先污染句子、再还原整句”的去噪目标预训练;它们是 decoder-only 大潮之前 seq2seq 路线的巅峰。

## 直觉:不止“填一个空”,而是“整段挖空再补回”
[[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|BERT]] 的 MLM 只遮**单个词**、让 encoder 填回,擅长理解但不会生成长文本;[[036 GPT 系列：自回归与规模化|GPT]] 只会从左到右续写,擅长生成但没有双向编码。T5 与 BART 想要**两头都占**:用一个能双向看全文的 encoder + 一个能自回归生成的 decoder。

它们的预训练秘诀是**去噪(denoising)**:故意把训练句子破坏掉,再让模型还原。模型为了把句子补回来,必须真正理解语义和语法——这就是免费的监督信号,不需要任何人工标注。

- **T5**:把连续的几个词挖掉换成一个“哨兵 token”,decoder 只生成被挖掉的那几段(span corruption)。
- **BART**:用更花哨的噪声(遮挡、删除、打乱顺序、旋转),decoder 重建**整条原始句子**。

T5 还有个统一的大招:**所有任务都写成“文本→文本”**。翻译、摘要、情感分类、问答……全都给输入加一个任务前缀(如 `translate English to German:`),输出也是一段文本。一个模型、一种格式,通吃所有任务——这正是 “Text-To-Text Transfer Transformer” 名字里五个 T 的由来。

## 例子:一句话的去噪流程(小数字)
原句 `Thank you for inviting me to your party`(8 个词)。

T5 随机选中两段连续片段 `you for` 和 `to`,各换成一个哨兵 `<X>`、`<Y>`:

- Encoder 输入(被破坏):`Thank <X> inviting me <Y> your party`(7 个 token)
- Decoder 目标(只生成被遮内容):`<X> you for <Y> to <Z>`(`<Z>` 是结束哨兵)

注意 decoder **只产出被挖掉的那几个词**,不是重写整句——目标序列短,训练算力省。T5 论文实测:被遮比例约 **15%**、span 平均长度约 **3** 时效果最好。

BART 对同一句话可能这样加噪:`Thank ___ me your party .`(遮挡 + 删除 + 句子旋转),然后要求 decoder 输出**完整原句** `Thank you for inviting me to your party`。

**BART 的五种噪声(具体)**。BART 论文系统试了多种破坏方式,最终配方混用:① **token masking**(像 BERT 遮单个词);② **token deletion**(直接删词,模型还要自己判断「哪里少了」);③ **text infilling**(挖掉一段连续 span,用**单个** `[MASK]` 代替——逼模型预测「这里原本有几个词」,这一项最关键);④ **sentence permutation**(句子顺序打乱);⑤ **document rotation**(把文档旋转到从某个随机 token 开头)。其中 text infilling + sentence permutation 组合效果最好。

![[hist-bart五噪声.png]]

**T5 的 text-to-text 前缀长什么样(更多例子)**。同一个 T5 模型靠不同前缀通吃所有任务,输出永远是文本:

| 任务 | 输入(带前缀) | 目标输出 |
|---|---|---|
| 翻译 | `translate English to German: That is good.` | `Das ist gut.` |
| 摘要 | `summarize: <长文章>` | `<摘要>` |
| 情感 | `sst2 sentence: it was great` | `positive` |
| 蕴含 | `mnli premise: ... hypothesis: ...` | `entailment` |
| 语义相似 | `stsb sentence1: ... sentence2: ...` | `3.8`(回归值也当文本输出!) |

注意最后一行:连「打分」这种回归任务,T5 也把数字**当字符串生成**——这就是「一切皆文本」的彻底之处。

![[hist-T5去噪.png]]

## 原理:双向编码 + 自回归解码 + 去噪损失
**结构**:标准 [[012 交叉注意力 Cross-Attention|encoder-decoder]] Transformer。Encoder 用双向自注意力(无掩码,token 互相全可见);decoder 用[[007 因果掩码与 padding 掩码|因果掩码]]的自注意力 + 对 encoder 输出做 [[012 交叉注意力 Cross-Attention|cross-attention]]。

**去噪目标(以 T5 为例)**:设原序列 $x=(x_1,\dots,x_n)$,被破坏后的输入为 $\tilde{x}$,被遮片段拼成目标 $y=(y_1,\dots,y_m)$($m\ll n$)。模型最小化目标的[[30 交叉熵与负对数似然|交叉熵]]:
$$\mathcal{L} = -\sum_{t=1}^{m}\log P_\theta\big(y_t \mid y_{<t},\,\tilde{x}\big)$$
其中 $y_{<t}$ 是 decoder 已生成的前缀,$\tilde{x}$ 经 encoder 编码后由 cross-attention 注入。这和 [[036 GPT 系列：自回归与规模化|GPT]] 的 next-token 损失形式一致,差别只在**条件**多了一个 encoder 端的 $\tilde{x}$,以及目标 $y$ 是“被遮片段”而非整条续写。

**为什么 span 比单 token 好**:遮单个词(BERT 式)信息泄露多——上下文几乎确定了答案;遮连续 span 迫使模型建模**多词的联合分布**,更接近真实生成。T5 论文系统对比了多种破坏方式,得出 span corruption 是 encoder-decoder 上的最优配方。

**T5 的位置编码**:用**相对位置偏置**(每个注意力头加一个可学习的、按相对距离分桶的标量),而非正弦或 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]]——这是 T5 与后来 decoder-only 模型的一个显著差异。具体做法:把相对距离 $|i-j|$ 按对数分桶(近处分得细、远处粗,共 32 个桶),每个桶 × 每个头对应一个可学习标量,直接加到注意力 logits 上。好处是不占输入维度、相对位置天然、可一定程度外推。

**为什么 encoder-decoder 的 cross-attention 不可省**。decoder 生成每个 token 时,通过 [[012 交叉注意力 Cross-Attention|cross-attention]] 去「查」encoder 编码好的整段输入:Query 来自 decoder 当前位置,Key/Value 来自 encoder 输出。这让 decoder 在生成时**始终能看到完整、双向编码过的源文本**——这是 seq2seq(翻译/摘要)的天然契合点。相比之下 decoder-only 把「输入」和「输出」拼成一条序列、共用一套因果注意力,没有独立的双向 encoder。

![[tf-bart-crossattn.png]]

**架构消融:T5 论文比过三种结构**。Raffel 等系统对比了 encoder-decoder、language model(decoder-only)、prefix-LM 三种,结论是在他们的迁移学习设定下 **encoder-decoder + span corruption 最优**——但这个结论是在「中等规模 + 有监督迁移」下得到的;到了**超大规模纯自监督**,decoder-only 反而胜出(见下方面试高频)。这是「为什么论文说 enc-dec 好、实际主流却是 decoder-only」这个常见困惑的根源。

![[hist-encdec家族.png]]

## 代码:T5 风格的 span 破坏(可运行)
```python
import random

def t5_span_corrupt(tokens, mask_ratio=0.15, mean_span=3, seed=0):
    """把连续片段换成哨兵 <X>,<Y>...;返回 (encoder 输入, decoder 目标)。"""
    random.seed(seed)
    n = len(tokens)
    n_mask = max(1, round(n * mask_ratio))      # 要遮的 token 总数
    # 随机决定每个位置是否被遮(简化:伯努利后再合并相邻为 span)
    masked = sorted(random.sample(range(n), n_mask))
    spans, cur = [], [masked[0]]
    for p in masked[1:]:
        if p == cur[-1] + 1: cur.append(p)      # 相邻 → 合并成同一 span
        else: spans.append(cur); cur = [p]
    spans.append(cur)

    sentinel = lambda i: f"<extra_id_{i}>"
    enc, tgt, i, covered = [], [], 0, set(p for s in spans for p in s)
    for pos in range(n):
        if pos in covered:
            if pos == 0 or (pos - 1) not in covered:   # span 起点 → 放一个哨兵
                enc.append(sentinel(i)); tgt.append(sentinel(i)); i += 1
            tgt.append(tokens[pos])                     # 被遮词进 decoder 目标
        else:
            enc.append(tokens[pos])                     # 没遮的词留在 encoder 输入
    tgt.append(sentinel(i))                             # 收尾哨兵
    return enc, tgt

words = "Thank you for inviting me to your party".split()
enc, tgt = t5_span_corrupt(words, seed=3)
print("encoder 输入:", " ".join(enc))   # 例: Thank <extra_id_0> inviting me <extra_id_1> your party
print("decoder 目标:", " ".join(tgt))   # 例: <extra_id_0> you for <extra_id_1> to <extra_id_2>
```

```python
# ❌ 错误理解:以为 decoder 要重写整句(那是 BART,不是 T5,且白白多算很多 token)
tgt_wrong = words                          # 整句作目标 → 序列长、训练慢、信息冗余

# ✅ T5:decoder 只生成“被遮片段 + 哨兵”,目标短、信号纯
#    BART 才是“加噪后重建完整原句”;两者目标不同,别混
```

```python
# 三种预训练目标的「目标序列长度」对比(同一句 8 词,看谁算得多)
sent = "Thank you for inviting me to your party".split()   # 8 词
print("BERT MLM  目标长度 =", round(8*0.15))   # ≈1，只预测被遮的 15%
print("T5  span  目标长度 =", "约 2~3 词 + 哨兵")  # 只生成被挖片段，短
print("BART      目标长度 =", len(sent))        # 8，重建整句,最长
print("GPT  自回归 目标长度 =", len(sent))       # 8，每个位置都预测下一个
# 结论:T5 目标最短(省算),BART/GPT 要写满整句;但 GPT 每位置都有监督信号
```

## 去噪目标家族:一张对照表

零基础容易把 MLM/span/去噪混成一团,记住这张「谁破坏什么、谁生成什么」就清楚了:

| 模型 | 结构 | 破坏方式 | 生成目标 | 代表 |
|---|---|---|---|---|
| BERT | encoder-only | 遮单个词(15%) | 只填被遮词 | 理解 |
| T5 | enc-dec | 挖连续 span→哨兵 | 只补被挖片段(短) | 统一 text-to-text |
| BART | enc-dec | 5 种噪声(遮/删/打乱/旋转/infill) | 重建完整原句 | 生成(摘要) |
| GPT | decoder-only | 不破坏 | 每位置预测下一词 | 生成(通用) |

一句话区分:**BERT 填空、T5 补片、BART 重建整句、GPT 续写**。

## 后续与衍生:T5/BART 没有死

- **mT5**:T5 的多语言版,101 种语言,词表 25 万(SentencePiece-Unigram,见 [[052 WordPiece、Unigram 与 SentencePiece|分词]])。
- **Flan-T5**:在 T5 上做大规模**指令微调**(上千个任务),零样本/少样本能力大涨,是「encoder-decoder + 指令微调」的代表,至今在很多专用任务上仍好用、便宜。
- **ByT5**:无分词器、直接吃字节的 T5,对噪声/拼写鲁棒(代价是序列长)。
- **UL2**:用「混合去噪器」(mixture-of-denoisers,同时训 span、前缀 LM 等多种目标)统一 enc-dec 与 decoder-only 的预训练。
- 现实中 BART/T5 在**摘要、翻译、语法纠错**等「输入→输出」明确的任务上依然是性价比之选;decoder-only 赢在「通用对话 + scaling」。

## 面试高频
- **T5 和 BERT、GPT 的关系?** 三者都用 Transformer:BERT 是 encoder-only(双向、只填空、不擅生成);GPT 是 decoder-only(单向、只续写);T5/BART 是 encoder-decoder(双向编码 + 自回归解码,理解与生成兼顾)。见 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|三种结构]]。
- **T5 的“span corruption”比 BERT 的 MLM 好在哪?** 遮**连续片段**而非单词,迫使模型建模多词联合分布,更贴近生成;且 decoder 只生成被遮内容,目标序列短、训练高效。
- **T5 与 BART 去噪目标的区别?** T5:挖空→只补空(输出短)。BART:加多种噪声(遮/删/打乱/旋转)→重建整句(输出 = 完整原文)。BART 的 decoder 因此更像一个标准自回归 LM。
- **既然 encoder-decoder 这么强,为什么后来主流是 decoder-only?** decoder-only 训练目标更简单(纯 next-token)、能直接吃海量无标注文本、in-context learning 强、scaling 曲线更平滑且参数利用率高;cross-attention 与单独 encoder 的额外结构在超大规模上性价比下降。这是 [[040 现代 decoder-only 配方汇总|现代配方]]的历史背景。
- **T5 用什么位置编码?** 相对位置偏置(分桶标量),不是 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]]——常被拿来和现代模型对比。
- **T5 论文不是说 encoder-decoder 最好吗,为什么现在都用 decoder-only?** 那个结论在「中等规模 + 有监督迁移」下成立;超大规模纯自监督下,decoder-only 训练目标更简单(纯 next-token)、能吃海量无标注、ICL 强、scaling 更平滑、无 cross-attention 的额外结构开销 → 性价比反超。
- **BART 的哪种噪声最关键?** text infilling(挖连续 span 用单个 `[MASK]` 代替),逼模型预测「这里原本有几个词」,比单 token masking 更难、更接近生成。
- **T5 怎么把分类/回归也变成文本?** 加任务前缀,输出永远是字符串——连相似度打分(如 `3.8`)都当文本生成,这就是「一切皆 text-to-text」。
- **ByT5/字节级 T5 解决什么?** 去掉分词器、直接吃字节,对拼写错误/噪声/多语言更鲁棒,代价是序列更长(见 [[051 BPE 与 Byte-level BPE|字节级]])。

## 关键事实
- T5 出处:Raffel et al.,*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*,JMLR 2020(arXiv:1910.10683,2019 年放出)。核心:**text-to-text** 统一框架 + **span corruption** 去噪;系统对比了架构、目标、数据后给出最优配方。
- BART 出处:Lewis et al.,*BART: Denoising Sequence-to-Sequence Pre-training…*,ACL 2020(arXiv:1910.13461,2019 年放出)。核心:**多种噪声 + 重建完整原文**;在生成类任务(摘要、翻译)上很强。
- T5 规模:从 60M(small)到 11B(T5-XXL);多语种版 mT5。BART-large 约 400M。
- 去噪超参(T5 最优):破坏比例约 **15%**、span 平均长度约 **3**。
- BART 五种噪声:token masking / deletion / **text infilling** / sentence permutation / document rotation;最优组合是 text infilling + sentence permutation。
- T5 位置编码:**相对位置偏置**,相对距离对数分桶(32 桶),每桶每头一个可学习标量加到注意力 logits。
- 衍生:**mT5**(101 语言,Unigram 25 万词表)、**Flan-T5**(大规模指令微调)、**ByT5**(字节级无分词器)、**UL2**(混合去噪器统一目标)。
- 与邻接概念:结构属于 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|encoder-decoder]],核心是 [[012 交叉注意力 Cross-Attention|cross-attention]];被 [[036 GPT 系列：自回归与规模化|GPT 系自回归]]路线取代,后者演化出 [[038 LLaMA 架构解剖|LLaMA]] 等现代 decoder-only。
