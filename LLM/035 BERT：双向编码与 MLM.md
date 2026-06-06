[[035 BERT：双向编码与 MLM]]:第一个真正成功的双向预训练语言模型(Devlin et al., 2018,arXiv 1810.04805),用「完形填空」式的掩码语言模型(MLM)+ 下一句预测(NSP)预训练一个 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|encoder-only]] Transformer,刷爆了一众理解类任务。

## 直觉:让模型同时看左右

GPT 式的语言模型只能**从左往右**预测下一个词,每个位置只看到它左边的上下文。但人读「巴黎是法国的 ___」时,**左右文都用上**才好猜空。BERT 的核心主张:**理解类任务应该双向看**。

怎么让模型双向看?直接双向预测下一个词会「偷看答案」(要预测的词在它能看到的上下文里)。BERT 的妙招是**完形填空**:随机把句子里 15% 的词换成 `[MASK]`,让模型用**两侧**的上下文猜被遮的词。既双向,又不泄题。

代价:BERT 不能像 GPT 那样**逐词生成**——它是个「理解器/判别器」,不是「生成器」。所以它擅长分类、抽取、检索([[04 Embedding 与向量数据库|句向量/Embedding]] 的早期主力),但不会写文章。这正是 [[034 发展史总览：2017 到今|发展史]] 里「理解 vs 生成」两条路线的理解侧代表。

![[hist-bert-bidirectional.svg]]

## 例子:MLM 怎么遮、NSP 怎么判

**MLM**。句子「巴黎 是 法国 的 首都」,随机选中「法国」来遮(占总 token 的 15%)。被选中的 token,**不是**全部换成 `[MASK]`,而是按 8:1:1:

- 80% 概率 → 换成 `[MASK]`:「巴黎 是 [MASK] 的 首都」
- 10% 概率 → 换成**随机词**:「巴黎 是 香蕉 的 首都」(逼模型不能盲信原词)
- 10% 概率 → **保持不变**:「巴黎 是 法国 的 首都」(但仍要预测它)

为什么这么折腾?因为下游微调和推理时**根本没有 `[MASK]` 这个符号**。如果训练时被预测的位置永远是 `[MASK]`,模型会过度依赖这个特殊符号,造成**训练/推理失配**(pretrain-finetune discrepancy)。掺入随机词和原词,逼模型对**每个**位置都保持良好表示。

**NSP**。取两句拼成 `[CLS] 句A [SEP] 句B [SEP]`:50% 的 B 是 A 真正的下一句(IsNext),50% 是语料里随机抽的句子(NotNext)。模型用 `[CLS]` 位置的输出向量做二分类。目的是学**句子间关系**(问答、自然语言推理需要)。后来 RoBERTa(2019)发现 NSP 贡献很小甚至有害,去掉它反而更好。

**逐 token 算一遍 15% 是怎么落到具体句子上的**。拿 BERT 的真实输入长度感受一下:一条序列若有 512 个 token,则 $512\times15\%\approx 77$ 个会被选中。这 77 个里:约 $77\times0.8\approx 61$ 个换成 `[MASK]`,约 $77\times0.1\approx 8$ 个换成随机词,约 $8$ 个保持原样。**整条序列只有这 77 个位置贡献交叉熵 loss**,其余 435 个位置前向算了、但不回传 loss——这就是「样本效率低」的直观来源(GPT 这 512 个位置每个都贡献 loss)。

**为什么偏偏是 15%?** 这是个权衡:遮太少(如 5%),每步能学到的监督信号太稀疏、训练慢;遮太多(如 50%),剩下的上下文不够,模型「线索不足」难以还原,任务退化成瞎猜。15% 是 BERT 实验给出的甜点,后续不少工作(如某些高效预训练)发现 40% 上下也可行,但 15% 成了默认惯例。

![[hist-bert-mlm-nsp.svg]]

## 原理

**架构**。BERT 是纯 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|encoder]] 堆叠(BERT-base 12 层、768 维、110M 参数;BERT-large 24 层、340M),**无因果掩码**,每个位置的 [[002 自注意力 Self-Attention|自注意力]] 都能看到整个序列(这就是「双向」的来源)。输入嵌入 = 词嵌入 + 段嵌入(区分句 A/句 B)+ [[030 可学习与相对位置编码|可学习位置编码]](BERT 用的是可学习绝对位置,不是 RoPE,也不是正弦)。

**MLM 损失**。设被遮位置集合 $M$,对每个 $i\in M$,模型在该位置输出隐藏向量 $h_i$,经线性层 + softmax 得词表分布,只在 $M$ 上算 [[32 困惑度 Perplexity|交叉熵]]:

$$\mathcal{L}_\text{MLM}=-\sum_{i\in M}\log p(x_i\mid x_{\setminus M})$$

注意:**只对 15% 被遮的位置算 loss**,所以 MLM 的样本效率天生低于 GPT 的自回归(后者对**每个**位置都算 loss)——这也是后来生成式路线在大规模下更划算的原因之一,详见 [[056 预训练目标：自回归、掩码与去噪|预训练目标对比]]。

**NSP 损失**。`[CLS]` 输出向量 $h_\text{[CLS]}$ 过二分类头:

$$\mathcal{L}_\text{NSP}=-\log p(\text{IsNext}\mid A,B)$$

总损失 $\mathcal{L}=\mathcal{L}_\text{MLM}+\mathcal{L}_\text{NSP}$。

**判别式 vs 生成式**。BERT 是判别式预训练:学的是 $p(\text{被遮词}\mid\text{上下文})$ 和句间关系,用于下游**判别任务**(分类/标注)。它建模的不是整句的联合生成概率,所以不能直接当语言模型采样生成——这是它与 [[036 GPT 系列：自回归与规模化|GPT]] 的根本分野。

**输入的三层嵌入要相加**。BERT 的每个 token 进网络前,嵌入是三份相加:① 词嵌入(WordPiece token);② 段嵌入(Segment,只有 A/B 两种,标记属于句 A 还是句 B);③ 位置嵌入(可学习绝对位置,0…511)。三者**同维相加**后过 LayerNorm + dropout 再进 encoder。段嵌入是 NSP 任务需要区分两句的必备件;若去掉 NSP(如 RoBERTa),段嵌入也可弱化。

## 下游怎么用:四类微调范式

预训练好的 BERT 不能直接生成,但能当**通用特征提取器**接各种下游头(head),只需在顶上加一层、用少量标注数据微调:

- **单句分类**(情感、垃圾邮件):取 `[CLS]` 位置的输出向量 → 一层 Linear + softmax。`[CLS]` 是专门为「整句表示」预留的。
- **句对分类/匹配**(NLI、问答对相关性):`[CLS] 句A [SEP] 句B [SEP]` → `[CLS]` 向量分类。
- **序列标注**(NER、词性):**每个** token 的输出向量 → 各自过同一个分类头,逐 token 打标签(BIO 等)。
- **抽取式问答**(SQuAD):`[CLS] 问题 [SEP] 文章 [SEP]`,对文章里每个 token 预测它是答案 span 的「起点」和「终点」的概率,取 argmax 得起止位置。

⚠️ 易错点:`[CLS]` 向量**不是**天然的好句向量。直接拿 BERT 的 `[CLS]`(或词向量平均)做语义相似度检索效果很差,需要像 Sentence-BERT(2019)那样用孪生网络 + 对比/回归目标**再训练**才好用(详见 [[04 Embedding 与向量数据库|句向量]])。这是工程里反复踩的坑。

## 代码:MLM 数据构造

```python
import torch, random

def mask_tokens(tokens, vocab_size, mask_id, mask_prob=0.15):
    """返回 (输入, 标签)。标签里 -100 表示该位置不算 loss。"""
    labels = tokens.clone()
    probs = torch.full(tokens.shape, mask_prob)
    masked = torch.bernoulli(probs).bool()           # 选中 15%
    labels[~masked] = -100                            # 未选中位置不算 loss

    # ✅ 被选中的 15% 内部再分 80/10/10
    r = torch.rand(tokens.shape)
    to_mask   = masked & (r < 0.8)                    # 80% → [MASK]
    to_random = masked & (r >= 0.8) & (r < 0.9)       # 10% → 随机词
    # 剩 10% 保持原样（但仍在 labels 里算 loss）
    tokens[to_mask] = mask_id
    tokens[to_random] = torch.randint(0, vocab_size, (to_random.sum(),))
    return tokens, labels

# ❌ 误区一：把 15% 全换成 [MASK] —— 造成训练/推理失配，下游掉点
# ❌ 误区二：对所有位置算 loss —— BERT 只在被遮位置算，否则等于让模型抄输入
# ❌ 误区三：拿 BERT 直接做文本生成 —— 它无因果掩码、非自回归，不能逐词采样

toks = torch.tensor([101, 2400, 2003, 2848, 1997, 3007, 102])
inp, lab = mask_tokens(toks.clone(), vocab_size=30522, mask_id=103)
# 只有 lab != -100 的位置参与 cross_entropy
```

```python
# 整条 MLM loss 怎么算:只在被遮位置(label != -100)算交叉熵
import torch.nn.functional as F

def mlm_loss(logits, labels):
    # logits: (B, L, V)，labels: (B, L)，未遮位置为 -100
    # ✅ F.cross_entropy 的 ignore_index=-100 会自动跳过未遮位置
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=-100)
# 关键:分母只数被遮的 ~15% 个 token,不是全部 L 个 —— 这正是样本效率低的根源
```

## 变体谱系:BERT 之后改了什么

零基础的人常把「BERT」当一个点,其实它是一整条改良线,面试常顺着问:

- **RoBERTa**(2019):去 NSP、用更大 batch + 更多数据 + 更长训练 + 动态掩码(每个 epoch 重新随机遮,而非预先遮死);证明「BERT 没训够」,同架构下大幅超原版。
- **ALBERT**(2019):参数共享(各层共用一套权重)+ 嵌入分解(把大词嵌入矩阵拆成 $V\times e$ 与 $e\times d$)大幅省参;把 NSP 换成更难的 SOP(句子顺序预测,判断两句是否被调换顺序)。
- **ELECTRA**(2020):换掉 MLM,改用**替换 token 检测**(RTD)——一个小生成器先把部分 token 替换成看似合理的词,判别器对**每个**位置二分类「这个 token 是不是被换过的」。优势是**所有位置都贡献 loss**(不像 MLM 只 15%),样本效率大涨。
- **DeBERTa**(2020):解耦注意力(把内容与相对位置分开建模)+ 增强的位置编码,刷新一众理解类榜单。

记忆线:**RoBERTa 训得更满 → ALBERT 省参 → ELECTRA 改目标提效率 → DeBERTa 改注意力。**

## 面试高频

- **BERT 为什么用 MLM 而不是预测下一个词?** 双向语言模型若直接预测下一词会泄题(目标词在可见上下文里)。MLM 完形填空既能双向看,又不泄题,得到真正双向的表示。
- **15% 里 80/10/10 的设计为什么?** 下游/推理无 `[MASK]`,若训练时被预测位永远是 `[MASK]` 会造成**训练-推理失配**;掺 10% 随机词(防盲信原词)+ 10% 原词,逼模型对所有位置都学好表示。
- **BERT 能做生成吗?** 不能(不擅长)。它是 encoder-only、无因果掩码、判别式,适合分类/抽取/检索;生成是 GPT 系的活。
- **BERT vs GPT 核心区别?** 双向 vs 单向;掩码(MLM)vs 自回归;判别(理解)vs 生成;encoder-only vs decoder-only。见 [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|架构三类]]。
- **NSP 有用吗?** 帮助句间关系任务,但 RoBERTa 证明其贡献有限,去掉 NSP、用更大 batch 和更多数据训 MLM 反而更好。
- **BERT 用什么位置编码?** 可学习的绝对位置嵌入(不是正弦、不是 RoPE),所以长度受预训练 max_len(512)限制。
- **`[CLS]` 向量能直接做句向量检索吗?** 不能(效果差)。需 Sentence-BERT 式孪生网络再训练才得到好句向量;直接拿 `[CLS]` 或词向量平均做相似度是常见的错误用法。
- **MLM 的 80/10/10 里,「10% 保持原样但仍算 loss」有什么用?** 逼模型对**没被破坏**的位置也维持准确表示,而非「看到原词就直接抄」;再配 10% 随机词防止盲信原词。三者合起来缓解训练-推理失配。
- **ELECTRA 为什么比 BERT 样本效率高?** 它的替换 token 检测对**每个**位置都算 loss,而 MLM 只对 15% 被遮位置算 → 同样数据监督信号更密。
- **BERT 的输入嵌入由几部分组成?** 三部分相加:词嵌入 + 段嵌入(区分句 A/B,NSP 用)+ 可学习位置嵌入。
- **想用 BERT 处理超过 512 的长文本怎么办?** 截断、滑窗分段、或换用专门的长文本模型(Longformer/BigBird,见 [[021 局部与滑窗注意力(Longformer、Mistral SWA)|稀疏注意力]]);可学习绝对位置无法直接外推超过 512。

## 关键事实

- BERT:Devlin、Chang、Lee、Toutanova,2018,arXiv 1810.04805,《Pre-training of Deep Bidirectional Transformers for Language Understanding》。
- 预训练目标:MLM(随机遮 15%,其中 80% 换 `[MASK]`、10% 随机词、10% 原词)+ NSP(下一句二分类)。
- 架构:encoder-only,无因果掩码 → 双向;BERT-base 12 层/110M,BERT-large 24 层/340M;可学习绝对位置编码,max_len=512。
- 判别式:只在被遮位置算 loss,样本效率低于自回归(见 [[056 预训练目标：自回归、掩码与去噪|预训练目标]])。
- RoBERTa(Liu et al., 2019,arXiv 1907.11692)去掉 NSP、加大数据与 batch、用动态掩码,显著超过原版 BERT,说明 NSP 非必需。
- 分词:BERT 用 **WordPiece**(30522 词表),与 GPT 系的 BPE 不同(见 [[051 BPE 与 Byte-level BPE|BPE]] 与 [[050 分词总览与子词动机|分词总览]])。
- 变体:ALBERT(Lan et al., 2019,参数共享 + SOP)、ELECTRA(Clark et al., 2020,替换 token 检测,所有位置算 loss)、DeBERTa(He et al., 2020,解耦注意力);Sentence-BERT(Reimers & Gurevych, 2019,arXiv 1908.10084)把 BERT 改造成可用的句向量编码器。
- 训练规模:BERT 在 BooksCorpus(8 亿词)+ 英文维基(25 亿词)上预训练;微调通常只需几个 epoch、少量标注。
- 历史定位:[[034 发展史总览：2017 到今|发展史]] 中「理解」路线的代表,与 [[036 GPT 系列：自回归与规模化|GPT 生成路线]] 并立;后被生成式 decoder-only 在通用性上超越,但在 [[04 Embedding 与向量数据库|Embedding]]、分类等子任务仍广泛使用。
