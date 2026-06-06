[[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only]]:Transformer 的三种用法——**Encoder-Only**(BERT,双向、擅理解)、**Decoder-Only**(GPT,因果、擅生成、当今 LLM 主流)、**Encoder-Decoder**(T5/BART,靠 [[012 交叉注意力 Cross-Attention|交叉注意力]]桥接,擅"输入→输出"转换)。

## ① 直觉:差别只在"谁能看谁"

三类架构用的都是同一块 Transformer 砖(多头注意力 + [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]] + [[009 残差连接与梯度流|残差]] + [[010 层归一化：Pre-LN 与 Post-LN|LayerNorm]]),区别只在**注意力的可见范围**和**有没有交叉注意力**:

- **Encoder-Only**:用**双向**自注意力,每个词能看到左右全文(没有 [[007 因果掩码与 padding 掩码|因果掩码]])→ 适合"读懂整段再做判断"的**理解**任务。
- **Decoder-Only**:用**因果**自注意力,每个词只能看左边(下三角掩码)→ 天然适合逐词**生成**。
- **Encoder-Decoder**:编码器双向读源句,解码器因果生成目标句,中间用[[012 交叉注意力 Cross-Attention|交叉注意力]]让解码器去"查"源句 → 适合**输入和输出是两段不同序列**的转换任务(翻译、摘要)。

一句话:**Encoder 看全文做表示,Decoder 只看左侧做生成,Encoder-Decoder 两者兼具**。

![[tf-三类架构.svg]]

## ② 例子:同一句话三种模型怎么处理

输入 "the cat sat on the [?]"。

- **BERT(Encoder-Only)**:把句子里某个词换成 `[MASK]`,如 "the cat [MASK] on the mat",任务是用**左右双向**上下文还原 "sat"。它输出的是每个 token 的**上下文向量**,可拿去做分类、NER、或当 [[04 Embedding 与向量数据库|句向量]]做检索;但它**不会续写**。
- **GPT(Decoder-Only)**:看 "the cat sat on the",**只用左侧**预测下一个词 "mat",再把 "mat" 接上去预测再下一个……[[036 GPT 系列：自回归与规模化|自回归]]地一路生成。
- **T5(Encoder-Decoder)**:把任务写成 "translate English to French: the cat sat" 喂编码器,解码器逐词吐出 "le chat assis…",生成每个法语词时通过交叉注意力回看英文源句对齐。

## ③ 原理:三类的注意力结构与适用任务

| 维度 | Encoder-Only | Decoder-Only | Encoder-Decoder |
|---|---|---|---|
| 代表 | BERT、RoBERTa、ELECTRA | GPT、LLaMA、Qwen | T5、BART、原版 Transformer |
| 自注意力 | 双向(无掩码) | 因果(下三角掩码) | 编码器双向 + 解码器因果 |
| 交叉注意力 | 无 | 无 | 有(解码器→编码器) |
| 训练目标 | MLM(还原 `[MASK]`) | 下一词预测(自回归) | 去噪 / span 重建 / seq2seq |
| 擅长 | 理解:分类、NER、检索、抽取 | 生成:对话、写作、代码、通用 LLM | 转换:翻译、摘要、改写 |
| 能否生成 | 不能(非自回归) | 能 | 能 |

**为什么现在 LLM 几乎都是 Decoder-Only?**

1. **规模化最简单**:只有一种 block、一个目标(预测下一词),数据只需海量原始文本(无需配对),最容易堆参数和数据。
2. **任务统一**:理解、生成、翻译都能写成"给 prompt 续写"的形式(in-context learning),不再需要为每类任务换架构。
3. Encoder-Decoder 的"回看源句"能力,Decoder-Only 用**把源句直接拼进 prompt** 来近似替代(靠因果自注意力扫到左侧的源句),省掉了单独的交叉注意力模块。

但 Encoder-Decoder 在**输入输出明显是两段、且输入需充分双向编码**的任务(机器翻译、长文摘要)上仍有结构优势——编码器能无掩码地双向理解源句,信息更充分。

**三类的训练目标拆开看(零基础该懂"它们各自怎么学")**:
- **Encoder-Only(BERT)= MLM**:随机把 15% 的 token 换成 `[MASK]`,让模型用**双向**上下文还原。因为能看左右,所以不能做"预测下一词"(会偷看右边)。早期 BERT 还有 NSP(判断两句是否相邻),后被证明帮助有限,RoBERTa 去掉了。
- **Decoder-Only(GPT)= 下一词预测(CLM)**:用因果掩码,逐位置预测下一个 token,损失是每个位置的交叉熵之和(见 [[30 交叉熵与负对数似然|交叉熵]])。**每个 token 都是一次监督信号**,数据利用率高(一句 $L$ 词 = $L$ 个训练样本),这是它易规模化的隐藏原因。
- **Encoder-Decoder(T5)= span corruption**:把源句里若干连续 span 挖掉换成哨兵 token,解码器重建被挖的内容;BART 用更多样的去噪(打乱、删除、填充)。

**"为什么 Decoder-Only 赢了"的更深层原因**:
1. **训练信号密度**:CLM 每个位置都产生 loss($L$ 个);MLM 只对被 mask 的 15% 位置产生 loss,数据利用率约为前者 1/7。
2. **无需配对数据**:CLM 只要原始文本;Encoder-Decoder 的 seq2seq 目标在预训练时需构造源-目标对(虽可自监督构造,但更绕)。
3. **in-context learning 涌现**:GPT-3 发现 Decoder-Only 在规模够大时能"看几个例子就学会新任务"(few-shot),这是统一成 prompt 续写的红利,Encoder-Only 不具备生成能力故没有。

**双向 vs 因果的本质权衡**:双向(BERT)看得全、表示更强,但**不能自回归生成**;因果(GPT)能生成,但每个 token 只见左侧、表示偏弱。鱼与熊掌——这就是为什么"理解任务"历史上偏爱 BERT,而"通用 LLM"走 GPT 路线后,靠超大规模把"表示偏弱"补了回来。

![[tf-交叉注意力连线.svg]]

## ④ 代码:用掩码区分三类的可见范围

```python
import numpy as np

def softmax(x): 
    e = np.exp(x - x.max(-1, keepdims=True)); return e/e.sum(-1, keepdims=True)

seq = 4
scores = np.random.randn(seq, seq)

# Encoder-Only:双向,谁都能看谁 → 不加因果掩码
w_encoder = softmax(scores)                       # 上三角非 0

# Decoder-Only:因果,只看左侧 → 加下三角掩码
causal = np.triu(np.ones((seq, seq)), k=1).astype(bool)
w_decoder = softmax(np.where(causal, -1e9, scores))  # 上三角全 0

print("encoder 第0行能看后文:", (w_encoder[0,1:] > 0).any())  # True
print("decoder 第0行只看自己:", np.allclose(w_decoder[0,1:], 0))  # True
```

```python
# Hugging Face 里三类对应不同基类:
# Encoder-Only:  AutoModel / BertModel           → 输出 last_hidden_state(表示)
# Decoder-Only:  AutoModelForCausalLM / GPT2LMHeadModel → .generate() 续写
# Encoder-Decoder: T5ForConditionalGeneration / BartForConditionalGeneration
#                  forward(input_ids=源, labels=目标),解码器内部有 cross-attention
```

## 面试高频

- **Q:BERT、GPT、T5 各属哪类,擅长什么?** BERT=Encoder-Only(双向,理解类:分类/NER/检索);GPT=Decoder-Only(因果,生成类);T5=Encoder-Decoder(转换类:翻译/摘要)。
- **Q:为什么 BERT 不能直接做文本生成?** 它是双向、非自回归的,训练目标是还原 `[MASK]` 而非预测下一词,没有"只看左侧逐词生成"的机制。
- **Q:为什么现代 LLM 偏好 Decoder-Only?** 架构单一、目标单一(下一词预测)、数据无需配对,最易规模化;且各类任务可统一成 prompt 续写。
- **Q:Decoder-Only 怎么替代交叉注意力?** 把源内容直接拼进 prompt,用因果自注意力扫到左侧上下文来"读"源句,无需独立的交叉注意力模块。
- **Q:Encoder-Decoder 还有什么不可替代的优势?** 编码器能无掩码双向充分编码源序列,在机器翻译、长文摘要等"输入输出两段"任务上结构更对口。
- **Q:三类共享哪些组件?** 都用多头注意力、FFN、残差、LayerNorm;只差注意力可见范围(掩码)和有无交叉注意力。
- **Q:三类的训练目标分别是什么?** Encoder-Only=MLM(还原 `[MASK]`,双向);Decoder-Only=下一词预测(CLM,因果);Encoder-Decoder=去噪/span 重建(T5 span corruption、BART 多样去噪)。
- **Q:为什么 CLM 的数据利用率比 MLM 高?** CLM 每个位置都产生 loss($L$ 个监督信号);MLM 只对被 mask 的约 15% 位置产生 loss,利用率约 1/7。这是 Decoder-Only 易规模化的隐藏原因之一。
- **Q:Prefix-LM 是哪一类?** 介于之间——前缀双向、生成部分因果(用 prefix 掩码,见 007),如 GLM、UniLM,试图兼顾理解与生成。
- **Q:BERT 的 NSP 是什么,为什么后来被去掉?** Next Sentence Prediction,判断两句是否相邻;RoBERTa 实验证明它帮助有限甚至有害,遂去掉只留 MLM。
- **Q:既然双向表示更强,为什么通用 LLM 不用 Encoder-Decoder?** 双向不能自回归生成;Decoder-Only 靠超大规模 + in-context learning 补足表示,并统一所有任务为 prompt 续写,工程上更简单、更易 scaling。

## 关键事实

- 原始 Transformer 即 Encoder-Decoder,为机器翻译设计(Vaswani et al., 2017)。
- BERT(Encoder-Only,Devlin et al., 2018)用 MLM + 双向注意力做理解。
- GPT 系列(Decoder-Only,Radford et al., 2018/2019;GPT-3, 2020)用自回归预测下一词,推动了 Decoder-Only 成为 LLM 主流。
- T5(Raffel et al., 2019)与 BART(Lewis et al., 2019)是 Encoder-Decoder,以去噪/span 重建为目标,依赖交叉注意力路由源句信息。详见 [[037 T5 与 BART：去噪 encoder-decoder]]。
