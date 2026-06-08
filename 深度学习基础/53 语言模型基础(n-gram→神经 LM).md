[[53 语言模型基础(n-gram→神经 LM)|语言模型]]就是给一段文字打分的概率机器:输入前文,输出"下一个词是谁"的概率分布 $P(w_t\mid w_{1..t-1})$——这是从 n-gram 到神经 LM 再到 GPT 的同一条根。

## 直觉

人脑读到"今天天气真"会自然预期"好/不错/糟",这就是语言模型在做的事:**对下一个词的概率分布建模**。

把一整句话的概率拆开,用链式法则一字一字往后写:
$$P(w_1,\dots,w_T)=\prod_{t=1}^{T}P(w_t\mid w_1,\dots,w_{t-1})$$
每一项都是"看着前文猜下一个词"。这种**逐词、只依赖左侧前文**的生成方式就叫**自回归(autoregressive)**——今天的 [[LLM/036 GPT 系列：自回归与规模化|GPT]] 仍然是这个范式,只是把 $P(\cdot\mid\text{前文})$ 换成了 Transformer。

三代技术,解决的都是同一个问题"如何估计 $P(w_t\mid\text{前文})$":
- **n-gram**:用马尔可夫假设把前文截短到固定 $n-1$ 个词,直接数频率。简单,但**词表一大就数不过来**(维度灾难),且对没见过的组合束手无策。
- **神经 LM(Bengio 2003)**:把词映射成稠密向量([[054 词嵌入层与权重绑定|词嵌入]]雏形),用神经网络拟合概率,**靠相似词共享统计强度**,泛化远超 n-gram。
- **RNN / Transformer LM**:不再固定窗口,理论上能看任意长前文。

## 例子

语料只有三句:`我 爱 猫`、`我 爱 狗`、`你 爱 猫`。

**bigram(n=2)** 估 $P(\text{爱}\mid\text{我})$:看"我"出现 2 次,后面接"爱" 2 次 → $\frac{2}{2}=1.0$。
估 $P(\text{猫}\mid\text{爱})$:"爱"出现 3 次,后接"猫" 2 次 → $\frac{2}{3}\approx0.67$。

**数据稀疏的痛点**:问 $P(\text{鱼}\mid\text{爱})$,语料里"爱 鱼"出现 0 次 → 概率 $0$,整句概率连乘后归零。n-gram 必须靠**平滑**(给没见过的组合分一点概率,如 Laplace 加一、Kneser-Ney)硬撑。

**神经 LM 的优势**:若训练中"猫""狗""鱼"的词向量因上下文相似而靠拢,模型见过"爱 猫""爱 狗",就能给"爱 鱼"一个合理的非零概率——这是 n-gram 永远做不到的**软泛化**。

**加一平滑手算(看平滑怎么"借"概率)**。词表里有 $V=5$ 个不同词。问 $P(\text{鱼}\mid\text{爱})$:"爱"出现 3 次,后接"鱼" 0 次。
- 朴素 MLE:$\frac{0}{3}=0$ —— 整句概率连乘后归零,模型直接判定"这句不可能"。
- 加一(Laplace)平滑:$\frac{0+1}{3+5}=\frac18=0.125$ —— 给没见过的组合也分一点概率。代价:把概率从高频组合"匀"给低频,$P(\text{猫}\mid\text{爱})$ 从 $\frac23\approx0.667$ 降到 $\frac{2+1}{3+5}=\frac38=0.375$。加一往往匀得太狠,实际用 Kneser-Ney 这类更精细的平滑。

**困惑度手算(直观理解"分支数")**。模型对一句 4 个词分别给概率 $0.5,0.25,0.5,0.25$。先算平均负对数概率:$-\frac14(\log0.5+\log0.25+\log0.5+\log0.25)$。用以 2 为底:$-\frac14(-1-2-1-2)=\frac64=1.5$,$\text{PPL}=2^{1.5}\approx2.83$。含义:模型每步平均像在 $\approx2.83$ 个等概率选项里犹豫。若模型完美($P=1$ 每步),PPL=1;若纯瞎猜($P=1/V$),PPL=$V$。**PPL 越低,语言模型越好。**

![[rnn-ngram稀疏.png]]

## 原理

**n-gram + 马尔可夫假设**:近似 $P(w_t\mid w_{1..t-1})\approx P(w_t\mid w_{t-n+1..t-1})$,用最大似然(数频率):
$$P(w_t\mid w_{t-n+1..t-1})=\frac{\text{count}(w_{t-n+1..t})}{\text{count}(w_{t-n+1..t-1})}$$
代价:参数量随 $n$ **指数爆炸**($|V|^n$),且大多数 n-gram 在语料里出现 0 次——这正是 Bengio 说的**维度灾难(curse of dimensionality)**。

**神经概率语言模型(Bengio 2003)**:三步走。
1. **查表得词向量**:每个词查一个可学习矩阵 $C\in\mathbb R^{|V|\times m}$,得 $m$ 维向量。前 $n-1$ 个词向量拼成 $x=[C(w_{t-n+1});\dots;C(w_{t-1})]$。
2. **前馈隐层**:$h=\tanh(Hx+d)$。
3. **softmax 输出下一个词分布**:
$$P(w_t\mid\text{前文})=\text{softmax}(Wx+Uh+b)$$
用 [[30 交叉熵与负对数似然|交叉熵]]作损失,梯度下降学 $C,H,W,U$。**词向量 $C$ 是副产品却极有价值**:语义相近的词自动靠拢,这就是后来 word2vec、嵌入层的源头。

**平滑全家(统计 LM 时代的核心)**。n-gram 必须靠平滑对付零频组合:
- **加一 / 加 k(Laplace / Lidstone)**:给每个计数加常数,简单但匀得太均。
- **回退(back-off)**:高阶 n-gram 没见过就退到低阶(trigram→bigram→unigram),用低阶估计兜底。
- **插值(interpolation)**:把各阶 n-gram 概率加权平均(权重可学),比硬回退平滑。
- **Kneser-Ney**:统计 LM 时代最强平滑,核心洞见是用"一个词出现在**多少种不同上下文**里"(continuation count)而非单纯频率来估计低阶概率——能正确处理"Francisco 频率高但几乎只跟在 San 后面"这类情况。

**子词切分(连接统计与神经时代)**。整词词表有「未登录词(OOV)」和「词表爆炸」问题。现代 LM 用 **BPE / WordPiece / Unigram** 把词切成子词单元(如 "playing"→"play"+"ing"),让词表可控、罕见词由子词拼出、几乎没有 OOV——这是从神经 LM 通向 GPT 的工程基石(详见 [[LLM/050 分词总览与子词动机|分词总览与子词动机]])。

**评价指标**:语言模型好坏看 [[32 困惑度 Perplexity|困惑度]]:
$$\text{PPL}=\exp\Big(-\frac1T\sum_t\log P(w_t\mid\text{前文})\Big)$$
即"模型在每步平均要在多少个等概率选项里犹豫",越低越好。它等价于 [[30 交叉熵与负对数似然|交叉熵]]取指数。注意:**PPL 依赖分词方式**(词级 vs 子词级不可直接比),且只衡量"预测下一词"的能力,不直接等于下游任务好坏。

**神经 LM 的训练细节**。损失是逐位置交叉熵(等价最大化条件似然):$\mathcal L=-\frac1T\sum_t\log P_\theta(w_t\mid\text{前文})$。Bengio 2003 的前馈 LM 输出层是对**整个词表**的 softmax,词表一大($|V|=10^5$)softmax 计算就极贵——后续用**层级 softmax**、**噪声对比估计(NCE)**、**负采样**(word2vec)等近似加速,这也是词嵌入训练能 scale 的关键。

![[rnn-神经LM结构.png]]

## 代码

```python
import numpy as np
from collections import defaultdict

corpus = ["我 爱 猫", "我 爱 狗", "你 爱 猫"]
toks = [["<s>"] + s.split() + ["</s>"] for s in corpus]

# ❌ 朴素 bigram MLE:没见过的组合直接 0,整句概率塌成 0
bi = defaultdict(lambda: defaultdict(int)); uni = defaultdict(int)
for s in toks:
    for a, b in zip(s, s[1:]):
        bi[a][b] += 1; uni[a] += 1

def p_naive(prev, w):
    return bi[prev][w] / uni[prev] if uni[prev] else 0.0
print("P(鱼|爱) 朴素 =", p_naive("爱", "鱼"))   # 0.0  → 句子概率归零

# ✅ 加一(Laplace)平滑:未见组合也有非零概率
V = len({w for s in toks for w in s})
def p_smooth(prev, w):
    return (bi[prev][w] + 1) / (uni[prev] + V)
print("P(鱼|爱) 平滑 =", round(p_smooth("爱", "鱼"), 4))  # >0

# 自回归地给整句打分(对数概率,避免下溢)
def sent_logprob(sent, pfn):
    s = ["<s>"] + sent.split() + ["</s>"]
    return sum(np.log(pfn(a, b) + 1e-12) for a, b in zip(s, s[1:]))
print("logP(我 爱 猫) =", round(sent_logprob("我 爱 猫", p_smooth), 4))
# 困惑度 = exp(-logprob / 词数)
lp = sent_logprob("我 爱 猫", p_smooth)
print("PPL =", round(np.exp(-lp / 4), 3))
```

要点:朴素 MLE 一遇未登录组合就崩,**平滑/神经泛化**是语言模型能用的前提;打分一律在对数域累加防下溢。

```python
# 神经 LM 雏形(Bengio 2003)的前向骨架:查表 → 拼接 → 隐层 → softmax
import numpy as np
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()

V, m, n, hdim = 6, 8, 2, 16      # 词表、词向量维、上下文窗(n-1)、隐层
C = np.random.randn(V, m) * 0.1  # 可学习词向量表(副产品却极有价值)
H = np.random.randn(hdim, n*m) * 0.1
U = np.random.randn(V, hdim) * 0.1
W = np.random.randn(V, n*m) * 0.1   # 直连项(可选)

def nlm_prob(ctx_ids):           # ctx_ids: 前 n 个词的 id
    x = np.concatenate([C[i] for i in ctx_ids])   # ① 查表 + 拼接
    h = np.tanh(H @ x)                            # ② 前馈隐层
    logits = W @ x + U @ h                        # ③ 输出(含直连)
    return softmax(logits)                        # 下一个词的分布

p = nlm_prob([0, 1])
print("下一个词分布(和为1):", p.round(3), " sum=", round(p.sum(), 3))
# 词向量 C 训练后:语义相近的词行向量靠拢 → 软泛化的来源
```

## 面试高频

- **"什么是语言模型?自回归是什么意思?"** LM 给序列建概率 $P(w_{1..T})=\prod_t P(w_t\mid w_{<t})$;自回归 = 逐词生成、每步只条件于已生成的左侧前文。GPT 就是大号自回归 LM。
- **"n-gram 的两大缺陷?"** ① 维度灾难,参数 $|V|^n$ 指数增长、绝大多数组合零频;② 无法泛化到语义相近但没共现过的词。靠平滑只能缓解①。
- **"神经 LM 凭什么比 n-gram 强?"** 用稠密词向量共享统计强度,相似词互相"借"概率,实现软泛化;还顺带产出可复用的词嵌入。
- **"困惑度怎么解读?"** 交叉熵的指数,= 模型每步平均的"有效分支数";PPL=1 完美,PPL=词表大小约等于瞎猜。
- **"为什么句子概率要在对数域算?"** 多个 $<1$ 的概率连乘会数值下溢;取对数变连加,稳定且可加。
- **"困惑度为什么不能跨模型直接比?"** PPL 依赖分词(词级 vs 子词级、词表大小不同)和测试集;只有相同分词、相同数据下比较才公平。它也只反映"预测下一词"的能力,不等于下游任务表现。
- **"为什么需要子词切分(BPE)?"** 整词词表有 OOV 和爆炸问题;BPE/WordPiece 把词切成子词,词表可控、罕见词由子词拼出、几乎无 OOV,是现代 LM 的工程基石。
- **"神经 LM 的 softmax 为什么贵?怎么加速?"** 输出层要对整个词表($10^5$+)做 softmax,计算正比于词表大小。加速法:层级 softmax、NCE、负采样(word2vec)、采样 softmax。
- **"自回归 LM 和掩码 LM(BERT)有什么区别?"** 自回归(GPT)逐词、只看左侧、能生成;掩码 LM(BERT)随机遮词、双向看上下文、擅长理解但不能自然生成。本篇的 $P(w_t\mid w_{<t})$ 是自回归范式。

## 关键事实

- **神经概率语言模型**:Bengio, Ducharme, Vincent & Jauvin, *A Neural Probabilistic Language Model*,JMLR vol. 3, pp. 1137-1155(2003)。首次用前馈网络 + 可学习词向量做 LM,显著超越当时最强 trigram,提出"维度灾难"动机,是词嵌入与现代 LM 的源头。
- **n-gram 与平滑**:Kneser-Ney 平滑(1995)是统计 LM 时代最强的平滑法;经典综述见 Jurafsky & Martin《Speech and Language Processing》第 3 章。
- **自回归范式**贯穿至今:RNN-LM(Mikolov 2010)、GPT 系列均是 $P(w_t\mid w_{<t})$ 的逐词建模,只是底层网络从前馈→RNN→Transformer。
- **词嵌入的爆发**:word2vec(Mikolov et al., 2013,CBOW/Skip-gram + 负采样)、GloVe(Pennington et al., 2014)把 Bengio 的"副产品词向量"做成独立工具,语义类比($king-man+woman\approx queen$)轰动。
- **子词切分**:BPE(Sennrich et al., 2016)、WordPiece(Schuster & Nakajima, 2012;BERT 用)、Unigram LM(Kudo, 2018);解决 OOV 与词表爆炸。
- **加速 softmax**:层级 softmax(Morin & Bengio, 2005)、NCE(Gutmann & Hyvärinen, 2010)、负采样(word2vec)。
- 困惑度作为标准评测指标的定义与交叉熵等价,见 [[32 困惑度 Perplexity]]。
