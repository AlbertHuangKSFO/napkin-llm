[[052 WordPiece、Unigram 与 SentencePiece|WordPiece、Unigram 与 SentencePiece]] 讲的是 [[051 BPE 与 Byte-level BPE|BPE]] 之外的两大子词算法和那个常被误当算法的工具:**WordPiece**(BERT 用,按**似然增益**而非纯频次选合并)、**Unigram LM**(Kudo 2018,从大词表**剪枝**、有概率模型、能采样多切)、**SentencePiece**(把空格也当字符、直接吃原始文本的**框架**,内部可跑 BPE 或 Unigram)。

## 直觉

[[051 BPE 与 Byte-level BPE|BPE]] 是「自底向上、贪心合并最高频对」。另两支换了思路:

- **WordPiece**(BERT):也是自底向上合并,但**选谁合并的标准不同**。BPE 选「出现最多的对」;WordPiece 选「**最值得合的对**」——即合并后让训练语料的**似然**涨得最多。直觉:`(un, ##able)` 这种,虽然一起出现频次高,但 `un` 和 `able` 各自也到处出现、单独很有用,合了反而亏;而 `(h, ##ug)` 这种 `h`、`ug` 单独都罕见、几乎只一起出现,合了划算。它给「**单独罕见、却总黏在一起**」的对更高分。
- **Unigram LM**(Kudo 2018):反着来,**自顶向下剪枝**。先搞一个**很大的候选子词表**,给每个子词学一个**概率** $p(\text{子词})$,然后反复**删掉「删了之后总似然损失最小」的那批子词**,缩到目标大小。它是真正的**概率模型**:一个词可以有多种切法,每种切法有个总概率,取最优(或训练时**随机采样**做正则化)。
- **SentencePiece**:不是第四种算法,是个**工具/框架**。它解决一个工程痛点——前面算法都假设「先按空格分词」,可中文/日文/泰文没空格。SentencePiece **把空格也当普通字符**(用 `▁` 表示),直接吃**原始字符串**,语言无关、端到端可逆;内部你可以选 BPE 或 Unigram 当算法。

一句话:**WordPiece = 按似然选合并的 BPE 表亲;Unigram = 从大表概率剪枝、能采样;SentencePiece = 把空格当字符、吞原文的框架,壳里装 BPE 或 Unigram。**

## 例子

**WordPiece 的评分**。语料里假设统计到:`hug` 整体出现频次让 `(h, ug)` 的频次 = 10;而 `h` 单独频次 = 12、`ug` 单独频次 = 11。BPE 会因 `(h,ug)` 频次 10 不算最高而未必先合;WordPiece 算**得分**:

$$\text{score}(h,ug)=\frac{\mathrm{count}(h,ug)}{\mathrm{count}(h)\cdot\mathrm{count}(ug)}=\frac{10}{12\times 11}\approx 0.076$$

对比一个**高频但各自常见**的对 `(t, ##he)`:频次 50,但 `t`、`he` 各自频次都上千,得分 $\frac{50}{1000\times 1200}\approx 4\times10^{-5}$,**远低**。所以 WordPiece **不会**先合 `(t,##he)`——这正是它和 BPE 行为分叉的地方:**分母惩罚了「各自就很常见」的对**。BERT 里续接子词带 `##` 前缀:`playing` → `[play, ##ing]`,`##` 表示「接在前一个子词后面、不带空格」。

**Unigram 的多切与采样**。给定词 `unhappily` 和已学好的子词概率,可能的切法有 `[un, happ, ily]`、`[un, happily]`、`[unhapp, ily]` 等。每种切法的概率 = 各子词概率连乘:

$$P([un, happ, ily]) = p(un)\,p(happ)\,p(ily)$$

推理时取**概率最高**的那种切;训练时按概率**随机采样**一种(subword regularization),让模型见过同一个词的多种切法、更鲁棒。BPE/WordPiece 给定输入**只有一种确定切法**,这是 Unigram 独有的能力。

![[tok-三算法对比.png]]

**SentencePiece 的空格处理**。句子 `"Hello world"` → SentencePiece 先把空格变成 `▁`:`▁Hello▁world`,再当字符序列跑算法,可能切成 `[▁Hello, ▁world]`。解码时把 `▁` 换回空格,**完美还原**(包括开头是否有空格)。中文 `"你好世界"` 没空格也照切不误,因为它根本不依赖空格分词。

![[tok-三算法对比.png]]

## 原理

**1. WordPiece 选择准则(似然增益)。** 把语言建模看成「子词独立」近似,合并对 $(a,b)$ 带来的对数似然增益正比于

$$\text{score}(a,b)=\frac{\mathrm{count}(a,b)}{\mathrm{count}(a)\,\mathrm{count}(b)}$$

每步选 score 最高的对合并。分子是「一起出现」,分母是「各自出现」——奖励「几乎只一起出现」的对,惩罚「各自就泛滥」的对。这与 BPE 的 $\arg\max\,\mathrm{count}(a,b)$ 不同(见 [[051 BPE 与 Byte-level BPE|BPE]])。**编码**用「最长匹配优先(greedy longest-match)」:从词首尽量取词表里最长的子词。

**2. Unigram LM(自顶向下 + EM)。** 假设每个子词 $x_i$ 有概率 $p(x_i)$,一种切分 $\mathbf{x}=(x_1,\dots,x_m)$ 的概率

$$P(\mathbf{x})=\prod_{i=1}^{m}p(x_i),\qquad \sum_{x\in V}p(x)=1$$

一个词的概率 = 所有合法切分概率之和(或取最优切分,Viterbi)。训练用 **EM**:E 步对每个词算各切分的后验、M 步更新 $p(x)$。然后**剪枝**:对每个候选子词,估计「删掉它后整个语料似然的下降量」,删掉下降最小的一批,重复 EM + 剪枝直到 $|V|$ 达标。**子词正则化**(Kudo 2018):训练时对每个句子从 $P(\mathbf{x})$ **采样**一种切分(而非永远取最优),等价数据增强,提鲁棒。

**3. SentencePiece(框架,非算法)。** 三个工程点:① **把空格当字符**,统一用 `▁`(U+2581)替换,于是「分词」不再需要语言相关的预切规则,中日泰等无空格语言一视同仁;② **直接处理原始字符串**,`encode`/`decode` 端到端**无损可逆**(detokenize 完美还原空白);③ **算法可插拔**,`--model_type=bpe` 或 `unigram`。所以「SentencePiece-BPE」「SentencePiece-Unigram」指的是「用 SP 这个壳 + 某算法」。

**4. 三者关系小结。** BPE 与 WordPiece 都是**合并式(自底向上)**,区别只在「选合并的打分」;Unigram 是**剪枝式(自顶向下)+ 概率模型**,独有「多切/采样」;SentencePiece 与它们**正交**,是承载算法的工具。现代 LLM 常见组合:GPT/RoBERTa = (byte-level) BPE;BERT = WordPiece;T5/ALBERT/XLNet = SentencePiece-Unigram;LLaMA = SentencePiece-BPE。

**5. WordPiece 评分的推导(为什么是「频次 ÷ 频次积」)。** 把语料的对数似然在「子词独立(unigram)」近似下写出来:每个 token 出现概率 $p(t)=\mathrm{count}(t)/N$($N$ 为总 token 数)。合并 $(a,b)\to ab$ 前,语料对这两个单元的对数似然贡献含 $\mathrm{count}(a)\log p(a)+\mathrm{count}(b)\log p(b)$;合并后变成 $\mathrm{count}(ab)\log p(ab)$。把似然增量 $\Delta\ell$ 对各项展开、丢掉与待选对无关的常数,整理后**增益的主导项正比于** $\log\frac{p(ab)}{p(a)p(b)}=\log\frac{\mathrm{count}(a,b)\cdot N}{\mathrm{count}(a)\,\mathrm{count}(b)}$。$N$ 对所有候选对一样,故排序时只看

$$\frac{\mathrm{count}(a,b)}{\mathrm{count}(a)\,\mathrm{count}(b)}$$

这就是 score 公式的来历——本质是**点互信息(PMI)**的同款形式:合并那些「比独立假设下更常共现」的对。BPE 的 $\arg\max\,\mathrm{count}(a,b)$ 则是**不带分母**的退化版,等于假设 $p(a)p(b)$ 对所有对都一样。

**6. Unigram 的 Viterbi 与前向算法。** 「一个词的概率 = 所有切分概率之和」用**前向(forward)**动态规划算:设 $\alpha_j$ 为「前 $j$ 个字符」的所有切分概率和,$\alpha_0=1$,$\alpha_j=\sum_{i<j,\,x[i:j]\in V}\alpha_i\,p(x[i:j])$,词总概率 $=\alpha_{|x|}$。「取最优切分」则把求和换成 max,即 **Viterbi**:$\beta_j=\max_{i<j}\beta_i\,p(x[i:j])$,回溯得最优切法。两者都是 $O(|x|^2)$(或限定最大子词长度后近线性)。EM 的 E 步正是用前向-后向算每个子词在各词上的期望出现次数。

![[tok-Unigram切分格子.png]]

**7. WordPiece 编码:最长匹配的失败回退。** WordPiece 编码用「贪心最长匹配」:从词首尽量取词表里最长的子词,匹配不上就退一格。**关键易错**:若某个字符在词表里根本没有(连单字符都缺),整个词标 `[UNK]`——这与 BPE 不同,BPE 退到字符/字节级几乎不产 UNK,字节级更是零 OOV。这也是 WordPiece 词表必须含全部基础字符的原因。

## 代码

```python
import math, collections

# —— WordPiece vs BPE 的「选对标准」对比 ——
count_pair  = {("h","ug"):10, ("t","##he"):50}      # 一起出现的频次
count_token = {"h":12, "ug":11, "t":1000, "##he":1200}  # 各自出现的频次

def bpe_score(pair):                                  # ✅ BPE：只看绝对频次
    return count_pair[pair]
def wordpiece_score(pair):                            # ✅ WordPiece：似然增益
    a, b = pair
    return count_pair[pair] / (count_token[a] * count_token[b])

for p in count_pair:
    print(p, "BPE分=", bpe_score(p), " WP分=", round(wordpiece_score(p), 6))
# (t,##he) BPE 分最高(50) → BPE 先合它
# (h,ug)   WP  分最高(0.076 ≫ 4e-5) → WordPiece 先合它(各自罕见、总黏一起)
print("BPE 先合:", max(count_pair, key=bpe_score))
print("WP  先合:", max(count_pair, key=wordpiece_score))   # 两者结论不同！
```

```python
# —— Unigram：一个词的多种切法 + 取最优 ——
p = {"un":0.02, "happ":0.005, "ily":0.01, "happily":0.004,
     "unhapp":0.001, "y":0.03, "il":0.008}            # 已学好的子词概率
def seg_prob(seg):                                     # 切法概率 = 各子词概率连乘
    pr = 1.0
    for s in seg:
        if s not in p: return 0.0
        pr *= p[s]
    return pr
cands = [["un","happ","ily"], ["un","happily"], ["unhapp","ily"]]
for c in cands:
    print(c, "P=", seg_prob(c))
best = max(cands, key=seg_prob)
print("Unigram 推理取最优切法:", best)                  # 概率最高的那种
# 训练时则按 P 随机采样一种(subword regularization)

# ❌ 误区：把 SentencePiece 当成第四种算法
#   "用了 SentencePiece" 没说清算法 —— 它内部还是 BPE 或 Unigram
# ✅ 正确：SentencePiece 是工具，--model_type=bpe / unigram 才指定算法
```

```python
# —— Unigram 的前向(求和)与 Viterbi(求最优)对比 ——
import math
p = {"un":0.02,"happ":0.005,"ily":0.01,"happily":0.004,
     "unhapp":0.001,"y":0.03,"il":0.008,"h":0.05,"a":0.06,"pp":0.004}
word = "unhappily"
V = set(p); maxlen = max(len(s) for s in V)

def forward_total(x):                      # 所有切分概率之和(前向 DP)
    n = len(x); alpha = [0.0]*(n+1); alpha[0] = 1.0
    for j in range(1, n+1):
        for i in range(max(0, j-maxlen), j):
            sub = x[i:j]
            if sub in p: alpha[j] += alpha[i] * p[sub]
    return alpha[n]

def viterbi_best(x):                       # 最优切分(Viterbi + 回溯)
    n = len(x); dp = [-1e30]*(n+1); bk = [0]*(n+1); dp[0] = 0.0
    for j in range(1, n+1):
        for i in range(max(0, j-maxlen), j):
            sub = x[i:j]
            if sub in p and dp[i] + math.log(p[sub]) > dp[j]:
                dp[j] = dp[i] + math.log(p[sub]); bk[j] = i
    seg, j = [], n
    while j > 0: seg.append(x[bk[j]:j]); j = bk[j]
    return seg[::-1], math.exp(dp[n])

print("P(word) 求和 =", forward_total(word))   # 边缘化所有切法
print("Viterbi 最优 =", viterbi_best(word))     # 推理实际用这个
# 训练时 EM 用前向-后向;推理用 Viterbi;subword regularization 则按后验采样
```

## 面试高频

- **Q:WordPiece 和 BPE 的核心区别?** A:都自底向上合并子词,但选合并的标准不同。BPE 选**绝对频次最高**的对;WordPiece 选**似然增益最高**(得分 = 频次 ÷(两部分频次之积))的对,倾向合并「各自罕见、却几乎只一起出现」的对。WordPiece 还用 `##` 标续接子词。
- **Q:Unigram LM 怎么工作?和前两者最大不同?** A:**自顶向下**——先建大候选词表,用 EM 学每个子词的概率,再反复剪掉「删掉后似然损失最小」的子词到目标大小。最大不同:它有**概率模型**,一个词可多种切法,推理取最优、训练可**采样**(子词正则化),前两者给定输入只有唯一确定切法。
- **Q:SentencePiece 是一种分词算法吗?** A:不是,是**工具/框架**。贡献是「把空格当普通字符(用 ▁)、直接吃原始字符串、端到端可逆、语言无关」,内部算法可选 BPE 或 Unigram。说「用 SentencePiece」还要补一句用的是哪种算法。
- **Q:为什么无空格语言(中文/日文)偏爱 SentencePiece?** A:BPE/WordPiece 默认先按空格预切,中日泰没空格就没法预切。SentencePiece 不依赖空格、把整段原文当字符流处理,天然支持。
- **Q:子词正则化是什么,有什么用?** A:Unigram 在训练时对同一句子按切分概率**随机采样**不同切法(而非永远最优),相当于数据增强,让模型对切分扰动更鲁棒、抗噪、利于低资源/多语言(Kudo 2018)。
- **Q:各模型用什么?** A:GPT/RoBERTa = (byte-level) BPE;BERT/ELECTRA = WordPiece;T5/ALBERT/XLNet/mBART = SentencePiece-Unigram;LLaMA = SentencePiece-BPE。
- **Q:WordPiece 编码会产生 UNK 吗?和 BPE 比?** A:会——贪心最长匹配若连某个基础字符都不在词表,整词标 `[UNK]`;BPE 退到字符/字节级几乎不产 UNK,字节级 BPE 零 OOV。所以 WordPiece 词表必须覆盖全部基础字符。
- **Q:WordPiece 的 score 公式怎么推出来的?** A:在「子词独立」近似下写语料对数似然,合并 $(a,b)$ 的似然增益主导项 $\propto\log\frac{p(ab)}{p(a)p(b)}$,这正是**点互信息 PMI**;丢掉与待选无关的常数后,排序只看 $\frac{\mathrm{count}(a,b)}{\mathrm{count}(a)\mathrm{count}(b)}$。BPE 是去掉分母的退化版。
- **Q:Unigram 怎么算「一个词的概率」?推理和训练分别用什么算法?** A:用**前向 DP** 把所有切分概率求和得边缘概率(EM 训练用前向-后向求期望);推理用 **Viterbi** 取最优切分;子词正则化训练时则按切分后验**采样**。
- **陷阱**:① `##`(WordPiece)与 `▁`/`Ġ`(SP/GPT)含义相反——`##` 标「续接、无空格」,`▁`/`Ġ` 标「前有空格」;② 别把 SentencePiece 当算法;③ Unigram 的采样只在训练开,推理用确定最优切;④ WordPiece 缺基础字符会整词 UNK;⑤ Unigram 概率连乘易下溢,实现要在 log 域算。

## 关键事实

- **WordPiece** 最早用于 Google 语音/日韩分词(Schuster & Nakajima 2012),因 **BERT**(Devlin 等,2018,arXiv:1810.04805)广为人知;选合并按似然增益,得分 = 频次 ÷(两部分频次之积);续接子词加 `##`;特殊 token `[CLS] [SEP] [MASK] [PAD] [UNK]`。
- **Unigram LM** 出自 **Kudo《Subword Regularization》(ACL 2018,arXiv:1804.10959)**:大词表 + EM + 剪枝,概率化分词,提出子词正则化(训练采样多切法)。
- **SentencePiece** 出自 **Kudo & Richardson(EMNLP 2018 demo,arXiv:1808.06226)**,Google 开源(`google/sentencepiece`):把空格当字符(`▁`)、直接处理原始文本、可逆、语言无关,`--model_type` 选 bpe/unigram;为多语言 NMT(100+ 语言对单一分词器)而生。
- 采用对照:BERT=WordPiece;T5、ALBERT、XLNet、mBART、多数多语言模型=SentencePiece-Unigram;LLaMA=SentencePiece-BPE。
- 关联:三者对比的另一支 [[051 BPE 与 Byte-level BPE|BPE]];整体动机 [[050 分词总览与子词动机|分词总览]];词表与特殊 token [[053 词表、特殊 token 与对话模板|词表与特殊 token]];分词器训练流程 [[059 tokenizer 训练|tokenizer 训练]]。
