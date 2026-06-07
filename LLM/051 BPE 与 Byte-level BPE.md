[[051 BPE 与 Byte-level BPE|BPE 与 Byte-level BPE]] 讲的是最主流的子词算法 **BPE(字节对编码)**:从字符出发,**反复把语料里最高频的相邻对合并成一个新 token**,直到词表到目标大小;**Byte-level BPE**(GPT-2)先把文本变成 UTF-8 字节再跑 BPE,基词表恒为 256、做到**零 OOV**。它是 [[050 分词总览与子词动机|子词分词]]三大流派里 GPT/LLaMA 系用的那支。

## 直觉

BPE 原本是个**数据压缩**算法:反复把出现最多的字节对,替换成一个没用过的新符号,从而变短。Sennrich 等(2016)把它搬到分词上——只是不真的压缩,而是**把「合并了哪些对」记成一张有序规则表**,这张表就定义了子词词表。

学的过程极朴素,**贪心**:

1. 把每个词先拆成**单个字符**(加个词界符如 `_` 标末尾)。
2. 数一遍**所有相邻对**的出现频次,挑**最高频**那一对,合并成一个新 token,记进规则表。
3. 用新 token 重写语料,回到第 2 步,直到词表达到目标大小(或合并次数用完)。

高频组合(像 `es`、`ing`、`th`)会先被合并成整块;只有低频、罕见的组合留作零碎子词。**结果就是:常见词整块、罕见词拆成片**——正好是子词想要的。

**用的时候**(编码新词):按学到的规则表**从头到尾、严格按顺序**套合并,能合就合,最后剩下的片就是 token。因为规则有序且确定,编码可逆、可复现。

一句话:**BPE = 反复合并最高频相邻对学出子词;Byte-level = 在字节上做,基词表 256,任何字符都能切。**

## 例子

**手算一遍**。语料(词 + 词频):`low ×5`、`lower ×2`、`newest ×6`、`widest ×3`。每词先拆字符 + 词界符 `_`。

起点:
```
l o w _        (×5)
l o w e r _    (×2)
n e w e s t _  (×6)
w i d e s t _  (×3)
```

数相邻对频次,挑最高:

- **合并①** `(e,s)`:出现在 `newest`(6) 和 `widest`(3),共 **9** 次,最高 → 合成 `es`。
- **合并②** `(es,t)`:`newest`(6)+`widest`(3)=**9** → 合成 `est`。
- **合并③** `(l,o)`:`low`(5)+`lower`(2)=**7** → 合成 `lo`。
- **合并④** `(lo,w)`:`low`(5)+`lower`(2)=**7** → 合成 `low`。

学到的有序规则:`es → est → lo → low → …`。

现在**编码一个训练时没整词见过的 `lowest`**:按规则套——`l o w e s t _` →(es)`l o w est _` →(est 已合)→(lo)`lo w est _` →(low)`low est _`。结果 `[low, est, _]`,**没 OOV、还共享了 `low` 这个语义片**。这就是 [[050 分词总览与子词动机|子词动机]]里说的好处落地。

![[tok-bpe-合并过程.png]]

**Byte-level 的必要性**。上面在字符上做,基词表是「所有出现过的字符」。可全 Unicode 有十几万字符,且总有没收进去的生僻字/emoji → 仍可能 `[UNK]`。GPT-2 的解法:**先把文本编成 UTF-8 字节**,基词表就恒为 **256**,任何字节串(任意语言、emoji、二进制)都能切,**零 OOV**。GPT-2 词表 `50257 = 256 字节 + 50000 合并 + 1 个 <|endoftext|>`。它还用特殊符号 `Ġ`(可见空格)标记「这个 token 前面有空格」,这样 ` the` 和 `the` 是不同 token,解码能精确还原空格。

**字节级的代价(一个汉字 = 多个 token)**。`你` 的 UTF-8 是 3 个字节 `[228,189,160]`。字节级 BPE 在字节上合并,中文常见字若语料够多会被合成 1~2 个 token,但生僻字就退化成 3 个独立字节 token。所以**中文在字节级 BPE 下普遍比英文费 token**(fertility 高),这是 GPT-2/早期模型「中文又贵又笨」的根源之一;LLaMA-3 把词表扩到 12.8 万、收进更多整块中文/代码片段,正是为缓解这点(见 [[050 分词总览与子词动机|fertility]])。

**WordPiece / Unigram 怎么和 BPE 区分(下一篇详讲)**。三大子词流派选「合并/保留哪些子词」的标准不同:

- **BPE**(GPT/LLaMA):选**绝对频次最高**的相邻对合并。贪心、只看频次。
- **WordPiece**(BERT):选**似然增益最高**的对,得分 ≈ $\dfrac{\text{count}(ab)}{\text{count}(a)\cdot\text{count}(b)}$——倾向合并「单看都不算常见、但总一起出现」的对。
- **Unigram**(T5/SentencePiece):反向——先放一个**大词表**,用语言模型似然反复**删掉贡献最小的子词**,直到目标大小;切分时按概率选最优分法(可给多种切分概率)。

一句话:**BPE 看频次、WordPiece 看似然增益、Unigram 看「删了它似然掉多少」**。细节见 [[052 WordPiece、Unigram 与 SentencePiece|052]]。

![[tok-byte-level-bpe.png]]

## 原理

**1. 训练目标(贪心最大压缩)。** 设语料按词频展开的子词序列集合。每一步选使「合并后总 token 数下降最多」的相邻对——等价于选**当前频次最高的相邻对** $(a,b)$:

$$(a,b)^\star=\arg\max_{(a,b)}\ \mathrm{count}(a,b)$$

合并成新符号 $ab$,加入词表与规则表,重写语料。重复到 $|V|$ 达标。**注意 BPE 只看频次,不算概率**(这点与 [[052 WordPiece、Unigram 与 SentencePiece|WordPiece]] 的似然评分不同)。

**2. 复杂度。** 朴素实现每步要重数全部对,$\mathcal{O}(\text{语料长}\times\text{合并次数})$;实际用「对 → 位置」索引 + 优先队列做增量更新,降到近线性。规则数 = 合并次数 = 目标词表 − 基词表。

**3. 编码的确定性。** 给定规则表(有序),对新词把它拆成基单元,然后**按规则的学习顺序**逐条尝试合并(谁先学的谁先合),直到无可合。顺序固定 ⇒ 同一文本永远切成同一串 token ⇒ 编码解码互逆。

**4. Byte-level 形式化。** 先 $s \mapsto \text{UTF-8}(s)\in\{0,\dots,255\}^*$,基词表 $V_0=\{0,\dots,255\}$(GPT-2 还把字节双射到可打印字符避免空白/控制符问题)。再在字节序列上跑标准 BPE。因 256 字节覆盖一切字节,$P(\text{OOV})=0$。代价:罕见字符按字节展开(一个汉字 UTF-8 占 3 字节,常被切成多个 token),多语言 **fertility** 偏高(见 [[055 分词的坑：数字、代码、多语言与 token 攻击面|分词的坑]])。

**5. pre-tokenization(预切)。** 真正实现里,跑 BPE 前先用正则把文本粗切(按空格/标点/数字),**禁止跨词边界合并**(否则 `dog.` 和 `dog ` 会乱合)。GPT-2 的正则会把数字、空格、标点单独拎出——这也是「数字被切得乱七八糟」坑的根源。

**6. 数字怎么被切坏(具体)。** GPT-2/早期 GPT 的正则不把多位数字当整体,`12345` 可能被切成 `123` + `45` 或别的怪片,导致模型做算术困难。后来不少模型(LLaMA、Mistral)专门**把每个数字单独切成 1 个 token**(digit-level),让 `12345` = `1 2 3 4 5`,数位对齐、算术更稳。这是「为什么有的模型算数好、有的差」的一个分词层面原因(详见 [[055 分词的坑：数字、代码、多语言与 token 攻击面|分词坑]])。

**7. dropout-BPE(正则化变体)。** 训练时偶尔**随机跳过**某些合并规则,让同一个词有时切成不同的子词序列(如 `lowest` 有时 `low+est`、有时 `lo+west`)。这给模型「同义不同切」的鲁棒性,类似数据增强;推理时关掉、用确定性切分。

## 代码

```python
import re, collections

# —— 训练 BPE：在小语料上学合并规则(复现上面的手算)——
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
# 每个词拆成「字符 + 词界符」的元组
vocab = {tuple(list(w) + ["_"]): c for w, c in corpus.items()}

def get_pair_counts(vocab):
    pairs = collections.Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq      # 按词频累加相邻对
    return pairs

def merge(vocab, pair):                                     # 把某对合并进所有词
    a, b = pair
    new_vocab = {}
    for symbols, freq in vocab.items():
        s, out = list(symbols), []
        i = 0
        while i < len(s):
            if i < len(s)-1 and s[i] == a and s[i+1] == b:
                out.append(a + b); i += 2                   # 合并！
            else:
                out.append(s[i]); i += 1
        new_vocab[tuple(out)] = freq
    return new_vocab

rules = []
for _ in range(4):                                         # 学 4 条规则
    pairs = get_pair_counts(vocab)
    best = max(pairs, key=pairs.get)                       # ✅ 选最高频对(不是最高概率)
    rules.append(best)
    vocab = merge(vocab, best)
    print("合并", best, "  频次", pairs[best])
# 输出: ('e','s') 9 / ('es','t') 9 / ('l','o') 7 / ('lo','w') 7

# —— 用规则编码一个新词 ——
def encode(word, rules):
    sym = list(word) + ["_"]
    for a, b in rules:                                     # 严格按学习顺序套
        i = 0
        while i < len(sym)-1:
            if sym[i] == a and sym[i+1] == b:
                sym[i:i+2] = [a + b]
            else:
                i += 1
    return sym
print(encode("lowest", rules))    # ['low', 'est', '_']  —— 训练没整词见过，照样切出来
```

```python
# ❌ 错：以为 BPE 像 WordPiece 那样算「概率得分」选合并
#   best = max(pairs, key=lambda p: pairs[p] / (freq[p[0]] * freq[p[1]]))   # 这是 WordPiece
# ✅ 对：BPE 只看「绝对频次」最高
best = max(get_pair_counts(vocab), key=get_pair_counts(vocab).get)

# Byte-level：先转字节，基词表恒 256，零 OOV
text = "你好🙂"
print(list(text.encode("utf-8")))   # [228,189,160, 229,165,189, 240,159,153,130] 全是 0~255
```

```python
# BPE vs WordPiece 选合并的判据对比(同一组频次,选出来的对可能不同)
pairs_count = {("u","n"): 30, ("n","g"): 80}      # 假设的相邻对频次
unigram = {"u": 40, "n": 200, "g": 90}            # 单 token 频次
# ✅ BPE:绝对频次最高 → 选 (n,g) [80 > 30]
print("BPE 选:", max(pairs_count, key=pairs_count.get))
# ✅ WordPiece:似然增益 count(ab)/(count(a)*count(b)) 最高
score = lambda p: pairs_count[p] / (unigram[p[0]] * unigram[p[1]])
print("WordPiece 选:", max(pairs_count, key=score))
# (u,n): 30/(40*200)=0.00375  vs  (n,g): 80/(200*90)=0.00444 → 这里仍是 (n,g)
# 但当某对「单独都罕见、却总一起出现」时,WordPiece 会优先合并它,BPE 不会
```

## 编码一段「混合输入」走一遍(直觉巩固)

把 BPE 编码想成「查表 + 按序合并」,对 `low newer` 这种含已学子词的串:

1. pre-tokenization 按空格粗切 → `low`、`newer`(各自带词界/前导空格标记)。
2. 各自拆成基单元,按学到的有序规则套合并:`low` → 直接命中已学的 `low` 这个 token;`newer` → `new`+`er`(若 `new`、`er` 在词表里)。
3. 查词表得 id,拼成序列。
4. 解码:id → 子词字符串 → 直接拼接(字节级下还原 `Ġ`/前导空格)→ 原文。

**为什么可逆**:规则有序 + pre-tokenization 边界固定 → 同一文本永远切成同一串;解码只是「把子词字符串接起来」,无歧义。

## 面试高频

- **Q:BPE 的训练过程一句话说清?** A:把词拆成字符,反复统计所有相邻对的**频次**,把**最高频**那对合并成新 token 记入规则表,重写语料,循环到词表达标。贪心、只看频次。
- **Q:BPE 怎么处理没见过的词?** A:按学到的有序合并规则把新词拆成已知子词(最坏退化到单字符/字节),所以不产生 `[UNK]`(字节级更是零 OOV)。
- **Q:BPE 和 WordPiece 选合并的标准有何不同?** A:BPE 选**绝对频次最高**的对;WordPiece 选**似然增益(得分 = 频次÷两部分频次之积)**最高的对,倾向合并「单独看都不常见、但常一起出现」的对(见 [[052 WordPiece、Unigram 与 SentencePiece|WordPiece]])。
- **Q:什么是 Byte-level BPE,解决了什么?** A:先把文本编成 UTF-8 字节再跑 BPE,基词表恒为 256,任何字符(emoji、生僻字、二进制)都能切,**零 OOV**;GPT-2 起广泛使用。代价是非拉丁字符被拆成多字节 token,fertility 偏高。
- **Q:GPT-2 词表 50257 怎么来的?** A:256 个字节基 token + 50000 次合并产生的子词 + 1 个 `<|endoftext|>` 特殊 token。
- **Q:Ġ(或 ▁)是什么?** A:标记「token 前有空格」的可见符号。把空格信息编进 token,使 ` the` 与 `the` 区分开、解码能精确还原空白。Ġ 是 GPT-2/BPE 用法,▁(U+2581)是 SentencePiece 用法。
- **Q:为什么有的模型算术差?** A:分词把多位数字切碎(`12345`→怪片),数位不对齐 → 算术难学。对策:digit-level 切分(每数字 1 token),LLaMA/Mistral 等采用。
- **Q:dropout-BPE 是什么?** A:训练时随机跳过部分合并规则,让同词有多种切法,做正则化/增强;推理时关掉用确定性切分。
- **Q:tiktoken 是什么?** A:OpenAI 的快速 BPE 库(GPT-3.5/4 用 cl100k_base),Rust 实现;LLaMA-3 也改用类似的 tiktoken 风格 BPE。
- **陷阱**:① BPE 切分依赖 pre-tokenization 正则,数字/代码常被切碎(见 [[055 分词的坑：数字、代码、多语言与 token 攻击面|分词坑]]);② 合并规则**有序**,乱序套用会切错;③ 字节级下「一个汉字 = 多个 token」,中文上下文更费 token。

## 关键事实

- BPE 用于子词分词出自 **Sennrich、Haddow、Birch《Neural Machine Translation of Rare Words with Subword Units》(ACL 2016,arXiv:1508.07909)**,把数据压缩算法 BPE(Gage 1994)改造为子词学习,解决罕见词/OOV。
- **Byte-level BPE** 由 GPT-2(Radford 等,2019,《Language Models are Unsupervised Multitask Learners》)引入:基词表 256 字节,词表 50257 = 256 + 50000 合并 + 1 个 `<|endoftext|>`,用 `Ġ` 标前导空格,正则 pre-tokenization。
- BPE 只用频次、贪心合并;不建概率模型(与 WordPiece/Unigram 对比,见 [[052 WordPiece、Unigram 与 SentencePiece|052]])。
- 采用方:GPT-2/3/4、RoBERTa、LLaMA(SentencePiece-BPE)、Mistral、Falcon 等大量主流 LLM 用 BPE 或其字节级变体。
- 工程参考实现:`rsennrich/subword-nmt`(原版)、Hugging Face `tokenizers`(Rust,快)、OpenAI `tiktoken`(GPT-3.5/4 的 cl100k_base BPE,LLaMA-3 也用类似风格)。
- 字节级代价:中文/生僻字按 UTF-8(常 3 字节)展开,fertility 偏高、上下文更费;LLaMA-3 扩词表(12.8 万)收更多整块缓解。
- 数字处理:GPT-2 正则会切碎多位数字 → 算术弱;LLaMA/Mistral 等改 digit-level(每数字 1 token)对齐数位。
- 三大流派判据:BPE 看绝对频次、WordPiece 看似然增益 $\frac{\text{count}(ab)}{\text{count}(a)\text{count}(b)}$、Unigram 看「删除该子词的似然损失」(见 [[052 WordPiece、Unigram 与 SentencePiece|052]])。
- 正则化变体:BPE-dropout(Provilkov et al., 2019)训练随机跳合并做增强;Ġ(BPE)/▁(SentencePiece)标前导空格。
- 关联:动机与对比 [[050 分词总览与子词动机|分词总览]];另两大流派 [[052 WordPiece、Unigram 与 SentencePiece|WordPiece、Unigram 与 SentencePiece]];词表与特殊 token [[053 词表、特殊 token 与对话模板|词表与特殊 token]];实际坑 [[055 分词的坑：数字、代码、多语言与 token 攻击面|分词的坑]]。
