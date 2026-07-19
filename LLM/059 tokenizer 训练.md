[[059 tokenizer 训练|tokenizer 训练]] 讲的是怎么在语料上「训练」一个分词器:**选算法(BPE / Unigram)、定目标词表大小、学出合并规则或子词概率、加特殊 token,然后冻结**。注意这里的「训练」是纯统计、无梯度,和神经网络训练是两回事;关键决策是**词表大小的权衡**。

## 直觉

分词器不是凭空设计的,而是**从语料里学出来的**。给一份有代表性的语料,跑一个算法,它自动找出「哪些字符组合该合并成一个 token」,凑够你设定的词表大小,然后把这套 `vocab + merges` 冻结存下来——之后模型训练和推理都用这套固定的切法。

两件最重要的事:

1. **选算法**:[[051 BPE 与 Byte-level BPE|BPE]](反复合并最高频相邻对)或 [[052 WordPiece、Unigram 与 SentencePiece|Unigram]](从大词表逐步删低价值子词)。GPT/LLaMA 系多用 BPE,T5/ALBERT 用 Unigram。
2. **定词表大小 $|V|$**:这是核心权衡。**小词表**(如 8k):参数省,但序列变长(fertility 高)、算力贵、长依赖难。**大词表**(如 256k):序列短、多语言友好,但嵌入/softmax 层参数暴涨,长尾 token 训练不足易出 [[055 分词的坑：数字、代码、多语言与 token 攻击面|故障 token]]。

一句话:**tokenizer 训练 = 在语料上统计出一套固定切法;词表大小在「序列长度/算力」与「参数量/多语言覆盖」之间做取舍。**

## 例子

**训练流程(五步)。**
```
代表性语料 → 选算法(BPE/Unigram) → 定 |V|=32k…256k → 学合并/概率 + 加特殊 token → 冻结
```
语料要**采样得有代表性**:若是多语言模型,得按目标语言比例混采,否则小语种 fertility 会爆。

![[tok-词表大小权衡.png]]

**词表大小怎么影响序列长度(手算)。** 同一句英文 `"internationalization"`(20 字符):
- 词表小(8k):没学到这个长词的整块,切成 `inter / national / ization` 等 → 3~4 个 token。
- 词表大(100k):可能学到 `internationalization` 整词或 `inter / nationalization` → 1~2 个 token。

token 少一半,序列短一半,注意力 $O(L^2)$ 成本约降到 1/4,同样上下文窗口装得下两倍内容。代价:那个大词表多出来的几万个 token 都要在嵌入矩阵和输出 softmax 里各占一行,$|V|\times d$ 的参数随 $|V|$ 线性涨。

**Unigram 的小数字手算。** 令候选词表的概率为 $p(a)=0.4,p(b)=0.1,p(ab)=0.3,p(ba)=0.2$。字符串 `aba` 有三条合法切分:

$$[a,ba]:0.4\times0.2=0.08,\quad [ab,a]:0.3\times0.4=0.12,\quad [a,b,a]:0.4\times0.1\times0.4=0.016$$

故训练时的句子概率是三者**求和** $P(\texttt{aba})=0.08+0.12+0.016=0.216$;Viterbi 只在编码时选最大的一条 $[ab,a]$(概率 $0.12$)。最大路径不是总概率,更不是 EM 的训练目标。

**经验数值**:英文 ~32k 词表足够;多语言模型常取 64k(LLaMA-2 的 32k → LLaMA-3 扩到 128k,Qwen 用 ~152k,Gemma 用 256k),给非英语足够 token 配额来压低 fertility。

## 原理

**1. BPE 训练(频次贪心,回顾)。** 见 [[051 BPE 与 Byte-level BPE|BPE]]:每步合并当前**频次最高**的相邻对,记入有序规则表,直到 $|V|$ 达标。合并次数 = 目标词表 − 基词表(字节级基为 256)。

**2. Unigram 训练(似然剪枝,反向)。** 与 BPE 相反:**先建一个超大候选子词集合,再逐步删除**。每个子词 $u$ 有概率 $p(u)$,而一段文本 $x$ 的合法切分集合为 $\mathcal S(x)$。模型的句子概率必须把所有切分的概率相加:

$$P(x)=\sum_{s\in\mathcal S(x)}P(s)=\sum_{s\in\mathcal S(x)}\prod_{u\in s}p(u),\qquad \mathcal L=-\sum_{x\in D}\log P(x)$$

EM/forward-backward 以这一个**和**计算期望出现次数,再估 $p(u)$;随后删掉令总对数似然损失最小的一批子词,重复到 $|V|$ 达标。部署时常用 Viterbi:

$$s^\star=\arg\max_{s\in\mathcal S(x)}\prod_{u\in s}p(u)$$

它只给一个确定的最高概率切分,不能替代上式训练中的边际化。详见 [[052 WordPiece、Unigram 与 SentencePiece|Unigram]]。

**3. 词表大小的代价模型。** 嵌入层 + 输出层参数 $\approx 2|V|d$(若 [[054 词嵌入层与权重绑定|权重绑定]] 则 $|V|d$)。序列长度 $L\propto 1/\text{(平均 token 覆盖字符数)}$,大致随 $|V|$ 增大而减小(边际递减)。总算力既含 $O(L^2)$ 的注意力(偏好大 $|V|$ 缩短 $L$),也含 $O(|V|d)$ 的输出投影(偏好小 $|V|$)。最优 $|V|$ 是这两项的折中,且依赖语言数:**英文 ~33k 够,多语言需数倍**(arXiv:2310.08754)。

**4. 长尾 token 的训练充分性。** 词表越大,越多 token 落在长尾、出现频次极低 → 其嵌入收到的梯度少、训练不足 → 易成 [[055 分词的坑：数字、代码、多语言与 token 攻击面|故障 token]]。所以**词表与训练语料必须同分布**:词表里的每个 token 都该在训练数据里有足够出现。

**5. 特殊 token 与冻结。** 训练时预留特殊 token(`<|endoftext|>`、`<pad>`、对话模板的角色标记,见 [[053 词表、特殊 token 与对话模板|词表与特殊 token]])。一旦冻结,**词表不可再改**——改了等于换了模型的「眼睛」,已训权重全废。所以扩词表(如做领域适配)要么从头训,要么用「加新 token + 微调新嵌入」的特殊流程。

**6. 词表大小也有「最优」(随模型规模增长)。** 不只是经验拍脑袋:Tao 等《Scaling Laws with Vocabulary》(NeurIPS 2024,arXiv:2407.13623)给出**词表也服从 scaling law**——计算最优词表参数应比非词表参数**增长更慢**(亚线性),但仍随模型变大而变大,即「大模型配大词表」。直觉:模型越大,越「养得起」更大的词表(嵌入/softmax 参数占比相对变小),且更大词表缩短序列、省注意力算力。该工作指出**多数 LLM 词表偏小**:Llama2-70B 用 32k,而计算最优应 **≥216k(约 7 倍)**。这把「凭经验选 $|V|$」升级成「按算力预算算 $V_{\text{opt}}$」。

**7. 每 token 成本与 $V_{\text{opt}}$ 的权衡(把账列出)。** 单步算力两块:① 注意力 $\propto L^2\propto f^2$(fertility 高则贵,偏好大 $|V|$ 压低 $f$);② 输出投影 + 嵌入 $\propto |V|\,d$(偏好小 $|V|$)。最优 $|V|$ 让二者边际相等。举例:$|V|$ 从 32k 翻到 64k,若 fertility 从 1.5 降到 1.3(序列短 ~13%、注意力省 ~25%),但嵌入/输出参数翻倍——对**小模型**(嵌入占比大)不划算,对**大模型**(嵌入占比小)划算。这就是为什么大模型/多语言模型倾向大词表。

![[tok-词表最优交点.png]]

**8. 字节回退(byte-fallback)与零 OOV。** SentencePiece-BPE 常开 `byte_fallback=True`:任何词表里没有的字符,**回退成其 UTF-8 字节**(256 个字节 token 兜底),于是**永不产 UNK**(对比纯 Unigram/WordPiece 可能 UNK)。代价是罕见字符 fertility 高(一个字符拆成多字节)。这让 LLaMA 等能处理任意 emoji、生僻字、二进制片段而不崩。

**9. 训练复杂度与编码复杂度。** **训练**:朴素 BPE 每步扫全语料找最高频对,$O(N\cdot\text{merges})$;实际用优先队列 + 增量更新加速。**编码**:BPE 按合并规则贪心,$O(L\log L)$ 或线性化实现(tiktoken 用正则预切 + 字节级 BPE,极快);Unigram 编码用 Viterbi,$O(L)$(限定最大子词长)。面试常问「分词器训练有梯度吗」——没有,纯统计;「encode 是确定的吗」——BPE/WordPiece 是,Unigram 推理确定(Viterbi)、训练可采样。

## 代码

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# —— 在语料上训练一个 BPE 分词器 ——
tokenizer = Tokenizer(models.BPE(unk_token=None))            # 字节级可设无 unk
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=32000,                                       # ✅ 核心超参：词表大小
    special_tokens=["<|endoftext|>", "<pad>"],             # 预留特殊 token
    min_frequency=2,                                        # 太罕见的对不合并（防长尾 token）
)
# corpus_files = ["data/*.txt"]
# tokenizer.train(corpus_files, trainer)                   # 纯统计、无梯度

# —— 词表大小如何改变序列长度（示意）——
def fertility(tok, texts):
    n_tok = sum(len(tok.encode(t).ids) for t in texts)
    n_word = sum(len(t.split()) for t in texts)
    return n_tok / n_word                                  # 越低越省

# 小词表 fertility 高（序列长），大词表 fertility 低（序列短、但参数多）
```

```python
# —— Unigram：训练边际化全部切分；部署 Viterbi 只选一条最大路径 ——
from math import inf

pieces = {"a": 0.4, "b": 0.1, "ab": 0.3, "ba": 0.2}
def unigram_sum_and_viterbi(text, probs):
    # alpha[j] = 前 j 个字符所有合法切分的概率和
    alpha = [0.0] * (len(text) + 1); alpha[0] = 1.0
    best = [-inf] * (len(text) + 1); best[0] = 0.0
    back = [None] * (len(text) + 1)
    for i in range(len(text)):
        if alpha[i] == 0.0:
            continue
        for piece, p in probs.items():
            j = i + len(piece)
            if text.startswith(piece, i):
                alpha[j] += alpha[i] * p
                score = best[i] + __import__("math").log(p)
                if score > best[j]:
                    best[j], back[j] = score, (i, piece)
    path, j = [], len(text)
    while j:
        j, piece = back[j]; path.append(piece)
    return alpha[-1], list(reversed(path))

total, path = unigram_sum_and_viterbi("aba", pieces)
assert round(total, 3) == 0.216
assert path == ["ab", "a"]  # 最大路径 0.12；不是句子总概率 0.216
```

```python
# ❌ 错：单语英文语料训分词器，却拿去训多语言模型
#    → 中文/泰文 fertility 爆表、序列暴长、算力浪费
# ✅ 对：按目标语言比例混采语料再训分词器，给小语种足够 token 配额

# ❌ 错：词表设得过大（如 500k）以为序列越短越好
#    → 嵌入/softmax 参数暴涨、长尾 token 训练不足 → 故障 token
# ✅ 对：英文 ~32k；多语言 64k~256k，在 fertility 与参数量间折中

# ❌ 错：模型训完后想随便往词表里加 token
#    → 冻结的词表一改，已训嵌入/输出层对不上；需专门的扩词表+微调流程
```

```python
# —— 词表大小的成本权衡:注意力(∝f²)vs 嵌入/输出(∝|V|d)——
def step_cost(V, d, fertility, L_words, batch):
    L = L_words * fertility                     # 实际 token 序列长
    attn  = batch * L**2 * d                    # 注意力 ∝ L²
    embed = batch * L * V * d                   # 输出投影/嵌入 ∝ |V|
    return attn, embed
for V, f in [(32000, 1.5), (64000, 1.3), (256000, 1.1)]:
    a, e = step_cost(V, d=4096, fertility=f, L_words=2048, batch=1)
    print(f"|V|={V:>6}  f={f}  attn={a:.2e}  embed={e:.2e}")
# 大词表压低 f → 注意力省;但 embed ∝|V| 涨。小模型嵌入占比大 → 大词表不划算;
# 大模型嵌入占比小 → 大词表划算(Tao 2024:大模型配大词表)

# —— 字节回退:任意字符都能编码,永不 UNK ——
# SentencePiece: --byte_fallback=true → 词表外字符拆成 UTF-8 字节(256 个字节 token 兜底)
# ✅ LLaMA 能处理 emoji/生僻字/二进制片段;代价:罕见字符 fertility 高
```

## 面试高频

- **Q:tokenizer 是怎么"训练"的?和模型训练一样吗?** A:在语料上**纯统计**学出固定切法(BPE 合并规则 / Unigram 子词概率),无梯度、无反向传播,和神经网络训练完全两回事。
- **Q:词表大小怎么权衡?** A:小词表参数省但序列长(fertility 高)、算力贵($O(L^2)$)、长依赖难;大词表序列短、多语言友好但嵌入/softmax 参数暴涨、长尾 token 易成故障 token。英文 ~32k 够,多语言 64k~256k。
- **Q:BPE 和 Unigram 训练方向有何不同?** A:BPE 自底向上,从字符反复合并最高频对;Unigram 自顶向下,从超大候选集按似然损失逐步剪枝。关键是 Unigram 的训练概率要对所有合法切分求和,不是只取一条最佳路径。
- **Q:Unigram 的 EM 与 Viterbi 分别做什么?** A:EM/forward-backward 用 $P(x)=\sum_s\prod_{u\in s}p(u)$ 边际化全部切分,据此估子词期望计数和剪枝损失;Viterbi 在部署编码时求 $\arg\max_s\prod p(u)$,输出一个确定切分。把 Viterbi 最大值写成训练概率会低估句子似然并改变学习目标。
- **Q:为什么词表太大反而有害?** A:① 嵌入/softmax 参数 $\approx |V|d$ 线性暴涨;② 更多 token 落长尾、训练不足 → 故障 token;③ 序列缩短的边际收益递减。
- **Q:分词器训练语料有什么讲究?** A:必须与模型训练数据同分布、按目标语言比例采样;否则小语种 fertility 爆、或出现训练不足的长尾 token。
- **Q:训练好的词表能改吗?** A:基本不能。词表冻结后是模型的"眼睛",改了已训嵌入/输出层全对不上;扩词表需从头训或专门的加 token+微调流程。
- **Q:词表大小有没有「最优值」?** A:有,且服从 scaling law(Tao 2024):计算最优词表随模型变大而变大(但比非词表参数增长慢)。多数模型词表偏小——Llama2-70B 用 32k,最优应 ≥216k(~7×)。「大模型配大词表」。
- **Q:字节回退是什么,解决什么?** A:词表外字符回退成 UTF-8 字节(256 字节 token 兜底),永不产 UNK;LLaMA 等用它处理任意 emoji/生僻字/二进制,代价是罕见字符 fertility 高。
- **Q:encode 是确定性的吗?训练有梯度吗?** A:训练纯统计无梯度;BPE/WordPiece 编码确定(贪心),Unigram 推理用 Viterbi 确定、训练可按后验采样(子词正则化)。
- **陷阱**:词表与训练语料必须同分布;别用单语分词器去训多语言模型;按算力预算选 $V_{\text{opt}}$ 而非一律 32k;Unigram 训练要边际化全部切分,Viterbi 只是部署解码;实现中在 log 域做求和以防下溢。

## 关键事实

- 词表大小影响算力与多语言:**《Tokenizer Choice For LLM Training: Negligible or Crucial?》(arXiv:2310.08754, NAACL Findings 2024)**——英文 ~33k 足够,多语言需约 3 倍;高 fertility 使训练算力增加约 68%。
- 词表 scaling law:**Tao et al.《Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies》(NeurIPS 2024,arXiv:2407.13623)**——计算最优词表随算力/模型增大而增大(亚线性于非词表参数);多数 LLM 词表偏小,Llama2-70B 最优应 ≥216k(约 7×)。
- BPE:**Sennrich et al.(arXiv:1508.07909, ACL 2016)**,频次贪心合并;Byte-level BPE 见 GPT-2(2019)。Unigram:**Kudo(arXiv:1804.10959, ACL 2018)**,以 unigram LM 的所有切分概率之和训练,再做似然损失剪枝;SentencePiece 库(Kudo & Richardson, arXiv:1808.06226)直接在原始文本上训,适合多语言。
- 现实词表规模:GPT-2/3 ~50k,LLaMA-1/2 32k,LLaMA-3 128k,Qwen ~152k,Gemma 256k——多语言/代码模型倾向更大词表。
- 长尾 token 训练不足是 [[055 分词的坑：数字、代码、多语言与 token 攻击面|故障 token]] 的根因;词表须与训练语料同分布。
- 关联:子词算法 [[051 BPE 与 Byte-level BPE|BPE]] / [[052 WordPiece、Unigram 与 SentencePiece|Unigram、SentencePiece]];词表与特殊 token [[053 词表、特殊 token 与对话模板|词表]];嵌入层与权重绑定 [[054 词嵌入层与权重绑定|词嵌入]];分词实战坑 [[055 分词的坑：数字、代码、多语言与 token 攻击面|分词的坑]];词表大小与参数/算力配比关联 [[079 Scaling Law 与 Chinchilla 最优|Scaling Law]]。
