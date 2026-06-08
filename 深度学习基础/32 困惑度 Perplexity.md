[[32 困惑度 Perplexity]]:**困惑度 $\mathrm{PPL}=e^{\text{交叉熵}}$,是语言模型评估的根本指标**,直观含义是"模型预测下一个词时,平均在几个候选里犹豫"——越小越好,下界是 1。

## 直觉

语言模型每一步都在预测"下一个词的概率分布"。我们想知道:**它对真实文本有多"困惑"?**

把每一步的 [[30 交叉熵与负对数似然|交叉熵]](也就是对真实下一个词的 $-\log$ 概率)在整段文本上**平均**,再取指数 $e^{(\cdot)}$,得到困惑度。它有个极漂亮的解释:**有效分支数**。

- 模型对每个位置都很确定(几乎只剩 1 个候选)→ 交叉熵 ≈ 0 → PPL ≈ 1。
- 模型对每个位置都在 $V$ 个词里**均匀乱猜** → 交叉熵 = $\ln V$ → PPL = $V$。

所以"PPL = 30"近似等于"模型平均像在 30 个等可能的词里抽签"。从交叉熵($\log$ 尺度,不直观)换到困惑度(词数尺度,人能感知),就是为了好读。

![[info-困惑度有效分支.png]]

## 例子

**例 1:完美确定。** 模型对每个真词都给概率 1 → 每步 $-\ln 1=0$ → 平均交叉熵 0 → $\mathrm{PPL}=e^0=1$。下界就是 1。

**例 2:均匀乱猜。** 词表 $V=1000$,模型对每个词都给 $1/1000$ → 每步交叉熵 $-\ln(1/1000)=\ln 1000=6.908$ → $\mathrm{PPL}=e^{6.908}=1000=V$。乱猜的 PPL 等于词表大小。

**例 3:手算一小段。** 三个词,模型给真词的概率依次 $0.5,\ 0.25,\ 0.5$:

$$\text{平均交叉熵}=-\tfrac13(\ln 0.5+\ln 0.25+\ln 0.5)=-\tfrac13(-0.693-1.386-0.693)=0.924$$

$$\mathrm{PPL}=e^{0.924}=2.52$$

模型平均像"在 2.5 个词里挑"。

**例 4:几何平均视角(同一段)。** PPL = 整段概率几何平均的倒数 $=\big(0.5\times0.25\times0.5\big)^{-1/3}=(0.0625)^{-1/3}=2.52$,与上面 $e^{0.924}$ 完全一致——两种算法殊途同归。

**例 5:底数换算核对。** 用 $\log_2$ 算平均交叉熵 $=-\tfrac13(\log_2 0.5+\log_2 0.25+\log_2 0.5)=-\tfrac13(-1-2-1)=1.333$ bit,$\mathrm{PPL}=2^{1.333}=2.52$。用 $\ln$ 得 $0.924$ nat,$e^{0.924}=2.52$。**同一段文本,底数不影响 PPL**(因为 $2^{H_{\text{bit}}}=e^{H_{\text{nat}}}$)。

**例 6:比乱猜还差。** 词表 $V=1000$,但模型对真词只给 $10^{-4}$(比均匀的 $10^{-3}$ 还低):每步交叉熵 $-\ln 10^{-4}=9.21$,$\mathrm{PPL}=e^{9.21}\approx10000=10V$。**PPL 可以远超 $V$**——只有"恰好均匀乱猜"才等于 $V$,预测得比乱猜还离谱时会爆表。

## 原理

对长度 $N$ 的文本 $w_1\dots w_N$,模型给出条件概率 $p(w_t\mid w_{<t})$。**每词平均交叉熵(以 $e$ 为底)**:

$$\mathrm{CE}=-\frac{1}{N}\sum_{t=1}^{N}\ln p(w_t\mid w_{<t})$$

**困惑度**就是它的指数:

$$\boxed{\ \mathrm{PPL}=\exp\!\left(-\frac{1}{N}\sum_{t=1}^{N}\ln p(w_t\mid w_{<t})\right)=\left(\prod_{t=1}^{N}p(w_t\mid w_{<t})\right)^{-1/N}\ }$$

第二个等式说明:**PPL 是整段文本概率的几何平均的倒数**。"赋予真实文本越高的概率 → PPL 越低"。

**底数无所谓,但要一致:** 用 $\log_2$ 时 $\mathrm{PPL}=2^{\text{以 bit 计的交叉熵}}$,用 $\ln$ 时 $\mathrm{PPL}=e^{\text{以 nat 计的交叉熵}}$,**同一段文本两者相等**(因为 $2^{\log_2 x}=e^{\ln x}$)。报告时务必说清是否用了相同分词。

**性质:**

1. $\mathrm{PPL}\ge 1$,等号当且仅当模型对每个真词都给概率 1。
2. $\mathrm{PPL}\le V$ 不一定成立——若模型对真词给了极低概率,PPL 可远超 $V$。乱猜(均匀)才恰好 = $V$。
3. **强烈依赖分词(tokenizer)**:同一模型,子词切得越细、序列越长,逐 token 的 PPL 一般越低,**不同分词的 PPL 不可直接比较**。

**跨分词比较:bits-per-byte / bits-per-char**。为了让不同 tokenizer 的模型可比,改用"每字节比特数"(BPB)或"每字符比特数"(BPC):把整段的总负对数似然(以 bit 计)除以**字节数/字符数**而非 token 数(LLM 评估里 PPL 与 BPB 的完整用法见 [[LLM/109 语言模型评估：困惑度与 bits-per-byte|语言模型评估]])。因为字节/字符是与分词无关的物理单位,BPB 才能横向比较 LLaMA、GPT 等不同分词的模型。换算:$\text{BPB}=\frac{\text{总 nat}}{\ln 2\times\text{字节数}}$。

**PPL 与压缩的对偶**。每词交叉熵就是"用这个模型做算术编码,平均每词要花的 bit 数";PPL = $2^{\text{bit 交叉熵}}$。**好的语言模型 = 好的无损压缩器**——这是信息论(香农信源编码)给 PPL 的另一重解读,也是"语言建模即压缩"观点的依据。

**滑动窗口评估(长文本的正确姿势)**。模型上下文有限(如 4K),评长文本时不能每段从头开始(那样早期 token 没有上下文、PPL 虚高)。标准做法是滑动窗口:窗口右移,只对**有足够左侧上下文**的 token 计 loss(strided/sliding-window PPL),HuggingFace 文档明确推荐这种算法。

## 代码

```python
import numpy as np
import torch, torch.nn.functional as F

# ❌ 错:把每步概率直接连乘再开方,长序列下连乘下溢成 0 → PPL = inf
def ppl_bad(probs):                # probs: 每步给真词的概率
    return np.prod(probs) ** (-1 / len(probs))   # N 大时 prod → 0

# ✅ 对:在 log 域累加,最后取 exp(等价但不下溢)
def perplexity(probs):
    probs = np.asarray(probs, float)
    ce = -np.mean(np.log(probs))   # 平均交叉熵(nat)
    return np.exp(ce)

print(perplexity([0.5, 0.25, 0.5]))    # 2.52
print(perplexity([1/1000] * 1000))     # 1000.0  = V(均匀乱猜)

# 几何平均视角 + 底数无关性核对
probs = np.array([0.5, 0.25, 0.5])
print("几何平均倒数:", round(np.prod(probs)**(-1/len(probs)), 3))    # 2.52
ce_bit = -np.mean(np.log2(probs)); ce_nat = -np.mean(np.log(probs))
print("2^bit:", round(2**ce_bit, 3), " e^nat:", round(np.exp(ce_nat), 3))  # 都是 2.52

# bits-per-byte:跨分词可比指标
def bits_per_byte(total_nll_nat, num_bytes):
    return total_nll_nat / (np.log(2) * num_bytes)
print("BPB 示例:", round(bits_per_byte(100.0, 200), 3))   # 总 100 nat / 200 字节
```

```python
# 从 logits 直接算(LM 评估常见写法)
def ppl_from_logits(logits, targets):
    # logits: [N, V],targets: [N] 真实下一个 token
    ce = F.cross_entropy(logits, targets, reduction='mean')  # 已是平均 nat 交叉熵
    return torch.exp(ce)

logits = torch.randn(5, 1000)          # 随机未训练 → 接近均匀
targets = torch.randint(0, 1000, (5,))
print(ppl_from_logits(logits, targets))  # ≈ 1000 量级(乱猜)
```

## 面试高频

- **"困惑度到底是什么?"** $e^{\text{平均交叉熵}}$;直觉 = 模型每步平均犹豫的有效候选数;= 文本概率几何平均的倒数。
- **"PPL 的取值范围?"** 下界 1(完美);均匀乱猜 = 词表大小 $V$;预测得比乱猜还差时可超过 $V$。
- **"为什么用 PPL 而不直接看交叉熵?"** 数学等价,但 PPL 在"词数"尺度上人类可解释,便于横向对比。
- **"两个模型的 PPL 能直接比吗?"** 只有**相同 tokenizer + 相同测试集 + 相同上下文长度**才能比;换分词后逐 token PPL 不可比(常改用 bits-per-byte 等做跨分词比较)。
- **"跨 tokenizer 怎么比?"** 用 bits-per-byte / bits-per-char(除以字节/字符数而非 token 数),物理单位与分词无关,才可比。
- **"PPL 和压缩什么关系?"** 每词交叉熵 = 算术编码每词比特数,好的 LM 就是好的无损压缩器;"语言建模即压缩"。
- **"长文本怎么算 PPL?"** 滑动窗口:右移窗口,只对有足够左侧上下文的 token 计 loss,避免开头无上下文导致虚高。
- **"PPL 能超过词表大小吗?"** 能。只有恰好均匀乱猜才等于 $V$;给真词比均匀更低的概率时 PPL 远超 $V$。
- **"PPL 低就一定生成好吗?"** 不一定。PPL 只衡量"对测试文本的概率拟合",和指令遵从、事实性、人类偏好弱相关,所以现代 LLM 评估还要叠加下游任务与人评。

## 关键事实

- 困惑度 $\mathrm{PPL}=\exp(\text{每词交叉熵})$,是语言模型内在评估的标准指标(Jelinek et al. 1977 引入;Jurafsky & Martin《Speech and Language Processing》第 3 版,LM 章节)。
- $\mathrm{PPL}=\bigl(\prod_t p(w_t\mid w_{<t})\bigr)^{-1/N}$,即文本概率几何平均的倒数。
- PPL 强依赖分词,跨 tokenizer 不可直接比较;HuggingFace 文档明确建议固定分词与滑动窗口计算(2024)。
- 现代 LLM 报告中 PPL 已不是唯一指标,需配合下游基准与人类评估(GPT/LLaMA 等技术报告通行做法)。
