[[101 采样解码：温度、top-k、top-p、min-p、重复惩罚|采样解码：温度、top-k、top-p、min-p、重复惩罚]] 是**随机性**地从概率分布抽 token 的一族方法:[[27 Softmax 与温度|温度]] $T$ 先调分布尖/平,再用 **top-k**(留概率前 k 个)、**top-p / nucleus**(留累积概率到 p 的最小集合)、**min-p**(门槛随最高概率自适应缩放)做截断去掉长尾垃圾,最后归一化 [[28 采样方法(逆变换、Gumbel-Softmax、重参数化)|采样]]一个;**重复惩罚**额外压低已出现 token 的 logit,防止复读。

## 直觉

[[100 解码策略：贪心与 Beam|贪心/Beam]] 一味取高概率,输出保守又重复。采样反过来**按概率掷骰子**,引入多样性——但裸采样会偶尔抽到概率极低的"垃圾 token",一旦抽中就把后文带偏。于是有了"先调形、再截尾、后采样"的流水线:

- **温度 $T$**:除在 logits 上再 softmax。$T<1$ 分布变**尖**(更确定,接近贪心),$T>1$ 变**平**(更随机)。它只改形状,不删 token。
- **top-k**:只保留概率**前 k 个**,其余清零。简单,但 k 是死的——分布很尖时 k 太大放进噪声,分布很平时 k 太小砍掉好词。
- **top-p / nucleus**:按概率从高到低累加,**够 p(如 0.9)就停**,只留这个"核(nucleus)"。保留个数**随分布自适应**:尖留少、平留多。比 top-k 更合理,是事实标准。
- **min-p**:门槛 = `p × 最高 token 概率`,**随峰高浮动**。模型很自信(峰高)→ 门槛高、留得少更保守;模型犹豫(峰矮)→ 门槛低、留得多更发散。在**高温下**比 top-p 更稳(高温把分布压平后 top-p 容易放进垃圾)。
- **重复惩罚**:对已经出现过的 token,把它的 logit 除以(或减去)一个惩罚值,降低它再次被选中的概率,治"复读机"。

一句话:**温度调形 → top-k/top-p/min-p 截尾 → 重新归一 → 抽样;再叠重复惩罚防复读。**

## 例子

设 4 个 token 的概率 `[0.5, 0.25, 0.2, 0.05]`(已降序)。

- **top-k=2**:留前 2 个 → `[0.5, 0.25]` 归一化为 `[0.67, 0.33]`,后两个永不被抽。
- **top-p=0.9**:累加 `0.5 → 0.75 → 0.95 ≥ 0.9`,停在第 3 个 → 留 `[0.5, 0.25, 0.2]`,砍掉 `0.05`。换个尖分布 `[0.92, 0.05, …]`,top-p=0.9 只留第 1 个(自适应)。
- **min-p=0.1**:门槛 `0.1 × 0.5 = 0.05`,留所有 `≥0.05` 的 → `[0.5, 0.25, 0.2, 0.05]`(这里 0.05 刚好够)。若最高概率是 0.9,门槛抬到 `0.09`,就只留高于 0.09 的几个 → **峰越高留越少**。

**温度的效果**。logits `[2, 1, 0]`:
- $T=1$:softmax ≈ `[0.67, 0.24, 0.09]`。
- $T=0.5$(变尖):logits/T = `[4,2,0]` → ≈ `[0.87, 0.12, 0.02]`,更像贪心。
- $T=2$(变平):logits/T = `[1,0.5,0]` → ≈ `[0.51, 0.31, 0.19]`,更随机。

**温度 × top-p 的交互(min-p 的动机)**。续上面 4-token 例 `[0.5,0.25,0.2,0.05]`,先升温 $T=2$ 把它压平到约 `[0.34,0.24,0.22,0.20]`,再 top-p=0.9:累加 `0.34→0.58→0.80→1.0`,要到第 4 个才够 0.9 → **连那个本来概率仅 0.05 的尾巴 token 也被纳入核**。这就是高温下 top-p 的毛病:分布被压平后,「累积到 p」会把低质 token 也算进去。min-p 用「门槛 = p×峰值」躲开这点——峰值在高温下也降了,门槛跟着降,但它**相对**峰值筛选,不会因为整体压平就盲目放进一堆尾巴。所以**高温 + min-p** 比 **高温 + top-p** 更能兼顾创造性和连贯性。

![[infer-采样截断对比.png]]

## 原理

**温度**(详见 [[27 Softmax 与温度|Softmax 与温度]])。$p_i = \dfrac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$。$T\to0$ 退化为贪心(argmax),$T\to\infty$ 趋近均匀分布。

**top-k**。取概率最大的 $k$ 个下标集合 $V_k$,在其上重新归一后采样:$p'_i = \dfrac{p_i}{\sum_{j\in V_k}p_j}\ \mathbb 1[i\in V_k]$。

**top-p(nucleus)**(Holtzman et al. 2019)。按概率降序,取最小集合 $V_p$ 使 $\sum_{i\in V_p}p_i\ge p$,在其上归一采样。核大小随分布形状自适应——这是它优于 top-k 的关键。

**min-p**(2024)。设门槛 $\tau = p\cdot \max_i p_i$,保留 $\{i: p_i\ge \tau\}$ 再归一。因为门槛**正比于峰值概率**,模型越自信门槛越高、候选越少(更连贯),越不确定门槛越低、候选越多(更有创意)。论文表明在高温下能兼顾创造性与连贯性。

**重复惩罚**(CTRL 风格)。对已生成集合 $G$ 中的 token,改其 logit:

$$
z_i \leftarrow
\begin{cases}
z_i / \rho, & z_i>0 \\
z_i \cdot \rho, & z_i\le 0
\end{cases}\quad (i\in G,\ \rho>1)
$$

另有 **presence penalty / frequency penalty**(OpenAI 风格,减法式):$z_i \leftarrow z_i - \alpha\cdot\mathbb 1[i\in G] - \beta\cdot c_i$($c_i$ 为出现次数)。**presence**(出现过就固定减 $\alpha$)鼓励**换新话题**;**frequency**(按出现次数线性减 $\beta c_i$)鼓励**少复读高频词**——两者常一起用。**顺序很关键**:重复惩罚要在 logits 上、**温度缩放之前或之后**都有人做,但主流是「先在 raw logits 上加重复惩罚 → 再除温度 → softmax → 截断 → 归一 → 采样」;若惩罚放在温度之后,惩罚力度会被温度缩放扭曲。$T=0$ 等价贪心([[100 解码策略：贪心与 Beam|贪心]])。

**进阶采样(扩展知识)**:① **typical sampling**(2022)——不按概率高低,而按「token 的信息量是否接近分布的期望信息量(熵)」筛选,留「典型」token,理论更贴近人类语言的信息率;② **eta / epsilon sampling**——用绝对或自适应概率阈值截尾;③ **tail-free sampling**——按概率的二阶差分找分布「尾巴拐点」截断。实务里 **top-p 仍是默认主力,min-p 在高温创作场景流行**,这些进阶法多见于开源社区的精调。

## 代码

```python
import numpy as np

def softmax(z): e = np.exp(z - z.max()); return e / e.sum()

def sample_decode(logits, T=1.0, top_k=0, top_p=0.0, min_p=0.0,
                  penalty=1.0, generated=()):
    z = logits.astype(np.float64).copy()
    # 重复惩罚:压低已出现 token 的 logit(正除负乘)
    for i in set(generated):
        z[i] = z[i] / penalty if z[i] > 0 else z[i] * penalty
    z = z / T                                   # 温度调形(T<1 更尖,T>1 更平)
    p = softmax(z)
    order = np.argsort(p)[::-1]                  # 概率降序索引
    keep = np.zeros_like(p, dtype=bool)
    if top_k:
        keep[order[:top_k]] = True              # top-k:固定留前 k
    elif top_p:
        c = np.cumsum(p[order])
        cut = np.searchsorted(c, top_p) + 1     # top-p:累积到 p 的最小集合
        keep[order[:cut]] = True
    elif min_p:
        keep = p >= (min_p * p.max())           # min-p:门槛随峰值缩放
    else:
        keep[:] = True
    p = np.where(keep, p, 0.0)
    p /= p.sum()                                # 截断后重新归一(必须!)
    return int(np.random.choice(len(p), p=p))

# ❌ 错误:裸采样不截断 —— 长尾垃圾 token 偶尔被抽中,带偏后文
def sample_bad(logits, T=1.0):
    return int(np.random.choice(len(logits), p=softmax(logits / T)))

logits = np.array([4.0, 2.0, 1.5, 0.2, -1.0, -3.0])
print("top-p:", sample_decode(logits, T=0.9, top_p=0.9))
print("min-p:", sample_decode(logits, T=1.2, min_p=0.1))
```

## 面试高频

- **温度、top-k、top-p 各管什么?** 温度调分布尖/平(不删词);top-k 固定留前 k 个;top-p 留累积概率到 p 的最小集合(个数自适应)。常先调温度再截断。
- **top-p 为什么通常比 top-k 好?** top-k 的 k 是死的:尖分布会放进噪声、平分布会砍掉好词;top-p 按累积概率自适应保留,尖留少、平留多,更匹配分布形状。
- **min-p 解决了 top-p 的什么问题?** 高温把分布压平后,top-p 容易把低质 token 也纳入核;min-p 门槛 = p×最高概率,随峰值浮动,模型自信时收紧、犹豫时放宽,高温下更连贯。
- **温度 T=0 / T→∞ 是什么?** $T=0$ 退化为贪心(argmax);$T\to\infty$ 趋近均匀随机。$T<1$ 更确定,$T>1$ 更发散。见 [[27 Softmax 与温度|温度]]。
- **重复惩罚 / presence / frequency penalty 区别?** 重复惩罚对已现 token 的 logit 正除负乘;presence 是出现过就固定减一个常数;frequency 按出现次数线性减。都为压复读、增多样。
- **截断后为什么必须重新归一化?** 清零部分 token 后概率和 <1,不归一就不是合法分布,采样比例错误。
- **采样 vs Beam 怎么选?** 创作/对话要多样性用采样(温度+top-p/min-p);翻译/摘要等求"标准答案"用 [[100 解码策略：贪心与 Beam|Beam/贪心]]。
- **高温 + top-p 为什么容易出垃圾?min-p 怎么救?** 高温把分布压平,top-p「累积到 p」会把本来概率极低的尾巴 token 也算进核;min-p 用「门槛 = p×峰值」相对峰值筛选,峰降门槛也降,不会盲目放进尾巴,故高温下更连贯。
- **presence 和 frequency penalty 区别?** presence 出现过就固定减一个常数(鼓励换话题);frequency 按出现次数线性减(鼓励少复读高频词)。重复惩罚则是正除负乘式。三者都为压复读、增多样。
- **重复惩罚应在温度前还是后加?** 主流在 raw logits 上先加重复惩罚、再除温度,否则惩罚力度会被温度缩放扭曲。整体流水线:重复惩罚 → 温度 → softmax → 截断 → 归一 → 采样。
- **typical / eta / tail-free 这些是什么?** 进阶截断采样:typical 按「信息量是否接近期望熵」筛典型 token;eta/epsilon 用概率阈值截尾;tail-free 按概率二阶差分找尾巴拐点。都为更好地切掉长尾,top-p/min-p 仍是主力。
- **采样的随机性怎么复现?** 固定随机种子即可复现;温度 0(贪心)则无随机、天然可复现。生产里调试常先用贪心定位问题、再开采样看多样性。

## 关键事实

- top-p / nucleus 采样出自 Holtzman, Buys, Du, Forbes, Choi《The Curious Case of Neural Text Degeneration》(2019,arXiv:1904.09751,ICLR 2020),指出最大化似然导致退化文本,提出按累积概率自适应截断。
- min-p 出自 Nguyen et al.《Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs》(2024,arXiv:2407.01082,ICLR 2025);门槛 = p × 最高 token 概率,随模型置信度自适应。
- 温度 $T=0$ 等价贪心,$T\to\infty$ 趋近均匀;实务常配 top-p≈0.9–0.95、温度 0.7–1.0。
- 重复惩罚(正除负乘)源自 CTRL(Keskar et al. 2019,arXiv:1909.05858);OpenAI 的 presence/frequency penalty 为减法式变体。
- 典型流水线顺序:**温度 → 截断(top-k/top-p/min-p)→ 重复惩罚 → 归一化 → 采样**;采样底层方法见 [[28 采样方法(逆变换、Gumbel-Softmax、重参数化)|采样方法]]。
