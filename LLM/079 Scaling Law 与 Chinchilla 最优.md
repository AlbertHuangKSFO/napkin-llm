[[079 Scaling Law 与 Chinchilla 最优]]:大模型的 Loss 随参数量 N、数据量 D、算力 C 呈**幂律(power-law)**平滑下降(Kaplan 2020,arXiv 2001.08361);给定算力预算,存在一组**算力最优**的 N 与 D 配比,Chinchilla(Hoffmann 2022,arXiv 2203.15556)给出经验法则:**每个参数约配 20 个训练 token(D≈20N)**,纠正了早期「重模型、轻数据」的偏差。

## 直觉:把「炼丹」变成「可预测的工程」

训一个大模型动辄烧几百万美元,你**没法试错**。Scaling Law 的价值是:用一堆**小模型**画出曲线,就能**外推**出大模型的 Loss——把炼丹变成查表。

核心发现只有一句话:**Loss 是幂律的**。把参数量放大 10 倍,Loss 不是线性下降,而是按一个固定的「斜率」在**双对数图(log-log)上走直线**(见图)。这条直线跨越了七个数量级都不弯——这才是它能用来做预算规划的原因。

但「越大越好」会带来一个新问题:固定算力下,该把钱花在**更大的模型**还是**更多的数据**?这就是 [[077 训练 FLOPs 与 6ND 法则|6ND]] 法则要回答的——总算力 C≈6ND,N 和 D 此消彼长,必有一个最优配比。Kaplan(2020)当时算出来「大头给模型」,于是 [[036 GPT 系列：自回归与规模化|GPT-3]] 175B 只喂了约 300B token;Chinchilla(2022)用更细的实验推翻了它:**模型偏大、数据偏少,严重欠训**。

## 例子:同样 1 块算力,小模型反而赢

Chinchilla 团队训了 400+ 个模型(70M~16B 参数,5B~500B token),核验后给出:

| 模型 | 参数 N | 数据 D | D/N | 结果 |
|---|---|---|---|---|
| Gopher | 280B | 300B | ~1 | 基准 |
| GPT-3 | 175B | 300B | ~1.7 | 欠训 |
| **Chinchilla** | **70B** | **1.4T** | **20** | **同算力下全面超越 Gopher** |

Chinchilla 参数只有 Gopher 的 1/4,但喂了约 4.7 倍数据,**同样的算力预算**下在一大批下游任务上**反超**。结论一句话:**N 翻倍,D 就该翻倍**(同比放大);经验比例 **D≈20N**。

这直接改变了行业:后来的 [[038 LLaMA 架构解剖|LLaMA]] 把 7B 模型喂到 1T+ token(D/N≈140),远超 20——因为推理省钱比训练最优更重要(见下文「面试高频」)。

![[param-scaling-power-law.png]]

## 原理:幂律公式与算力最优的拉格朗日

**1）单变量幂律(Kaplan 2020)**。当只有某一项是瓶颈时:

$$
L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\quad
L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D}
$$

取对数:$\log L = \alpha_N(\log N_c - \log N)$,这是一条**斜率为 $-\alpha_N$ 的直线**——这就是双对数图上变直的数学原因。Kaplan 测得 $\alpha_N\approx0.076$、$\alpha_D\approx0.095$(很小,说明边际收益递减很慢,但确实递减)。

**2)联合 Loss(Chinchilla 形式)**。Hoffmann 把 N、D 一起建模,并加一个不可约项 $E$(数据本身的熵下限):

$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}
$$

拟合得 $\alpha\approx0.34,\ \beta\approx0.28,\ E\approx1.69$(nats)。

**3)算力约束下求最优**。固定算力 $C\approx 6ND$(见 [[077 训练 FLOPs 与 6ND 法则|6ND]]),用拉格朗日最小化 $L$:

$$
\min_{N,D}\ L(N,D)\quad\text{s.t.}\quad 6ND=C
$$

代入 $D=C/(6N)$,对 $N$ 求导置零。由于 $\alpha\approx\beta$,解出**最优 $N_{opt}\propto C^{a}$、$D_{opt}\propto C^{b}$ 且 $a\approx b\approx0.5$**——也就是 N 和 D **同比例随算力增长**。把拟合常数代入,落到工程经验上就是:

$$
\boxed{D_{opt}\approx 20\,N_{opt}}\quad(\text{每参数约 20 token})
$$

直观看「IsoFLOP 曲线」:固定算力时,Loss 关于 N 是一条 **U 形**——参数太大(数据被挤少)欠训,参数太小(浪费算力)也不优,**谷底**才是最优 N。把不同算力的谷底连起来,就是那条「最优配比轨迹」。

![[param-isoflop谷底.png]]

![[param-chinchilla-optimal.png]]

## 拉格朗日完整推导:为什么 N、D 同比放大

把"算力最优"算到底(面试问"D≈20N 怎么推的"能写)。最小化 $L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}$,约束 $C=6ND$。

**拉格朗日**:$\mathcal{L}=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}-\lambda(6ND-C)$。对 $N$、$D$ 求偏导置零:
$$-\alpha A N^{-\alpha-1}=6\lambda D,\qquad -\beta B D^{-\beta-1}=6\lambda N.$$
两式相除消 $\lambda$:
$$\frac{\alpha A N^{-\alpha-1}}{\beta B D^{-\beta-1}}=\frac{D}{N}\ \Rightarrow\ \alpha A N^{-\alpha}=\beta B D^{-\beta}.$$
即**最优时两个可约项成固定比例** $\frac{A}{N^\alpha}:\frac{B}{D^\beta}=\beta:\alpha$。代入 $D=C/(6N)$ 解出标度指数:
$$N_{\text{opt}}\propto C^{\frac{\beta}{\alpha+\beta}},\qquad D_{\text{opt}}\propto C^{\frac{\alpha}{\alpha+\beta}}.$$
Chinchilla 拟合 $\alpha\approx0.34$、$\beta\approx0.28$,两个指数 $\frac{\beta}{\alpha+\beta}\approx0.45$、$\frac{\alpha}{\alpha+\beta}\approx0.55$——**都接近 0.5**,所以 $N$ 和 $D$ 几乎**同比例随 $C$ 放大**(Kaplan 早期算出 $N$ 指数 ≈0.73、$D$ ≈0.27,严重偏向模型,这就是两篇结论分歧的数学根源)。把拟合常数代入,落到 $D_{\text{opt}}\approx20\,N_{\text{opt}}$。

**记忆**:$\alpha\approx\beta$(N、D 两项指数相近)→ 两个标度指数都 ≈0.5 → N、D 同比放大 → 经验 D≈20N。

## Chinchilla 的三种估计法(原论文三管齐下)

Hoffmann 用**三种独立方法**得到同一结论(面试加分点,显示你读过原文):
1. **固定模型、扫数据**(IsoFLOP 之一):训不同大小模型到不同 token 数,看每个 FLOP 预算下哪个模型最优。
2. **IsoFLOP 曲线**:固定算力预算 C,扫 N(随之 D=C/6N),画 Loss-vs-N 的 U 形,取谷底;连不同 C 的谷底得最优轨迹。
3. **拟合参数化 Loss** $L(N,D)=E+A/N^\alpha+B/D^\beta$,再做约束优化(上面的推导)。
三法都给出 **N、D 同比放大、D≈20N**,互相印证。

## 数据墙、重复数据与新一代 scaling

- **数据墙(data wall)**:高质量文本有限(Villalobos 2022 估计公开高质量数据约 2024-2026 年耗尽)。D 撞顶后,Chinchilla 最优("加数据")就失效——只能靠**重复数据、合成数据、多模态数据**续命。
- **重复数据的代价**(Muennighoff 2023,"Scaling Data-Constrained LLMs"):同一数据训约 **4 个 epoch 内几乎无损**(等效新数据),之后边际收益快速衰减,16 epoch 后基本没用。这给"数据不够就多刷几遍"划了上界。
- **推理感知 scaling(inference-aware)**:Chinchilla 只算训练算力最优。Sardana & Frankle(2023,arXiv:2401.00448)把**推理成本**加进目标,结论是:若模型要大量服务,**最优点应更小、训更多 token**(过训)——给 LLaMA 的 D/N≈140 提供了理论依据。
- **涌现能力(emergent abilities)争议**:某些能力(多步推理、in-context learning)看似在某规模"突然出现",Wei 2022 称"涌现"。但 Schaeffer 2023(arXiv:2304.15004)反驳:这是**评测指标不连续**(如精确匹配)造成的错觉,换连续指标后曲线平滑——这与 scaling law 的"平滑幂律"一致,而非真有相变。

## 代码:用小模型外推大模型 Loss(拟合幂律)

```python
import numpy as np
from scipy.optimize import curve_fit

# 已知:几个小模型的 (参数量 N, 实测 loss)
N = np.array([1e7, 3e7, 1e8, 3e8, 1e9])      # 参数量
L = np.array([3.10, 2.85, 2.60, 2.40, 2.24])  # 交叉熵 loss(nats)

# ❌ 错:在原始坐标上线性回归 —— 幂律不是直线,外推会错得离谱
# coef = np.polyfit(N, L, 1)        # 线性假设,大 N 处预测为负 loss(荒谬)

# ✅ 对:拟合幂律 L = E + A / N**alpha(Chinchilla 形式的单变量版)
def power_law(N, E, A, alpha):
    return E + A / np.power(N, alpha)

(E, A, alpha), _ = curve_fit(power_law, N, L, p0=[1.7, 100, 0.3], maxfev=10000)
print(f"不可约下限 E={E:.3f}, alpha={alpha:.3f}")

# 外推到 70B —— 训练前就能估出 loss,用来定预算
N_target = 7e10
print(f"预测 70B loss ≈ {power_law(N_target, E, A, alpha):.3f}")
```

要点:幂律必须在 **log-log** 上拟合(或显式拟合 `E + A/N^α`),**绝不能**在原始坐标做线性回归——那会在大 N 处给出负 Loss 这种荒谬外推。

## 面试高频

- **Kaplan 和 Chinchilla 矛盾吗?** 不矛盾,是「修正」。Kaplan 早期实验把学习率调度、warmup 等没对齐,且把算力大头分给了模型;Chinchilla 用更干净的 IsoFLOP 实验,得出 N、D 应**同比放大**。两者都同意「幂律」,只是**最优配比**不同。可补一句:Besiroglu 2024(arXiv 2406.12907)和 Pearce 2024(arXiv 2404.10102)分别讨论了二者的调和与复现。
- **为什么 LLaMA 用 D/N≈140 远超 20?** Chinchilla 最优是**训练算力最优**,没算**推理成本**。模型一旦部署,推理跑亿万次——用更小的模型(多喂数据「过训」)能永久省推理钱。所以工业界故意偏离 Chinchilla 点,选「小模型 + 超量数据」。
- **D≈20N 这个 20 是定值吗?** 不是。它依赖数据质量、架构、tokenizer;20 是 Chinchilla 在其设定下的拟合值,换数据/架构会变。记「同比放大」这个**定性结论**比记「20」更安全。
- **幂律会一直成立吗?** 不会。存在**不可约项 $E$**(数据熵下限),Loss 不可能降到 0;且高质量数据有限,「数据墙」会让 D 这一项先撞顶。
- **D≈20N 怎么推的?** 拉格朗日最小化 $E+A/N^\alpha+B/D^\beta$ s.t. $6ND=C$,解得 $N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$、$D_{\text{opt}}\propto C^{\alpha/(\alpha+\beta)}$;$\alpha\approx\beta$ 使两指数都 ≈0.5(N、D 同比),代入常数落到 D≈20N。Kaplan 因 N 指数算成 ≈0.73 才偏向大模型。
- **Chinchilla 怎么得出结论的?** 三种独立方法:① 固定模型扫数据;② IsoFLOP 谷底连成最优轨迹;③ 拟合 $L(N,D)$ 做约束优化。三法都指向 N、D 同比放大、D≈20N。
- **数据不够能重复刷吗?** 约 4 个 epoch 内几乎无损(等效新数据,Muennighoff 2023),之后边际收益快速衰减,16 epoch 后基本没用。
- **涌现能力是真的相变吗?** 有争议。Schaeffer 2023 认为是评测指标不连续(精确匹配)的错觉,换平滑指标后曲线连续,与 scaling 的平滑幂律一致,而非真有相变。

## 关键事实

- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020,arXiv **2001.08361**:Loss 随 N、D、C 呈幂律;架构细节(宽深比)影响很小;$\alpha_N\approx0.076$。
- Hoffmann et al.(DeepMind), *Training Compute-Optimal Large Language Models*, 2022,arXiv **2203.15556**:训 400+ 模型,得「N、D 同比放大」,经验 **D≈20N**;Chinchilla 70B/1.4T 同算力击败 Gopher 280B、GPT-3 175B。
- 联合拟合常数:$\alpha\approx0.34,\ \beta\approx0.28,\ E\approx1.69$ nats(原文表 3)。
- 算力最优标度指数:$N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}\approx C^{0.45}$、$D_{\text{opt}}\propto C^{0.55}$($\alpha\approx0.34,\beta\approx0.28$),近似同比;Kaplan 早期得 $N\propto C^{0.73}$ 偏向大模型,是两篇分歧根源。
- 数据墙:高质量公开文本估计 2024-2026 耗尽(Villalobos et al., 2022, arXiv:2211.04325);数据受限下重复约 4 epoch 内近无损、16 epoch 后失效(Muennighoff et al., 2023, arXiv:2305.16264)。
- 推理感知 scaling:把推理成本纳入目标后最优模型更小、训更多 token(Sardana & Frankle, 2023, arXiv:2401.00448),为工业界"过训小模型"(LLaMA D/N≈140)提供理论支撑。
- 涌现能力是否相变有争议:Wei et al., 2022(arXiv:2206.07682)称涌现;Schaeffer et al., 2023(arXiv:2304.15004)归因于评测指标不连续,与平滑幂律一致。
- 关联:[[077 训练 FLOPs 与 6ND 法则|6ND 法则]](C≈6ND 是约束式)、[[036 GPT 系列：自回归与规模化|GPT 系列]](GPT-3 是欠训典型)、[[32 困惑度 Perplexity|困惑度]](Loss=交叉熵,exp 后即困惑度)。
