[[25 最大似然估计 MLE|最大似然估计]](MLE)是"选一组参数 $\theta$,让已观测到的数据出现的概率最大"——这是绝大多数损失函数的来源:**最小化负对数似然 = 做最大似然**,直接通到 [[24 常见分布(高斯、伯努利、类别)|分布假设]]决定的 MSE / 交叉熵。

## 直觉

你手里有一堆数据,但不知道生成它的参数(高斯的 $\mu,\sigma$、硬币的 $p$、网络的权重)。MLE 的逻辑朴素到极点:**哪组参数让"恰好看到这批数据"这件事最不离谱,就选哪组**。

把每个样本在某参数下的概率乘起来,得到**似然** $L(\theta)$——它是"参数的函数",不是数据的函数。似然越大,这组参数越"解释得动"数据。

两个工程化处理:① 连乘易下溢且难求导,取 $\log$ 变连加(对数单调,不改最大值位置);② 优化器习惯最小化,于是给对数似然取负——**最大化对数似然 = 最小化负对数似然(NLL)**。这个负号就是你天天写的 `loss` 的由来。

## 例子

**抛硬币估 $p$(离散,可手算)**。抛 10 次,7 正 3 反。似然

$$L(p)=p^7(1-p)^3$$

取对数:$\ell(p)=7\ln p+3\ln(1-p)$。求导置零:

$$\ell'(p)=\frac{7}{p}-\frac{3}{1-p}=0\ \Rightarrow\ 7(1-p)=3p\ \Rightarrow\ \hat p=\frac{7}{10}=0.7$$

MLE 就是**样本频率**——和直觉完全一致,但这是从"最大化似然"严格推出来的。

**高斯估 $\mu$ → 推出 MSE**。$n$ 个样本 $x_i\sim\mathcal N(\mu,\sigma^2)$,负对数似然

$$-\ell(\mu)=\frac{1}{2\sigma^2}\sum_i(x_i-\mu)^2+\text{常数}$$

对 $\mu$ 求最小,常数和 $\frac{1}{2\sigma^2}$ 不影响极值位置,于是**最小化 NLL ⟺ 最小化 $\sum(x_i-\mu)^2$ = 最小化 MSE**,解是样本均值 $\hat\mu=\bar x$。这就是"回归用 MSE"的根。取 $x=(2,4,6)$,$\hat\mu=4$。

**高斯估 $\sigma^2$ → MLE 有偏(经典陷阱)**。对同一高斯似然同时优化 $\sigma^2$,求导置零得
$$\hat\sigma^2_{\text{MLE}}=\frac1n\sum_i(x_i-\hat\mu)^2$$
注意它除以 $n$ 而非 $n-1$。可证 $E[\hat\sigma^2_{\text{MLE}}]=\frac{n-1}{n}\sigma^2<\sigma^2$——**MLE 系统地低估方差**(因为用了"量身定做"的样本均值 $\hat\mu$)。无偏估计要除以 $n-1$(贝塞尔校正)。这说明 **MLE 不一定无偏**,是高频追问点。取 $x=(2,4,6)$:$\hat\mu=4$,MLE 方差 $=\frac{(2{-}4)^2+0+(6{-}4)^2}{3}=\frac{8}{3}\approx2.67$,无偏版 $=\frac{8}{2}=4$。

**指数分布估 $\lambda$(再来一个手算)**。$n$ 个等待时间 $x_i\sim\text{Exp}(\lambda)$,$p(x\mid\lambda)=\lambda e^{-\lambda x}$。对数似然 $\ell=n\ln\lambda-\lambda\sum x_i$,求导置零 $\frac n\lambda-\sum x_i=0\Rightarrow\hat\lambda=\frac{n}{\sum x_i}=\frac1{\bar x}$:**MLE 是样本均值的倒数**,合直觉(平均等待越久 → 速率越低)。

**泊松估 $\lambda$**。$p(k\mid\lambda)=\frac{\lambda^k e^{-\lambda}}{k!}$,对数似然对 $\lambda$ 求导置零得 $\hat\lambda=\bar k$:**MLE 就是样本平均计数**。一连串例子都指向同一直觉:MLE 往往把参数估成最自然的样本统计量。

![[prob-似然曲线.png]]

## 原理

**似然与对数似然**。数据 $D=\{x_i\}_{i=1}^n$ 独立,参数 $\theta$:

$$L(\theta)=\prod_{i=1}^n p(x_i\mid\theta),\qquad \ell(\theta)=\log L(\theta)=\sum_{i=1}^n\log p(x_i\mid\theta)$$

**MLE**:

$$\hat\theta_{\text{MLE}}=\arg\max_\theta\ \ell(\theta)=\arg\min_\theta\ \underbrace{-\sum_i\log p(x_i\mid\theta)}_{\text{负对数似然 NLL}}$$

**分布假设决定损失**(这是全篇核心):

- $p(x\mid\theta)=\mathcal N$(固定 $\sigma$)→ NLL $=\frac{1}{2\sigma^2}\sum(x_i-\mu)^2+C$ → **MSE**。
- $p$ = 伯努利 → NLL $=-\sum[y_i\ln\hat p_i+(1-y_i)\ln(1-\hat p_i)]$ → **二元交叉熵**。
- $p$ = 类别 → NLL $=-\sum_i\sum_k y_{ik}\ln\hat p_{ik}$ → **多类交叉熵**。

所以"网络输出概率 + 交叉熵损失"不是约定俗成,而是"假设标签服从某分布,做 MLE"的必然结果。交叉熵与 NLL 在分类里是同一个东西,详细在 [[30 交叉熵与负对数似然|交叉熵]] 展开(此处前向呼应)。

**为什么是连加而非连乘**:$\log$ 单调,$\arg\max$ 不变;连加数值稳定(避免 $10^{-300}$ 下溢)、求导后是各项之和、对优化更友好。

**MLE 的统计性质**(面试可分点答):
- **一致性**:$n\to\infty$ 时 $\hat\theta_{\text{MLE}}\to\theta^*$(收敛到真值)。
- **渐近正态**:$\sqrt n(\hat\theta-\theta^*)\to\mathcal N(0,I(\theta)^{-1})$,误差棒来自费舍尔信息 $I(\theta)=-E[\partial^2\ell/\partial\theta^2]$(似然曲面在峰顶越尖 → 信息越大 → 估计越准)。
- **渐近有效**:方差达到 **Cramér–Rao 下界** $I(\theta)^{-1}$(任何无偏估计的方差下限),即大样本下最优。
- **不变性**:若 $\hat\theta$ 是 $\theta$ 的 MLE,则任意函数 $g(\theta)$ 的 MLE 就是 $g(\hat\theta)$(这一点 [[26 最大后验 MAP 与正则化的概率解释|MAP]] 没有)。
- **可能有偏**:如上面 $\hat\sigma^2$ 低估;无偏和有效是两码事。
- **会过拟合**:它只拟合训练数据、无正则,小样本下把噪声也学进去。给似然乘一个先验、做 [[26 最大后验 MAP 与正则化的概率解释|MAP]],等价于加正则项,正是为缓解这点。

![[nn-MLE费舍尔信息.png]]

**MLE = 最小化与经验分布的 KL**(把本篇和信息论接上)。最大化对数似然等价于最小化数据经验分布 $\hat p$ 与模型 $p_\theta$ 的 [[31 KL 散度与 JS 散度|KL 散度]] $D_{KL}(\hat p\|p_\theta)$,也等价于最小化二者的 [[30 交叉熵与负对数似然|交叉熵]]——这就是"分类损失 = 交叉熵 = NLL = MLE"四位一体的来源。

![[prob-MLE分布拟合.png]]

## 代码

```python
import numpy as np
from scipy.optimize import minimize_scalar

# 抛硬币:7正3反,MLE 应得 p=0.7
heads, tails = 7, 3
neg_log_lik = lambda p: -(heads*np.log(p) + tails*np.log(1-p))
res = minimize_scalar(neg_log_lik, bounds=(1e-6, 1-1e-6), method='bounded')
print("MLE p̂:", round(res.x, 3))           # 0.700 = 频率

# 高斯估 μ:最小化 NLL ⟺ 最小化 MSE,解=样本均值
x = np.array([2., 4., 6.])
nll_mu = lambda mu: np.sum((x - mu)**2)      # 去掉常数,正比于 NLL
mus = np.linspace(0, 8, 801)
print("argmin NLL 的 μ:", round(mus[np.argmin([nll_mu(m) for m in mus])], 2))  # 4.0
print("样本均值:", x.mean())                                                    # 4.0

# ❌ 直接连乘似然:n 大就下溢成 0,梯度全废
p = 1e-3
print("❌ 连乘 1000 个 0.001:", p**1000)     # 0.0,信息全丢

# ✅ 取对数变连加,数值稳定
logL = 1000 * np.log(p)
print("✅ 对数似然:", round(logL, 1))         # -6907.8,可用
```

`❌` 直接连乘大量小概率会下溢为 $0$,似然和梯度全部失效;`✅` 取对数把连乘变连加,数值稳定可优化。硬币 MLE $=0.7$ 即频率,高斯最小化 NLL 的解 $=4$ 即样本均值,印证手算与"NLL = MSE"。

```python
# 验证高斯方差 MLE 有偏:除以 n 系统低估,除以 n-1 才无偏
import numpy as np
rng = np.random.default_rng(0)
true_var = 4.0
biased, unbiased = [], []
for _ in range(20000):
    x = rng.normal(0, np.sqrt(true_var), size=5)   # 小样本 n=5
    biased.append(x.var(ddof=0))      # 除以 n   → MLE
    unbiased.append(x.var(ddof=1))    # 除以 n-1 → 无偏
print("MLE 方差均值(偏低):", round(np.mean(biased), 3))     # ≈ (n-1)/n * 4 = 3.2
print("无偏方差均值:", round(np.mean(unbiased), 3))          # ≈ 4.0
```

MLE 方差的样本平均收敛到 $\frac{n-1}{n}\sigma^2=3.2$($n=5$),无偏版收敛到真值 $4.0$——印证"MLE 不一定无偏"。

## 面试高频

- **"为什么用对数似然?"** ① 连乘变连加,避免下溢;② 求导成各项之和;③ 单调不改最大值。三点都要说。
- **"MLE 和损失函数什么关系?"** 最小化负对数似然 = 最大似然;选什么分布就得什么损失:高斯→MSE,伯努利→BCE,类别→交叉熵。这是把损失"推出来"的关键链路。
- **"MSE 背后的假设是什么?"** 误差服从等方差高斯。数据有重尾/异常值时该换 MAE(对应拉普拉斯假设)。
- **"MLE 会过拟合吗?"** 会,它只拟合训练数据、无正则。加先验 → [[26 最大后验 MAP 与正则化的概率解释|MAP]] = 加正则项。
- **"MLE 是无偏的吗?"** 不一定。如高斯方差的 MLE 除以 $N$ 是有偏的(低估),无偏估计要除以 $N-1$。常考陷阱。
- **"似然和概率区别?"** 同一函数 $p(x\mid\theta)$:固定 $\theta$ 看 $x$ 是概率(对 $x$ 积分为 1);固定 $x$ 看 $\theta$ 是似然(对 $\theta$ 不必积分为 1)。
- **"MLE 和最小化 KL/交叉熵的关系?"** 最大化对数似然 = 最小化 $D_{KL}(\hat p\|p_\theta)$ = 最小化交叉熵($\hat p$ 是数据经验分布)。这把统计和信息论缝在一起。
- **"什么是费舍尔信息 / Cramér–Rao 下界?"** 费舍尔信息 $I(\theta)=-E[\ell'']$ 衡量似然峰有多尖、数据含多少关于 $\theta$ 的信息;CRLB 说任何无偏估计方差 $\ge I(\theta)^{-1}$,MLE 大样本下达到它(渐近有效)。
- **"MLE 有不变性吗?MAP 呢?"** MLE 有:$g(\theta)$ 的 MLE = $g(\hat\theta_{\text{MLE}})$;MAP 没有(非线性重参数后众数会变)。
- **"为什么 NLL 用连加不用连乘?数值上?"** 连乘上千个 $<1$ 的概率会下溢到机器零($\sim10^{-300}$),梯度全废;取对数变连加既防下溢又让梯度成各项之和。

## 关键事实

- 极大似然法由 R. A. Fisher 在 1912–1922 年间系统提出,是频率派统计推断的核心方法。
- MLE 的一致性、渐近正态性与有效性(达 Cramér–Rao 下界)见 Casella & Berger《Statistical Inference》(2002)第 7 章。
- "高斯噪声下 MLE 等价于最小二乘 / MSE"是标准结论,见 Bishop《PRML》(2006)第 1.2.5 与 3.1 节;深度学习中损失即 NLL 的视角见 Goodfellow et al.《Deep Learning》(2016)第 5.5 节。
