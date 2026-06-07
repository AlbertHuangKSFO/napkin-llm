[[28 采样方法(逆变换、Gumbel-Softmax、重参数化)|采样方法]]研究"怎么从一个分布里抽样,且抽样过程还能求梯度":**逆变换采样**用均匀随机数生成任意分布;**重参数化技巧**把"采高斯"改写成"采固定噪声再变形",让梯度能穿过随机节点;**Gumbel-Softmax** 把这一招推广到离散的 [[24 常见分布(高斯、伯努利、类别)|类别分布]]。

## 直觉

计算机只会生成 $[0,1)$ 均匀随机数,其它分布都得"由它变出来"。

**逆变换采样**:任何分布的累积分布函数 $F$ 把 $x$ 映到 $[0,1]$ 的概率。反过来,抽一个均匀数 $u$,用 $F^{-1}(u)$ 就能得到服从该分布的样本——CDF 陡的地方(概率密)被映射到很宽的 $u$ 区间,自然抽得多。

**为什么需要"可微采样"**:在 VAE、强化学习、离散隐变量里,网络要**对"采样这一步的参数"求梯度**。但 $z\sim\mathcal N(\mu,\sigma)$ 这步是随机的、没法直接对 $\mu,\sigma$ 求导(随机节点挡住了梯度)。

**重参数化技巧**:把随机性"挪到外面"。不写 $z\sim\mathcal N(\mu,\sigma^2)$,改写成 $z=\mu+\sigma\cdot\epsilon$,其中 $\epsilon\sim\mathcal N(0,1)$ 是**与参数无关的固定噪声**。现在 $z$ 是 $\mu,\sigma$ 的**确定可微函数**,梯度能顺着 $z$ 流回 $\mu,\sigma$——随机性被隔离在 $\epsilon$ 这条不需要梯度的支路上。这就是"为何能让采样可微"的全部秘密。

**Gumbel-Softmax**:离散采样(从类别分布抽一个类)本身是 argmax,不可导。先用 Gumbel 噪声 + argmax 等价地实现类别采样(Gumbel-Max),再把硬 argmax 换成带温度的 [[27 Softmax 与温度|softmax]] 软化,就得到可微的近似离散样本。

## 例子

**逆变换采指数分布**。指数分布 CDF $F(x)=1-e^{-\lambda x}$,反函数 $F^{-1}(u)=-\frac1\lambda\ln(1-u)$。取 $\lambda=1$、$u=0.5$:$x=-\ln(0.5)\approx0.693$(正好是中位数,因为 $F(0.693)=1-e^{-0.693}=0.5$✓)。抽一堆 $u$ 套这个公式,就得到指数分布样本。

**重参数化(数字走一遍梯度)**。设 $z=\mu+\sigma\epsilon$,目标 $L=z^2$,某次采到 $\epsilon=0.5$,当前 $\mu=1,\sigma=2$,则 $z=1+2\times0.5=2$,$L=4$。梯度可直接算:$\frac{\partial L}{\partial\mu}=2z\cdot1=4$,$\frac{\partial L}{\partial\sigma}=2z\cdot\epsilon=2\times2\times0.5=2$。**梯度顺利穿过了"采样"**——因为 $\epsilon$ 被当常数,$z$ 对 $\mu,\sigma$ 完全可导。若不重参数化,$z\sim\mathcal N(\mu,\sigma)$ 这步对 $\mu,\sigma$ 没有解析梯度。

**Gumbel-Max**。类别概率 $\boldsymbol p=(0.2,0.5,0.3)$,logits $z_k=\ln p_k$。每类加独立 Gumbel 噪声 $g_k$,取 $\arg\max_k(z_k+g_k)$——可证这等价于按 $\boldsymbol p$ 采样。Gumbel-Softmax 把 argmax 换 softmax:$y_k=\text{softmax}((z_k+g_k)/T)$,$T$ 小则接近 one-hot 且可微。

**拒绝采样(逆变换不行时的通法)**。想从难采的 $p(x)$ 采样,找一个好采的"提议分布" $q(x)$ 和常数 $M$ 使 $p(x)\le Mq(x)$。每轮:从 $q$ 采 $x$、再采 $u\sim U(0,1)$,若 $u\le\frac{p(x)}{Mq(x)}$ 就接受、否则丢弃重来。接受率 $=1/M$,$M$ 越接近 1 越高效。缺点:高维下 $M$ 往往巨大,接受率指数级低,几乎没法用——这也是高维要改用 MCMC 的原因。

**重要性采样(估期望而非采样)**。想算 $E_{p}[f(x)]$ 但只能从 $q$ 采:$E_p[f]=E_q\big[f(x)\frac{p(x)}{q(x)}\big]$,用权重 $w=\frac{p}{q}$ 加权平均。它不产生 $p$ 的样本,但能无偏估计 $p$ 下的期望;$q$ 与 $p$ 差太远时方差爆炸。RLHF/离线 RL 的"重要性权重"就是它。

![[prob-重参数化.png]]

## 原理

**逆变换采样**。若 $U\sim\text{Uniform}(0,1)$,则 $X=F^{-1}(U)$ 的 CDF 恰为 $F$。证明:$P(X\le x)=P(F^{-1}(U)\le x)=P(U\le F(x))=F(x)$。要求 $F$ 可逆(或用广义逆),所以它对连续单变量最好用。

**重参数化技巧(pathwise gradient)**。要优化 $\nabla_\theta\,\mathbb E_{z\sim q_\theta}[f(z)]$。若能写 $z=g_\theta(\epsilon),\ \epsilon\sim p(\epsilon)$(噪声与 $\theta$ 无关),则期望对 $\theta$ 的梯度可搬进期望内:

$$\nabla_\theta\,\mathbb E_{\epsilon}[f(g_\theta(\epsilon))]=\mathbb E_{\epsilon}\big[\nabla_\theta f(g_\theta(\epsilon))\big]$$

对高斯:$g_\theta(\epsilon)=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal N(0,I)$。这就是 VAE 的核心:能对编码器输出 $\mu,\sigma$ 反传梯度。它比"打分函数 / REINFORCE"估计量**方差低得多**,所以连续隐变量首选它。

**对比:打分函数估计量(REINFORCE)**。当采样无法写成可微变换(纯离散、不可导奖励)时,用对数导数技巧:
$$\nabla_\theta\,\mathbb E_{z\sim q_\theta}[f(z)]=\mathbb E_{z\sim q_\theta}\big[f(z)\,\nabla_\theta\log q_\theta(z)\big]$$
它**无偏**、只需要能采样和算 $\log q$、不要求 $f$ 可导(所以 RL 里奖励是黑盒也能用),但**方差很高**,通常要配 baseline(减去 $f$ 的期望)降方差。两条路线的分工:**能重参数化就用重参数化(低方差),不能就用 REINFORCE(通用但高方差)**,Gumbel-Softmax 是介于两者之间的"可微松弛"。

**Gumbel-Softmax**。Gumbel 噪声:$g=-\ln(-\ln u),\ u\sim\text{Uniform}(0,1)$。**Gumbel-Max 定理**:$\arg\max_k(\ln p_k+g_k)\sim\text{Categorical}(\boldsymbol p)$。把不可导的 argmax 用温度 softmax 松弛:

$$y_k=\frac{\exp\big((\ln p_k+g_k)/T\big)}{\sum_j\exp\big((\ln p_j+g_j)/T\big)}$$

$T\to0$ 时 $y$ 趋 one-hot(接近真离散样本)但梯度方差大;$T$ 大时 $y$ 平滑可微但偏差大——训练中常从高温退火到低温。前向想要硬样本、反向想要梯度时,用 **straight-through**:前向取 $\arg\max$,反向用软 $y$ 的梯度。

三者关系:**逆变换**是通用生成法;**重参数化**让连续采样可微(高斯 $=\mu+\sigma\epsilon$);**Gumbel-Softmax** 是它在离散类别上的对应物(用 Gumbel 噪声 + 温度 softmax)。共同主题:**把"带参数的随机"拆成"固定噪声 + 确定可微变换"**。

![[prob-Gumbel采样.png]]

## 代码

```python
import numpy as np
rng = np.random.default_rng(0)

# 逆变换:从均匀采指数分布(λ=1),均值应≈1
u = rng.random(1_000_000)
x = -np.log(1 - u)                 # F⁻¹(u)
print("逆变换指数样本均值:", round(x.mean(), 3))   # ≈ 1.0

# ❌ 直接采样,梯度断:z~N(μ,σ) 这步对 μ,σ 无解析梯度
mu, sigma = 1.0, 2.0
# z = rng.normal(mu, sigma)   # 随机节点挡住反传,∂z/∂μ 拿不到

# ✅ 重参数化:z = μ + σ·ε,ε 固定 → 对 μ,σ 可微
eps = 0.5                          # 某次采到的固定噪声
z = mu + sigma * eps               # = 2.0
L = z**2                           # = 4.0
dL_dmu    = 2 * z * 1              # 4.0
dL_dsigma = 2 * z * eps           # 2.0
print("✅ ∂L/∂μ, ∂L/∂σ:", dL_dmu, dL_dsigma)   # 4.0 2.0

# Gumbel-Softmax:p=(0.2,0.5,0.3),低温≈one-hot 且频率对
p = np.array([0.2, 0.5, 0.3]); logits = np.log(p)
def gumbel_softmax(logits, T, rng):
    g = -np.log(-np.log(rng.random(len(logits))))   # Gumbel 噪声
    z = (logits + g) / T
    z -= z.max(); e = np.exp(z); return e / e.sum()
hard = [gumbel_softmax(logits, 0.1, rng).argmax() for _ in range(100_000)]
print("Gumbel 采样频率:", (np.bincount(hard, minlength=3)/100_000).round(3))  # ≈[0.2 0.5 0.3]
```

`❌` 直接 `rng.normal(mu,sigma)` 让随机节点挡住反传,拿不到 $\partial z/\partial\mu$;`✅` 改写成 $\mu+\sigma\epsilon$ 后 $z$ 对 $\mu,\sigma$ 可微,手算梯度 $(4,2)$ 与解析一致。逆变换的指数样本均值 $\approx1$,Gumbel-Softmax 低温采样频率收敛到 $(0.2,0.5,0.3)$,印证三法正确。

## 面试高频

- **"重参数化技巧解决什么问题?为什么可微?"** 解决"对采样步骤的参数求梯度"。把 $z\sim\mathcal N(\mu,\sigma)$ 改成 $z=\mu+\sigma\epsilon$($\epsilon$ 固定),随机性挪到无参数支路,$z$ 对 $\mu,\sigma$ 成可微函数,梯度可穿过。这是 VAE 核心。
- **"为什么不直接对采样求梯度?"** 采样是离散/随机操作,无解析梯度;替代的 REINFORCE(打分函数)无偏但方差很高,所以连续情形首选低方差的重参数化。
- **"离散变量怎么可微采样?"** Gumbel-Softmax:Gumbel-Max 用噪声+argmax 实现类别采样,再用温度 softmax 松弛 argmax;straight-through 让前向硬、反向软。
- **"逆变换采样要什么条件?"** 需要可逆 CDF $F^{-1}$;高维/复杂分布算不出反函数时改用拒绝采样、MCMC 等。
- **"Gumbel-Softmax 温度 $T$ 的权衡?"** 低温更接近真离散但梯度方差大,高温更平滑但偏差大;常退火。
- **"REINFORCE 和重参数化怎么选?"** 能写成可微变换(连续、可导)用重参数化,方差低;不能(纯离散、黑盒奖励)用 REINFORCE,无偏但方差高、需 baseline。Gumbel-Softmax 是离散情形的可微松弛折中。
- **"拒绝采样高维为什么不行?"** 接受率 $=1/M$,高维下使 $p\le Mq$ 的 $M$ 指数级大,接受率趋 0;改用 MCMC(随机游走探索高概率区)。
- **"重要性采样在做什么?"** 用 $q$ 的样本加权($w=p/q$)无偏估计 $p$ 下的期望;$p,q$ 差太远时权重方差爆炸。
- **"straight-through 估计器?"** 前向取硬 $\arg\max$(真离散输出),反向假装它是软 softmax 来传梯度;偏但实用,VQ-VAE、量化训练常用。
- **陷阱:重参数化要求采样可写成"参数的可微变换"。** 纯离散分布做不到,这正是要请出 Gumbel-Softmax 的原因。

## 关键事实

- 重参数化技巧(pathwise gradient)用于变分推断由 Kingma & Welling《Auto-Encoding Variational Bayes》(2013)提出;该思想在运筹学中早有"pathwise derivative"之名。
- Gumbel-Softmax 由 Jang, Gu & Poole《Categorical Reparameterization with Gumbel-Softmax》(arXiv 1611.01144, 2016;ICLR 2017)提出;同一分布由 Maddison et al.(2016)以"Concrete distribution"独立提出。
- 逆变换采样与 Gumbel-Max 技巧是经典结果,见 Devroye《Non-Uniform Random Variate Generation》(1986)与 Murphy《MLPP》(2012)第 23 章。
