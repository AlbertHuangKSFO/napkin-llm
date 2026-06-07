[[31 KL 散度与 JS 散度]]:**KL 散度 $D_{KL}(p\|q)=\sum p\log\frac{p}{q}$ 度量"分布 $q$ 偏离真分布 $p$ 多远"**,它非负、非对称、不是真正的距离;JS 散度是它对称化、有界的版本。KL 是 [[30 交叉熵与负对数似然|交叉熵]]扣掉真熵后剩下的那块。

## 直觉

[[29 熵与信息量|熵]] $H(p)$ 是"用真分布的最优码,编一个样本要几 bit"。[[30 交叉熵与负对数似然|交叉熵]] $H(p,q)$ 是"用错的码本 $q$ 去编,要几 bit"。两者之差,就是**因为用错码本而多花的 bit**——这就是 KL:

$$D_{KL}(p\|q)=H(p,q)-H(p)\ge 0$$

它永远 $\ge 0$,只有 $q=p$ 时为 0。**它是"浪费",不是"距离"。**

最反直觉的一点:**KL 不对称**,$D_{KL}(p\|q)\ne D_{KL}(q\|p)$。

- **前向 KL** $D_{KL}(p\|q)$:在 $p$ 大的地方,若 $q$ 小,$\log(p/q)$ 巨大 → 逼 $q$ **覆盖 $p$ 的所有峰**(mean-seeking,会摊平)。
- **反向 KL** $D_{KL}(q\|p)$:在 $q$ 大的地方,若 $p$ 小则受罚 → 逼 $q$ **躲进 $p$ 的某一个峰**(mode-seeking,会漏峰)。

这就是为什么生成模型里"用哪个方向的 KL"会让结果模糊还是锐利,差异巨大。

![[info-KL非对称.png]]

## 例子

**例 1:手算 KL(用 $\ln$)。** $p=[0.5,0.5]$,$q=[0.9,0.1]$:

$$D_{KL}(p\|q)=0.5\ln\frac{0.5}{0.9}+0.5\ln\frac{0.5}{0.1}=0.5(-0.588)+0.5(1.609)=0.511$$

**例 2:验证非对称。** 反过来 $D_{KL}(q\|p)=0.9\ln\frac{0.9}{0.5}+0.1\ln\frac{0.1}{0.5}=0.9(0.588)+0.1(-1.609)=0.529-0.161=0.368$。

$0.511\ne 0.368$ —— 同一对分布,换方向数值就变,**确实不对称**。

**例 3:JS 是对称平均。** 取中点 $m=\tfrac12(p+q)=[0.7,0.3]$:

$$\mathrm{JS}(p\|q)=\tfrac12 D_{KL}(p\|m)+\tfrac12 D_{KL}(q\|m)$$

$D_{KL}(p\|m)=0.5\ln\frac{0.5}{0.7}+0.5\ln\frac{0.5}{0.3}=0.0297$;$D_{KL}(q\|m)=0.9\ln\frac{0.9}{0.7}+0.1\ln\frac{0.1}{0.3}=0.0263$;
$\mathrm{JS}=\tfrac12(0.0297+0.0263)=0.0280$。换 $p,q$ 顺序结果不变(对称)。

**例 4:三类 KL 手算。** $p=[0.7,0.2,0.1]$,$q=[0.5,0.3,0.2]$:
$$D_{KL}(p\|q)=0.7\ln\tfrac{0.7}{0.5}+0.2\ln\tfrac{0.2}{0.3}+0.1\ln\tfrac{0.1}{0.2}=0.7(0.336)+0.2(-0.405)+0.1(-0.693)=0.235-0.081-0.069=0.085$$

**例 5:两个高斯的 KL(有闭式,VAE 必用)。** 一维 $p=\mathcal N(\mu_1,\sigma_1^2),q=\mathcal N(\mu_2,\sigma_2^2)$:
$$D_{KL}(p\|q)=\ln\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12$$
VAE 把后验拉向标准正态 $q=\mathcal N(0,1)$,代入得 $D_{KL}=\frac12(\mu^2+\sigma^2-\ln\sigma^2-1)$。验证 $\mu=0,\sigma=1$:$\frac12(0+1-0-1)=0$ ✓(两分布相同则 KL 为 0)。$\mu=1,\sigma=1$:$\frac12(1+1-0-1)=0.5$。这个闭式让 VAE 的 KL 项不用采样、直接可导。

## 原理

**KL 散度(相对熵):**

$$D_{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}=\mathbb{E}_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right]=H(p,q)-H(p)$$

性质:

1. **非负**(吉布斯不等式 / Jensen):$D_{KL}\ge 0$,等号 $\iff p=q$ 处处成立。
2. **非对称**:$D_{KL}(p\|q)\ne D_{KL}(q\|p)$,不满足三角不等式 → **不是度量(metric)**。
3. **吸收性**:若某处 $p>0$ 而 $q=0$,则 $D_{KL}=+\infty$($q$ 必须覆盖 $p$ 的支撑)。

**JS 散度(Jensen-Shannon):** 令 $m=\tfrac12(p+q)$,

$$\mathrm{JS}(p\|q)=\tfrac12 D_{KL}(p\|m)+\tfrac12 D_{KL}(q\|m)$$

它**对称**、**有界**($0\le\mathrm{JS}\le\log 2$,以 2 为底则 $\le 1$),$\sqrt{\mathrm{JS}}$ 还是真正的距离。原始 GAN 的目标在最优判别器下等价于最小化 $\mathrm{JS}(p_{\text{data}}\|p_g)$。

**为什么训练里常见的是交叉熵而非 KL?** 因为 $D_{KL}(p\|q)=H(p,q)-H(p)$,而真分布的熵 $H(p)$ 不含模型参数 $\theta$,对 $\theta$ 求梯度为 0。所以**优化交叉熵和优化 KL 完全等价**,直接算交叉熵更省事。

**前向 vs 反向 KL 的"覆盖 vs 抓峰"(深入)**。设真分布 $p$ 是双峰,用单峰高斯 $q$ 去拟合:
- **前向 $D_{KL}(p\|q)$**(以 $p$ 为权重):凡 $p>0$ 处 $q$ 都不能太小(否则 $\log(p/q)$ 爆),逼 $q$ **铺开盖住两个峰** → 落在两峰之间的低谷,生成"平均脸"/模糊样本(mean/mass-seeking)。最大似然就是前向 KL。
- **反向 $D_{KL}(q\|p)$**(以 $q$ 为权重):凡 $q>0$ 处 $p$ 不能太小(否则受罚),逼 $q$ **缩进某一个峰**、忽略另一个 → 锐利但漏模式(mode-seeking/zero-forcing)。变分推断/VAE 用反向 KL,故有"后验塌缩"、只抓主模式的倾向。

**其它分布距离(补全谱系)**:
- **总变差距离** $\text{TV}(p,q)=\tfrac12\sum|p-q|\in[0,1]$,是真正的度量,直观"两分布最大可区分概率"。
- **Wasserstein(推土机)距离**:把一堆土从 $p$ 搬成 $q$ 的最小搬运代价。即使两分布**完全不重叠**也能给出有意义的、随距离平滑变化的值——这正是 WGAN 用它取代 JS 的原因(JS 在不重叠时恒为 $\log2$、梯度为 0,GAN 训练崩)。

**实践中 KL 出现在哪(普通文字,非链接):** RLHF 的 PPO 目标里加一项 KL 惩罚,约束策略别离参考模型太远(防 reward hacking、防遗忘);变分推断 / VAE 的 ELBO 里有一项 $D_{KL}(q(z\mid x)\|p(z))$ 把后验拉向先验;知识蒸馏用 KL 让学生匹配老师的软标签;对比/自监督里也常见 KL 正则。

## 代码

```python
import numpy as np

# ❌ 错:q 出现 0 时 p/q 爆 inf;且不检查 p、q 是否同支撑
def kl_bad(p, q):
    return np.sum(p * np.log(p / q))           # q 有 0 → inf/nan,p 有 0 → 0*log0=nan

# ✅ 对:只在 p>0 处累加(p=0 项贡献 0);q=0 而 p>0 时按定义 = inf
def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = p > 0
    if np.any((q[mask] == 0)):
        return np.inf                          # q 没覆盖 p 的支撑
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def js(p, q):
    m = 0.5 * (np.asarray(p, float) + np.asarray(q, float))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

p, q = [0.5, 0.5], [0.9, 0.1]
print(kl(p, q), kl(q, p))        # 0.511  0.368  → 非对称
print(js(p, q), js(q, p))        # 0.028  0.028  → 对称

# 高斯闭式 KL:D_KL(N(μ,σ²) || N(0,1)) = ½(μ²+σ²-lnσ²-1)(VAE 的 KL 项)
def kl_gauss_to_std(mu, sigma):
    return 0.5 * (mu**2 + sigma**2 - np.log(sigma**2) - 1)
print(kl_gauss_to_std(0., 1.))   # 0.0  (相同分布)
print(kl_gauss_to_std(1., 1.))   # 0.5

# 三类手算核对
print(round(kl([0.7,0.2,0.1], [0.5,0.3,0.2]), 3))   # 0.085
```

```python
import torch, torch.nn.functional as F
# PyTorch:input 要 log 概率,target 要概率;方向是 D_KL(target || input)
logq = F.log_softmax(torch.tensor([2.0, 0.5]), dim=0)
p    = torch.tensor([0.5, 0.5])
print(F.kl_div(logq, p, reduction='sum'))      # 注意参数顺序与 log 要求,极易错
```

## 面试高频

- **"KL 是距离吗?"** 不是。非负但不对称、不满足三角不等式,所以不是 metric;它是"相对熵 / 信息增益"。
- **"KL 和交叉熵关系?"** $H(p,q)=H(p)+D_{KL}(p\|q)$;训练时 $H(p)$ 常数,优化交叉熵 = 优化 KL。
- **"前向 vs 反向 KL 区别?"** 前向 $D_{KL}(p\|q)$ 是 mean-seeking(覆盖、易摊平);反向 $D_{KL}(q\|p)$ 是 mode-seeking(抓单峰、易漏峰)。VAE 用反向,故重建偏模糊/塌缩到主模式。
- **"JS 比 KL 好在哪?"** 对称、有界、即使两分布不重叠也不会爆 $\infty$;但完全不重叠时 JS 恒为 $\log 2$、梯度为 0,这正是原始 GAN 训练不稳、催生 Wasserstein 距离的原因。
- **"两个高斯的 KL 有闭式吗?"** 有:$D_{KL}(\mathcal N(\mu_1,\sigma_1^2)\|\mathcal N(\mu_2,\sigma_2^2))=\ln\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac12$;VAE 拉向 $\mathcal N(0,1)$ 时简化为 $\frac12(\mu^2+\sigma^2-\ln\sigma^2-1)$。能写出来很加分。
- **"Wasserstein 距离比 KL/JS 好在哪?"** 即使两分布不重叠也给出平滑、有意义的距离和梯度;JS 在不重叠时恒为 $\log2$、梯度为 0,这是原始 GAN 不稳、WGAN 改用 Wasserstein 的根。
- **"为什么 RLHF 要 KL 惩罚?"** 约束新策略别离参考模型太远,防止为刷奖励而生成退化/越界文本(reward hacking),也防灾难性遗忘。
- **"哪些场景用 KL?"** RLHF(PPO 的 KL 惩罚)、VAE 的 ELBO、知识蒸馏、变分推断、最大似然(= 最小化前向 KL)。

## 关键事实

- KL 散度由 Kullback & Leibler 1951 提出;$D_{KL}\ge 0$ 由吉布斯不等式保证(Cover & Thomas,2006,第 2 章)。
- $D_{KL}(p\|q)=H(p,q)-H(p)$,非对称、不满足三角不等式,故非度量(同上)。
- JS 散度对称且有界 $[0,\log 2]$;$\sqrt{\mathrm{JS}}$ 是度量。原始 GAN 在最优判别器下最小化 JS(Goodfellow et al., 2014)。
- RLHF 用 KL 约束策略偏离参考模型(Stiennon et al. 2020 / Ouyang et al. InstructGPT 2022)。
