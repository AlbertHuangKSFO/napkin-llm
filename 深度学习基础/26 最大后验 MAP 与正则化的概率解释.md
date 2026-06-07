[[26 最大后验 MAP 与正则化的概率解释|最大后验估计]](MAP)是在 [[25 最大似然估计 MLE|MLE]] 的基础上乘一个**先验**再取最大——展开后正好是"损失 + 正则项":**MAP = MLE + 先验 = 数据项 + 正则项**,其中 **L2 正则 ⟺ 高斯先验,L1 正则 ⟺ 拉普拉斯先验**。

## 直觉

MLE 只问"哪组参数最能解释数据",数据少时容易被噪声带偏、把权重撑得很大(过拟合)。MAP 多问一句:"在看数据之前,我对参数本来有什么偏好?"

这个偏好就是**先验**。最常见的偏好是"权重应该小、靠近 0"——因为小权重的模型更平滑、更不容易过拟合。把这个"靠近 0"的偏好写成一个以 0 为中心的高斯先验,乘到似然上再最大化,代数化简后就**自动冒出一个 $\lambda\|w\|^2$ 的惩罚项**。

所以正则化不是"凭空加的罚款",而是"你对参数的先验信念"。L2(权重衰减)说"权重大概是小的、围着 0 呈钟形";L1(Lasso)说"很多权重干脆就是 0"(尖峰先验),于是产生稀疏解。

## 例子

**从高斯先验推出 L2(完整代数)**。线性回归,似然是高斯(噪声方差 $\sigma^2$),给权重加先验 $w\sim\mathcal N(0,\tau^2)$。MAP 最大化 $\log[\text{似然}\times\text{先验}]$,等价于最小化它的负值:

$$-\log p(D\mid w)-\log p(w)=\underbrace{\frac{1}{2\sigma^2}\sum_i(y_i-w^Tx_i)^2}_{\text{数据项(MSE)}}+\underbrace{\frac{1}{2\tau^2}\|w\|^2}_{\text{先验项}}+C$$

同乘 $2\sigma^2$ 不改最优解,令 $\lambda=\sigma^2/\tau^2$:

$$\min_w\ \sum_i(y_i-w^Tx_i)^2+\lambda\|w\|^2$$

**这就是岭回归 / L2 正则**。先验越紧($\tau$ 越小)⟹ $\lambda$ 越大 ⟹ 越强迫权重小。

**数字感受**。设某权重的数据项最优在 $w=10$,但先验拉向 0,$\lambda=1$。最小化 $(w-10)^2+1\cdot w^2$:求导 $2(w-10)+2w=0\Rightarrow w=5$。先验把 $10$ 拽到了 $5$——正则化"收缩"权重的效果一目了然。$\lambda$ 越大拽得越狠($\lambda=4$ 时 $w=2$,$\lambda\to\infty$ 时 $w\to0$)。

**从拉普拉斯先验推出 L1(对照高斯)**。换成拉普拉斯先验 $p(w)\propto e^{-|w|/b}$,$-\log p(w)=\frac1b|w|+C$,于是 MAP 目标变成
$$\min_w\ \sum_i(y_i-w^\top x_i)^2+\lambda\|w\|_1$$
**这就是 Lasso / L1 正则**。同一个标量例换 L1:最小化 $(w-10)^2+\lambda|w|$,对 $w>0$ 求导 $2(w-10)+\lambda=0\Rightarrow w=10-\lambda/2$——L1 是**恒定地往 0 平移 $\lambda/2$**(软阈值),只要数据项足够弱($10<\lambda/2$)就被直接压到 0;而 L2 是**按比例收缩**($w=10/(1+\lambda)$),永远缩不到精确 0。这正是 L1 出稀疏、L2 不出的代数根。

**Beta 先验 + 伯努利 = 拉普拉斯平滑(共轭先验手算)**。抛硬币估 $p$,给 $p$ 一个 Beta$(\alpha,\beta)$ 先验。后验仍是 Beta(共轭),MAP 解为
$$\hat p_{\text{MAP}}=\frac{k+\alpha-1}{n+\alpha+\beta-2}$$
取 $\alpha=\beta=2$:$\hat p=\frac{k+1}{n+2}$。这就是 **加一平滑 / 拉普拉斯平滑**:0 正 0 反时 MLE 给 $0/0$ 或 $0$,而 MAP 给 $\frac12$,避免"零概率"灾难。先验在这里相当于"凭空先塞进 1 正 1 反的虚拟观测"。这把 [[22 概率、条件概率与贝叶斯|贝叶斯]] 的共轭先验和工程里常见的平滑技巧连起来了。

![[prob-MAP正则.png]]

## 原理

**贝叶斯出发点**(见 [[22 概率、条件概率与贝叶斯|贝叶斯]]):

$$p(\theta\mid D)=\frac{p(D\mid\theta)\,p(\theta)}{p(D)}\propto\underbrace{p(D\mid\theta)}_{\text{似然}}\underbrace{p(\theta)}_{\text{先验}}$$

**MAP** 取后验最大(分母 $p(D)$ 与 $\theta$ 无关,可丢):

$$\hat\theta_{\text{MAP}}=\arg\max_\theta\big[\log p(D\mid\theta)+\log p(\theta)\big]=\arg\min_\theta\big[\underbrace{-\log p(D\mid\theta)}_{\text{损失}}\ \underbrace{-\log p(\theta)}_{\text{正则}}\big]$$

对比 MLE:$\hat\theta_{\text{MLE}}=\arg\min[-\log p(D\mid\theta)]$。**MAP 只比 MLE 多了 $-\log p(\theta)$ 这一项,它就是正则项**。

**先验 ⟷ 正则项对照**:

| 先验 $p(w)$ | $-\log p(w)$ | 正则 | 效果 |
|---|---|---|---|
| 高斯 $\mathcal N(0,\tau^2)$ | $\frac{1}{2\tau^2}\|w\|_2^2+C$ | **L2**(权重衰减) | 权重收缩、平滑 |
| 拉普拉斯 $\propto e^{-\|w\|/b}$ | $\frac1b\|w\|_1+C$ | **L1**(Lasso) | 稀疏、特征选择 |
| 均匀(无信息) | 常数 | 无 | 退化为 MLE |

可见**MLE 是 MAP 在均匀先验下的特例**:先验为常数时正则项消失。强度 $\lambda$ 来自先验与噪声方差之比 $\sigma^2/\tau^2$——这给了"$\lambda$ 调大调小"一个概率含义:你对"权重该多小"有多自信。

**为什么 L1 出稀疏、L2 不出**:拉普拉斯先验在 0 处有尖峰(不可导的尖角),最优解被"吸"到坐标轴上 ⟹ 很多分量恰为 0;高斯先验在 0 处光滑,只把权重压小但不压到精确的 0。这与 [[42 正则化(L2、Dropout、早停、标签平滑)]] 里几何图景一致(此处前向呼应,正式展开在那篇)。

**MAP 与偏差-方差权衡**。先验/正则把估计往 0 拉,引入了**偏差**(系统性偏离数据最优),但显著降低**方差**(对噪声不敏感)。小样本、高维、强噪声时,牺牲一点偏差换大幅降方差,总误差更小——这是正则化"为何有用"的统计解释。$\lambda$ 就是在偏差和方差之间滑动的旋钮:$\lambda\to0$ 退回高方差的 MLE,$\lambda\to\infty$ 退化成高偏差的"全 0"。

**MAP 的几个边界情形**:
- 先验为均匀(无信息)→ MAP = MLE。
- 数据无穷多($n\to\infty$)→ 似然压倒先验,MAP 与 MLE 都收敛到真值(先验被"洗掉")。
- 数据极少 → 先验主导,MAP 接近先验众数(此时正则最该上)。

![[prob-先验形状L1L2.png]]

## 代码

```python
import numpy as np
rng = np.random.default_rng(0)

# 造一个病态线性回归:特征多、样本少,MLE 会过拟合
n, d = 8, 12
X = rng.normal(size=(n, d)); w_true = rng.normal(size=d)
y = X @ w_true + 0.1 * rng.normal(size=n)

# ❌ MLE / 普通最小二乘:欠定,权重爆炸
w_mle = np.linalg.pinv(X) @ y
print("❌ MLE  ||w||:", round(np.linalg.norm(w_mle), 2))   # 很大

# ✅ MAP / 岭回归 = 高斯先验:(XᵀX + λI) w = Xᵀy
lam = 1.0
w_map = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
print("✅ MAP  ||w||:", round(np.linalg.norm(w_map), 2))   # 明显更小

# 标量例:数据项最优 10,λ=1,先验拉向 0 → 解到 5
w = np.linspace(0, 12, 1201)
obj = (w - 10)**2 + 1.0 * w**2
print("收缩后的最优 w:", round(w[obj.argmin()], 2))         # 5.0
```

`❌` MLE 在特征多于样本时权重范数爆炸(过拟合);`✅` 加高斯先验(岭回归 $+\lambda I$)把权重范数显著压小。标量例里先验把数据最优 $10$ 收缩到 $5$,印证"MAP = 损失 + 正则"的拉拽效果。

```python
# L1(软阈值)vs L2(比例收缩):同一数据项最优 w0=10
import numpy as np
w0 = 10.0
for lam in (0, 1, 4, 25):
    w_l2 = w0 / (1 + lam)                       # L2:argmin (w-w0)^2 + lam*w^2
    w_l1 = max(w0 - lam/2, 0.0)                 # L1:软阈值,可压到 0
    print(f"λ={lam:2d}  L2→{w_l2:5.2f}  L1→{w_l1:5.2f}")
# λ=25 时 L1 把 w 压到 0(稀疏),L2 只压到 0.38(不为 0)

# Beta(2,2) 先验下伯努利 MAP = 拉普拉斯平滑,避免零概率
def map_p(k, n, a=2, b=2):
    return (k + a - 1) / (n + a + b - 2)        # = (k+1)/(n+2)
print("0正0反 MLE=0/0?  MAP:", map_p(0, 0))     # 0.5(先验兜底)
print("0正3反 MLE=0    MAP:", round(map_p(0, 3), 3))   # 0.2,不会给 0
```

L1 在 $\lambda$ 够大时把权重精确压到 0(稀疏),L2 永远只按比例缩小;Beta$(2,2)$ 先验让"0 正 0 反"也能给出 $0.5$ 而非未定义,印证 MAP = MLE + 先验的兜底作用。

## 面试高频

- **"L2 正则等价于什么先验?"** 权重的零均值高斯先验;MAP 推导自动给出 $\lambda\|w\|^2$。L1 对应拉普拉斯先验,出稀疏解。必背。
- **"MLE 和 MAP 差在哪?"** MAP = MLE + $\log$ 先验项;均匀先验时 MAP 退化为 MLE。一句话讲清。
- **"为什么 L1 稀疏 L2 不稀疏?"** L1 先验在 0 处有尖角,最优点被吸到坐标轴(分量恰为 0);L2 光滑,只收缩不置零。
- **"$\lambda$ 的概率含义?"** $\lambda=\sigma^2/\tau^2$:噪声方差与先验方差之比;先验越紧、对小权重越自信,$\lambda$ 越大。
- **"MAP 是完整贝叶斯吗?"** 不是。MAP 只取后验的众数(一个点),丢了不确定性;完整贝叶斯对整个后验积分(贝叶斯预测)。常见进阶追问。
- **"L1 软阈值 vs L2 比例收缩,代数上?"** L1:$w=\text{sign}(w_0)\max(|w_0|-\lambda/2,0)$(恒定平移、可到 0);L2:$w=w_0/(1+\lambda)$(按比例缩、不到 0)。
- **"拉普拉斯平滑 / 加一平滑是怎么来的?"** Beta/Dirichlet 共轭先验下的 MAP;相当于先塞进虚拟计数,避免零概率。朴素贝叶斯、n-gram 都用它。
- **"正则化为什么能改善泛化(偏差-方差)?"** 引入偏差换取大幅降方差,小样本/高维下总误差更小;$\lambda$ 是偏差-方差旋钮。
- **"weight decay 和 L2 一样吗?"** 在 SGD 上等价,但在 Adam 上不等价——要用 AdamW 把衰减解耦(见 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|AdamW]])。常见进阶陷阱。
- **陷阱:MAP 不具参数化不变性。** 对 $\theta$ 做非线性重参数,众数会变(MLE 不变);这是 MAP 的理论瑕疵。

## 关键事实

- "L2 正则 = 高斯先验下的 MAP,L1 = 拉普拉斯先验"是标准结论,见 Bishop《PRML》(2006)第 3.3–3.4 节与 Murphy《MLPP》(2012)第 7.5 节。
- 岭回归由 Hoerl & Kennard(1970)提出;Lasso(L1)由 Tibshirani(1996)提出并以稀疏性著称。
- MLE 是 MAP 在均匀(无信息)先验下的特例,MAP 与完整贝叶斯的区别(众数 vs 后验积分)见 Goodfellow et al.《Deep Learning》(2016)第 5.6 节。
