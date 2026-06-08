[[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)]]:**激活函数是神经元里那个非线性 $\sigma$**,没有它[[34 MLP 与万能逼近|多层网络]]会塌缩成线性;选哪一个直接决定梯度好不好传——从早期会饱和的 Sigmoid/Tanh,到主流的 ReLU,再到现代 LLM 用的 GELU / SwiGLU。

## 直觉

激活函数只干一件事:**把神经元的加权和 $z$ 弯一下**,引入非线性。但"怎么弯"影响巨大,核心看两点:

1. **会不会饱和?** Sigmoid / Tanh 在 $|z|$ 大时曲线压平,导数趋 0 → 反传时梯度被反复乘以接近 0 的数 → 深层学不动(梯度消失)。ReLU 在正半轴导数恒为 1,不饱和,所以能训很深的网。
2. **负半轴怎么处理?** ReLU 把负数直接砍成 0,简单但有"死亡神经元"风险(一旦落进负区永远输出 0、梯度 0、再也激活不了)。GELU 给负半轴留一点平滑的负值,信息不全丢。

演化主线:**Sigmoid/Tanh(会饱和、慢)→ ReLU(不饱和、快,但会死)→ GELU(平滑版 ReLU,Transformer 标配)→ SwiGLU(门控,现代 LLM 的 FFN)**。

![[nn-激活函数族.png]]

## 例子

**例 1:各函数在 $z=2$ 与 $z=-2$ 的值(手算,$e^2\approx7.389$)。**

| 函数 | 公式 | $\sigma(2)$ | $\sigma(-2)$ |
|---|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $0.881$ | $0.119$ |
| Tanh | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | $0.964$ | $-0.964$ |
| ReLU | $\max(0,z)$ | $2$ | $0$ |
| GELU | $z\,\Phi(z)$ | $\approx1.95$ | $\approx-0.046$ |

ReLU 把 $-2$ 砍成 0;GELU 留了个小负值 $-0.046$,信息没全丢。

**例 1b:激活全家在 $z=-2,0,2$ 的导数(反传要乘的就是它)。**

| 函数 | 导数公式 | $\sigma'(-2)$ | $\sigma'(0)$ | $\sigma'(2)$ |
|---|---|---|---|---|
| Sigmoid | $\sigma(1-\sigma)$ | $0.105$ | $0.25$ | $0.105$ |
| Tanh | $1-\tanh^2$ | $0.071$ | $1.0$ | $0.071$ |
| ReLU | $\mathbb 1[z>0]$ | $0$ | $0$/未定义 | $1$ |
| Leaky ReLU(0.01) | $1$ 或 $0.01$ | $0.01$ | — | $1$ |

读这张表:sigmoid/tanh 两端导数都很小(饱和 → 梯度消失);ReLU 正区导数恒 1(不衰减),但负区恒 0(可能死);Leaky 给负区留 0.01 的活路。**反向传播每穿过一层就乘一次这个导数**,所以它的大小直接决定梯度能不能传到底层(见 [[38 反向传播在网络中的实现|反传]])。

**例 2:Sigmoid 的梯度消失(手算导数)。** $\sigma'(z)=\sigma(z)(1-\sigma(z))$。在 $z=0$:$\sigma'=0.5\times0.5=0.25$(最大);在 $z=4$:$\sigma(4)=0.982$,$\sigma'=0.982\times0.018=0.0176$。每层至多乘 0.25,**10 层连乘后梯度 $\le 0.25^{10}\approx 10^{-6}$**,几乎归零——这就是为什么深网弃用 Sigmoid 做隐藏层。

**例 3:死亡 ReLU。** 若某神经元的权重让所有训练样本都落进 $z<0$,则输出恒 0、$\mathrm{ReLU}'=0$、梯度恒 0 → 它永远不更新,等于"死了"。Leaky ReLU $\max(0.01z,z)$ 给负区留个小斜率来救它。

![[nn-饱和与梯度消失.png]]

## 原理

**Sigmoid:** $\sigma(z)=\dfrac{1}{1+e^{-z}}\in(0,1)$,导数 $\sigma'=\sigma(1-\sigma)\le 0.25$。输出非零中心(恒正)会让后层梯度同号、之字形收敛;两端饱和。今多只用于二分类输出层(配 BCE),不做隐藏层。

**Tanh:** $\tanh(z)=\dfrac{e^z-e^{-z}}{e^z+e^{-z}}\in(-1,1)$,$\tanh'=1-\tanh^2\le 1$。**零中心**(比 Sigmoid 好),但仍两端饱和。

**ReLU:** $\mathrm{ReLU}(z)=\max(0,z)$,导数为 $1\,(z>0)$ 或 $0\,(z<0)$。优点:正区不饱和、计算极廉、稀疏激活。缺点:非零中心 + **死亡神经元**。

**ReLU 变体(救死亡神经元)**:
- **Leaky ReLU** $\max(\alpha z,z)$,$\alpha=0.01$:负区给小斜率,梯度不为 0。
- **PReLU**:把 $\alpha$ 当**可学参数**,让网络自己决定负区斜率。
- **ELU** $z\,(z>0)$、$\alpha(e^z-1)\,(z\le0)$:负区平滑饱和到 $-\alpha$,输出更接近零中心。
- **Softplus** $\ln(1+e^z)$:ReLU 的处处可导光滑版,导数恰是 sigmoid。
- **Mish / SiLU(Swish)**:平滑、非单调,经验上有时优于 ReLU。

**GELU(Gaussian Error Linear Unit):**

$$\mathrm{GELU}(z)=z\,\Phi(z),\qquad \Phi(z)=\Pr[Z\le z],\ Z\sim\mathcal N(0,1)$$

用标准正态 CDF $\Phi$ 当"软门":按输入自身的大小**概率性地保留**它(而 ReLU 是按符号硬开关)。处处平滑可导、负区有小负值,常用 tanh 近似

$$\mathrm{GELU}(z)\approx 0.5\,z\Bigl(1+\tanh\bigl[\sqrt{\tfrac{2}{\pi}}(z+0.044715z^3)\bigr]\Bigr)$$

是 BERT、GPT 系列的默认激活。

**SwiGLU:** 不是单输入函数,而是一个**门控单元**。先定义 Swish(SiLU)$\mathrm{Swish}(z)=z\cdot\sigma(\beta z)$;SwiGLU 把 FFN 的输入投影成两路,一路当"门":

$$\mathrm{SwiGLU}(x)=\mathrm{Swish}(xW)\odot(xV)$$

$\odot$ 是逐元素乘。一路经 Swish 当 0~1 的"阀门",控制另一路线性投影通过多少。Shazeer(2020)发现 GEGLU/SwiGLU 在 Transformer 的 [[LLM/008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]] 上困惑度更低,**LLaMA、PaLM 等现代 LLM 采用**。代价:门控多一组投影矩阵(从 2 个变 3 个权重矩阵),通常把 FFN 隐藏维按 $\tfrac23$ 缩回去补偿参数量(LLaMA 即用 $\tfrac{8}{3}d$ 而非 $4d$)。

![[nn-SwiGLU门控.png]]

**GLU 家族**:门控线性单元的通式是 $\text{act}(xW)\odot(xV)$,换不同 act 得不同成员——Sigmoid→GLU、GELU→GEGLU、Swish→SwiGLU、ReLU→ReGLU。共同点:一路当"软阀门"控制另一路通过多少,比单一逐元素激活多了"按内容选择信息通路"的能力。

**输出层激活按任务定(单列一条,易错)**:
- 回归 → **恒等**(线性输出,配 MSE)。
- 二分类 → **sigmoid**(1 个 logit,配 BCE)。
- 多分类(互斥)→ **softmax**(配交叉熵)。
- 多标签(可同时多类)→ **逐类 sigmoid**(每类独立 BCE),**不是 softmax**——这是常见混淆点。

## 代码

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def tanh(z):    return np.tanh(z)
def relu(z):    return np.maximum(0, z)
def gelu(z):                                   # tanh 近似
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
def swish(z, beta=1.0): return z * sigmoid(beta * z)
def leaky(z, a=0.01): return np.where(z > 0, z, a * z)
def elu(z, a=1.0):     return np.where(z > 0, z, a * (np.exp(z) - 1))
def softplus(z):       return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)  # 稳定版

z = np.array([-2., 0., 2.])
print(relu(z))     # [0. 0. 2.]
print(gelu(z))     # [-0.046  0.  1.954]  负区留小负值
print(leaky(z))    # [-0.02  0.    2.  ]  负区有小斜率,不会死
print(elu(z).round(3))      # [-0.865 0. 2.]  负区平滑饱和
print(softplus(z).round(3)) # [0.127 0.693 2.127]  ReLU 的光滑版

# 验证梯度消失:sigmoid 导数连乘 vs ReLU
g_sig = 0.25**np.arange(1, 11)        # 每层最多 0.25
print("sigmoid 10 层梯度上界:", g_sig[-1])   # ~9.5e-7,几乎归零
print("ReLU 正区 10 层:", 1.0**10)            # 1.0,不衰减

# ❌ 错:深层隐藏层用 sigmoid,梯度连乘 ≤0.25^L 指数消失,深网几乎学不动
# ✅ 对:隐藏层用 ReLU/GELU 这类正区不饱和的激活,梯度能传到底层

# SwiGLU:门控,不是逐元素一元函数
def swiglu(x, W, V):
    return swish(x @ W) * (x @ V)              # ⊙ 逐元素乘,一路当门
x = np.random.randn(1, 4)
W = np.random.randn(4, 8); V = np.random.randn(4, 8)
print(swiglu(x, W, V).shape)                   # (1, 8)
```

```python
import torch.nn as nn
# 框架内置
nn.Sigmoid(); nn.Tanh(); nn.ReLU(); nn.GELU(); nn.SiLU()   # SiLU = Swish(β=1)
```

## 面试高频

- **"为什么深网络隐藏层不用 Sigmoid?"** $\sigma'\le 0.25$,多层连乘 → 梯度消失;且非零中心拖慢收敛。ReLU 正区导数恒 1,不饱和,能训很深。
- **"什么是死亡 ReLU?怎么救?"** 神经元落进 $z<0$ 后输出/梯度恒 0、永不更新;用 Leaky ReLU / PReLU / ELU 给负区留小斜率,或更小学习率 / 更好初始化。
- **"GELU 比 ReLU 好在哪?"** 处处平滑可导、负区不硬砍(保留少量负信息)、按输入大小概率性门控,Transformer 上效果更稳。
- **"现代 LLM 的 FFN 用什么激活,为什么?"** SwiGLU(门控,LLaMA/PaLM);Shazeer 2020 实验显示困惑度更低;门控让网络能自适应地选择信息通路。
- **"Tanh 比 Sigmoid 好在哪?"** 零中心(输出 $\in(-1,1)$),梯度更对称、收敛更快;但仍会饱和。
- **"ReLU 变体都有哪些,各解决什么?"** Leaky/PReLU(负区小斜率,救死亡);ELU(负区平滑饱和、近零中心);Softplus(光滑可导);Swish/Mish(平滑非单调,有时更好)。
- **"GLU 家族是什么?SwiGLU 为什么占 3 个矩阵?"** $\text{act}(xW)\odot(xV)$,一路当门;SwiGLU 用 Swish 当门,比普通 FFN 多一个投影矩阵,故隐藏维缩到 $\tfrac23$ 补参数。
- **"多标签分类输出层用 softmax 吗?"** 不用。多标签要逐类独立 sigmoid + BCE(各类可同时为正);softmax 强制各类概率互斥求和为 1,只适合单标签多分类。
- **"GELU 的精确式和近似式?"** 精确 $z\Phi(z)$($\Phi$ 是标准正态 CDF);常用 tanh 近似 $0.5z(1+\tanh[\sqrt{2/\pi}(z+0.044715z^3)])$,两者数值几乎一致。
- **陷阱:** 输出层激活按任务定——回归用恒等、二分类用 sigmoid、多分类用 [[27 Softmax 与温度|softmax]];激活选择影响梯度消失/爆炸,见 [[44 梯度消失、爆炸与梯度裁剪]]。

## 关键事实

- Sigmoid 导数 $\sigma'=\sigma(1-\sigma)\le 0.25$,是其在深层引发梯度消失的根因(Goodfellow《Deep Learning》2016,第 6.3 节)。
- ReLU 由 Nair & Hinton 2010 推广,缓解梯度消失、训练更快,成为长期默认隐藏层激活。
- GELU $=x\Phi(x)$,Hendrycks & Gimpel 2016(《Gaussian Error Linear Units》,arXiv:1606.08415);BERT/GPT 默认采用。
- SwiGLU 由 Shazeer 2020(《GLU Variants Improve Transformer》,arXiv:2002.05202)提出,GEGLU/SwiGLU 困惑度最优;LLaMA、PaLM 等现代 LLM 的 FFN 采用。
