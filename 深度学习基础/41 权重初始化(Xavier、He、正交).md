[[41 权重初始化(Xavier、He、正交)|权重初始化]]决定训练开始那一刻每个权重取什么随机值。看似小事,却决定网络能不能训起来:初始化不当会让信号在前向/反向中**逐层放大或缩小**,直接导致 [[44 梯度消失、爆炸与梯度裁剪|梯度消失或爆炸]]。核心思想是**让每层的激活方差和梯度方差保持稳定**——Xavier、He、正交都是这个目标的不同解法。

## 直觉

为什么不能全初始化成 0?那样同一层每个神经元收到完全相同的输入和梯度,**永远保持对称、学不出不同特征**(对称性破缺失败)。所以要随机。

但随机也有讲究。把一个信号送过几十层:每层做一次 $z=Wa$。如果 $W$ 的尺度偏大,信号方差逐层翻倍,几十层后**爆炸**(数值溢出);偏小则逐层减半,几十层后**消失**(信号变 0)。反向传播的梯度同理。

**好的初始化要让"方差守恒"**:信号过一层后,方差既不放大也不缩小($\approx1$)。要做到这点,$W$ 的方差必须和**这一层有多少输入/输出**($n_{\text{in}}, n_{\text{out}}$,即 fan-in / fan-out)挂钩——神经元接收的输入越多,每个权重就要越小,加起来才不超标。

- **Xavier(Glorot)**:为 tanh/sigmoid 这类**关于 0 对称**的激活设计,方差用 $n_{\text{in}}, n_{\text{out}}$ 的平均。
- **He(Kaiming)**:为 **ReLU** 设计。ReLU 把一半输入清零、砍掉一半方差,所以方差要**翻倍**来补偿。
- **正交初始化**:让 $W$ 是正交矩阵(保长度、保范数),信号过线性层范数严格不变,深网络/RNN 尤其稳。

**为什么"对称"这么致命(零基础再讲一遍)**。把一层想成 $k$ 个并排的神经元,如果它们的权重一模一样(全 0 是最极端的相同),那么对任意输入 $a$,它们算出的 $z$ 完全相同,激活也相同;反向传播时,损失对每个神经元权重的梯度也完全相同 —— 于是更新后它们**仍然相同**。这 $k$ 个神经元从头到尾是同一个"克隆体",等于整层只有 1 个有效神经元,白白浪费容量。打破这个克隆魔咒的唯一办法就是**让初始权重彼此不同**,即随机化。注意:偏置 $b$ 可以全初始化为 0,因为只要权重不同,神经元就已经不对称了,$b$ 相同不会重新引入对称。

**"尺度"到底要控多准?** 直觉量级:深度 $L$ 层,若每层方差放大率是 $r$,信号方差按 $r^L$ 走。$L=50$ 时,$r=1.1$ 给 $1.1^{50}\approx117$ 倍(还能撑),$r=1.5$ 给 $1.5^{50}\approx6\times10^8$(直接爆),$r=0.9$ 给 $0.9^{50}\approx0.005$(几乎归零)。可见**深度把微小的尺度偏差指数放大**,这正是初始化必须精确到"方差守恒 $r\approx1$"而不能凭感觉拍一个 $0.01$ 的原因。

![[nn-初始化方差.png]]

## 例子

一个全连接层,fan-in $n_{\text{in}}=100$,激活 $a$ 的方差为 1、均值 0,权重 $W$ 独立同分布、均值 0、方差 $\sigma_W^2$。输出 $z_j=\sum_{i=1}^{100} W_{ji}a_i$。

**输出方差**(独立项方差相加):
$$\text{Var}(z)=n_{\text{in}}\cdot\sigma_W^2\cdot\text{Var}(a)=100\,\sigma_W^2$$

**想让 $\text{Var}(z)=\text{Var}(a)=1$**,需要 $\sigma_W^2=\frac{1}{n_{\text{in}}}=\frac{1}{100}=0.01$,即 $\sigma_W=0.1$。这正是 **Xavier(只看 fan-in 版)**。

**若激活是 ReLU**:ReLU 把负半轴清零,输出方差只剩一半 $\text{Var}(z_{\text{after ReLU}})\approx\frac12 n_{\text{in}}\sigma_W^2$。要补回这一半,$\sigma_W^2=\frac{2}{n_{\text{in}}}=\frac{2}{100}=0.02$,$\sigma_W\approx0.141$。这就是 **He**——**比 Xavier 大 $\sqrt2$ 倍**。

**反例**:若用 $\sigma_W=1$(不缩放),$\text{Var}(z)=100$,过 10 层方差变 $100^{10}$——**彻底爆炸**;若 $\sigma_W=0.01$,$\text{Var}(z)=0.01$,10 层后 $\approx10^{-20}$——**彻底消失**。

**手算例 2:fan-in / fan-out 都给出来时怎么折中。** 一个 $512\to128$ 的全连接层,$n_{\text{in}}=512$,$n_{\text{out}}=128$。
- 只看前向(fan-in 版):$\sigma_W^2=\frac1{512}\approx0.00195$,$\sigma_W\approx0.0442$。
- 只看反向(fan-out 版):$\sigma_W^2=\frac1{128}\approx0.00781$,$\sigma_W\approx0.0884$。
- 两者矛盾,Xavier 取调和折中:$\sigma_W^2=\frac{2}{512+128}=\frac{2}{640}=0.003125$,$\sigma_W\approx0.0559$ —— 正好落在 0.0442 与 0.0884 之间。
- 若用 He(只看 fan-in 翻倍):$\sigma_W^2=\frac2{512}\approx0.0039$,$\sigma_W\approx0.0625$,比 Xavier 的 0.0559 大约 $\sqrt2/\sqrt{(640/512)\cdot...}$,直观上更大一点(He 总比同 fan-in 的 Xavier 偏大)。

**手算例 3:均匀分布版的界怎么来。** 想用 $U[-a,a]$ 实现 Xavier。$U[-a,a]$ 的方差是 $\frac{(2a)^2}{12}=\frac{a^2}{3}$。令它等于目标 $\frac{2}{n_{\text{in}}+n_{\text{out}}}$:$\frac{a^2}{3}=\frac{2}{n_{\text{in}}+n_{\text{out}}}\Rightarrow a=\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}$。这就是那个"神秘的 6"的出处——不是 2 也不是 3,正是 $2\times3$。He 均匀版同理:$\frac{a^2}{3}=\frac2{n_{\text{in}}}\Rightarrow a=\sqrt{\frac6{n_{\text{in}}}}$。

![[nn-初始化信号传播.png]]

## 原理

**目标**:逐层保持 $\text{Var}(a^{(\ell)})\approx\text{Var}(a^{(\ell-1)})$(前向)和 $\text{Var}(\delta^{(\ell)})\approx\text{Var}(\delta^{(\ell+1)})$(反向)。

**Xavier / Glorot(2010)**。前向要 $\sigma_W^2=\frac{1}{n_{\text{in}}}$,反向(梯度方差守恒)要 $\sigma_W^2=\frac{1}{n_{\text{out}}}$。两者一般矛盾,取调和折中:
$$\boxed{\ \sigma_W^2=\frac{2}{n_{\text{in}}+n_{\text{out}}}\ }$$
- 正态版:$W\sim\mathcal N\!\big(0,\frac{2}{n_{\text{in}}+n_{\text{out}}}\big)$。
- 均匀版:$W\sim U\!\big[-a,a\big],\ a=\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}$(因为 $U[-a,a]$ 方差为 $a^2/3$)。
适用 tanh / sigmoid(以 0 为中心、近原点 $\approx$ 线性)。

**He / Kaiming(2015)**。ReLU 让方差减半,前向守恒需把方差翻倍:
$$\boxed{\ \sigma_W^2=\frac{2}{n_{\text{in}}}\ }$$
- 正态版:$W\sim\mathcal N\!\big(0,\frac{2}{n_{\text{in}}}\big)$;均匀版界 $a=\sqrt{\frac{6}{n_{\text{in}}}}$。
那个 **因子 2 就是抵消 ReLU 砍掉一半方差**。这让 He 能训练上百层的 ReLU 网络(Glorot 在很深的 ReLU 网上会偏小、信号衰减)。Leaky-ReLU 的因子是 $\frac{2}{1+\alpha^2}$。

**正交初始化(Saxe et al. 2014)**。令 $W$ 为(半)正交矩阵($W^TW=I$),则线性变换 $z=Wa$ **保范数**,$\|z\|=\|a\|$,信号过任意多层范数严格不变(动态等距 dynamical isometry)。实现:对随机高斯矩阵做 QR 分解取 $Q$。对**很深的网络和 RNN**(梯度沿时间反复连乘)特别稳,常配增益系数 $\text{gain}$(ReLU 用 $\sqrt2$)。

**偏置**通常初始化为 0(ReLU 有时设小正数避免一上来全死)。

**LSUV(逐层序贯单位方差,Mishkin & Matas 2016)**。先做正交初始化,再喂一个 mini-batch 前向跑一遍,**逐层把权重按该层实测激活方差缩放到 1**。等于"先理论估一个,再用真实数据校准一次",对深网络稳健,无需依赖激活种类的假设。

**其它常见初始化与特例**:
- **LeCun 初始化**:$\sigma_W^2=\frac1{n_{\text{in}}}$(只看 fan-in、不翻倍),配 SELU 等自归一化激活(SNN, Klambauer 2017)。
- **Leaky-ReLU / PReLU 的 He 修正**:负区斜率 $\alpha$ 不为 0,砍掉的方差没那么多,因子从 2 变成 $\frac{2}{1+\alpha^2}$;$\alpha=0$ 退回标准 He,$\alpha=1$(纯线性)退回 LeCun。
- **残差网络的零初始化技巧**:把每个残差块最后一个 BN 的 $\gamma$ 初始化为 0(或最后一层权重置 0),让块初始输出 $F(x)\approx0$,网络一开始等价于恒等映射(见 [[52 残差连接与深度可训练性|残差连接]]),训练初期极稳(Goyal 2017 的 "zero-γ" / Fixup, Zhang 2019)。
- **Transformer 的缩放初始化**:深层 Transformer 常把残差分支输出乘 $1/\sqrt{2N}$($N$ 为层数),或用 T-Fixup/DeepNorm 之类方案,本质都是控制残差累加后的方差不随深度膨胀。
- **嵌入层 / 输出层**:词嵌入常用 $\mathcal N(0,1)$ 或较小方差;若做 [[054 词嵌入层与权重绑定|权重绑定]],输入嵌入和输出投影共享一张表,初始化要兼顾两端。

**bias 设小正数的细节**:ReLU 网络若想避免"开局一半神经元就死"(输入恰好把 $z$ 推到负区),可把 $b$ 设成 0.01 之类的小正数让初始 $z$ 略偏正;但有了 BN/合理 He 初始化后通常不必,设 0 即可。

**与归一化的关系**:[[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|BatchNorm/LayerNorm]] 在每层强制重新标准化激活,**降低了对初始化精确尺度的敏感度**;但即便有归一化,合理初始化仍能加速收敛、稳住训练早期。归一化"治标"(每层事后拉回),初始化"治本"(一开始就别跑偏),二者叠加最稳。

## 代码

```python
import numpy as np
np.random.seed(0)

def init(kind, n_in, n_out):
    if kind == "zeros":   return np.zeros((n_out, n_in))            # ❌ 对称,学不动
    if kind == "big":     return np.random.randn(n_out, n_in) * 1.0 # ❌ 不缩放,易爆炸
    if kind == "xavier":  return np.random.randn(n_out, n_in) * np.sqrt(2/(n_in+n_out))
    if kind == "he":      return np.random.randn(n_out, n_in) * np.sqrt(2/n_in)
    if kind == "orthogonal":
        a = np.random.randn(n_out, n_in)
        q, r = np.linalg.qr(a.T if n_out < n_in else a)            # QR 取正交基
        q *= np.sign(np.diag(r))                                  # 符号校正
        return (q.T if n_out < n_in else q)[:n_out, :n_in]

# 模拟信号过 30 层 ReLU 网络,看激活方差是否守恒
def propagate(kind, depth=30, width=256):
    a = np.random.randn(width)
    for _ in range(depth):
        W = init(kind, width, width)
        a = np.maximum(W @ a, 0)          # 线性 + ReLU
    return a.var()

for kind in ["big", "xavier", "he"]:
    print(f"{kind:8s} 30 层后激活方差: {propagate(kind):.3e}")
# big    → 巨大(爆炸)   xavier → 偏小衰减   he → ~O(1) 稳住 ✅(ReLU 专用)

# ❌ 全零:打印任意两行相同 → 对称性破缺失败
W0 = init("zeros", 4, 3); print("zeros 各行相同?", np.allclose(W0[0], W0[1]))  # True
# ✅ He(ReLU)/ Xavier(tanh):随机 + 正确方差缩放,信号稳定传播
```

手算对照:$n_{\text{in}}=100$ 时 Xavier(fan-in 版)$\sigma_W=0.1$、He $\sigma_W=\sqrt{0.02}\approx0.141$,He 比 Xavier 大 $\sqrt2$;代码里 `big`(不缩放)过 30 层 ReLU 方差爆到极大,`he` 维持在 $O(1)$,印证"He 专为 ReLU 守住方差"。

```python
# 用 PyTorch 内置初始化(实际工程更常这么写)
import torch, torch.nn as nn

lin = nn.Linear(512, 128)
nn.init.xavier_normal_(lin.weight, gain=nn.init.calculate_gain('tanh'))   # tanh 配 Xavier
nn.init.zeros_(lin.bias)

conv = nn.Conv2d(64, 128, 3, padding=1)
nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')  # ReLU 配 He
# mode='fan_in' 保前向方差(默认);'fan_out' 保反向梯度方差,ResNet 官方用 fan_out

rnn = nn.Linear(256, 256)
nn.init.orthogonal_(rnn.weight, gain=1.0)   # RNN 隐到隐权重常用正交,稳长程

# ❌ 易错:对 ReLU 网用 xavier(偏小)→ 深网信号逐层衰减
# ❌ 易错:gain 选错(给 ReLU 用了 tanh 的 gain)→ 方差没补够
# ✅ calculate_gain 帮你按激活拿对增益:relu→√2,tanh→5/3,linear/sigmoid→1
print("relu gain =", nn.init.calculate_gain('relu'))   # 1.4142 ≈ √2
```

## 面试高频

- **"为什么不能全 0(或全相同)初始化?"** 同层神经元收到相同梯度、永远同步更新,无法学到不同特征(对称性破缺失败)。必须随机。
- **"Xavier 和 He 的区别?各配什么激活?"** Xavier 方差 $\frac{2}{n_{\text{in}}+n_{\text{out}}}$,配 tanh/sigmoid(对称激活);He 方差 $\frac{2}{n_{\text{in}}}$,配 ReLU——多出的因子 2 补偿 ReLU 砍掉的一半方差。深 ReLU 网必须用 He。
- **"初始化方差为什么要和 fan-in / fan-out 挂钩?"** 输出方差 $=n_{\text{in}}\sigma_W^2\text{Var}(a)$;要让方差守恒(不放大不缩小),$\sigma_W^2$ 必须反比于扇入。否则逐层放大/缩小 → 梯度爆炸/消失。
- **"He 里那个 2 是怎么来的?"** ReLU 期望地把一半神经元清零,输出方差减半;乘 2 把它补回来。
- **"正交初始化好在哪?用在哪?"** $W$ 正交 → 线性变换保范数,信号过任意深度范数不变(动态等距),对极深网络和 RNN(梯度沿时间连乘)特别稳。
- **"有了 BatchNorm/LayerNorm 还需要好初始化吗?"** 归一化降低了敏感度,但好初始化仍能稳住训练早期、加速收敛;两者互补,不互相替代。
- **"Xavier 均匀版里那个 6 怎么来的?"** $U[-a,a]$ 方差 $=a^2/3$,令它等于目标方差 $\frac{2}{n_{\text{in}}+n_{\text{out}}}$,解出 $a=\sqrt{6/(n_{\text{in}}+n_{\text{out}})}$;6 = 2(方差守恒目标)×3(均匀分布方差里的 3)。
- **"fan_in 和 fan_out 模式选哪个?"** fan_in 守前向激活方差,fan_out 守反向梯度方差;ResNet 官方用 fan_out。差别通常不大(BN 会兜底),面试能说清"各守一个方向"即可。
- **"bias 为什么能初始化为 0,而权重不能?"** 只要权重彼此不同,对称性已破缺;bias 相同不会让神经元变成克隆体。LSTM 遗忘门 bias 例外,常设正数(见 [[56 LSTM 门控机制|LSTM]])。
- **"很深的网络(几百层)初始化还要注意什么?"** 残差网常把残差分支末端 BN 的 γ 或末层权重置 0,让初始 $F(x)\approx0$、网络近似恒等;深 Transformer 用 $1/\sqrt{2N}$ 缩放或 DeepNorm 控制残差累加方差。
- **"正交初始化为什么要 QR 分解?gain 怎么定?"** 对随机高斯矩阵做 QR,取 $Q$ 即得正交基(列正交);gain 按激活补偿:ReLU 用 $\sqrt2$、tanh 用 $5/3$、线性用 1。

## 关键事实

- **Xavier / Glorot 初始化**:Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*(AISTATS 2010),方差 $\frac{2}{n_{\text{in}}+n_{\text{out}}}$,为对称激活推导前向/反向方差守恒。
- **He / Kaiming 初始化**:He et al., *Delving Deep into Rectifiers*(ICCV 2015,arXiv:1502.01852),方差 $\frac{2}{n_{\text{in}}}$,专为 ReLU 补偿半方差,使训练极深网络成为可能。
- **正交初始化 / 动态等距**:Saxe, McClelland & Ganguli, *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks*(ICLR 2014)。
- **LSUV(逐层校准单位方差)**:Mishkin & Matas, *All you need is a good init*(ICLR 2016)——正交初始化 + 用一个 batch 实测方差逐层缩放。
- **SELU / LeCun 初始化**:Klambauer et al., *Self-Normalizing Neural Networks*(NeurIPS 2017),配 $\sigma_W^2=1/n_{\text{in}}$ 的自归一化网络。
- **残差/深网零初始化**:Goyal et al.(2017)的 zero-γ 技巧;Zhang et al., *Fixup Initialization*(ICLR 2019)无需归一化即可训练深残差网;Wang et al., *DeepNorm*(2022)把 Transformer 推到 1000 层。
- 初始化对深度网络可训练性的影响,见 Goodfellow, Bengio & Courville《Deep Learning》(2016)第 8.4 节。
