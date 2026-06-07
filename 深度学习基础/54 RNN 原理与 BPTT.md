[[54 RNN 原理与 BPTT|循环神经网络(RNN)]]用一个随时间递推的隐状态 $h_t$ 把"记忆"一路传下去:每个时间步用**同一组权重**吃当前输入 + 上一刻记忆,训练靠"沿时间展开再反向传播"(BPTT)。

## 直觉

[[34 MLP 与万能逼近|前馈网络]]每个输入互相独立,处理不了"顺序"。RNN 的核心改动只有一句话:**给网络加一条指向自己的循环边**,让它把上一步的隐状态 $h_{t-1}$ 也当输入。

于是 $h_t$ 像一个不断被改写的"记忆便签":
- 读到第 $t$ 个词,$h_t$ = 融合(这个新词 $x_t$,之前攒下的记忆 $h_{t-1}$);
- 因为权重在每一步**共享**,RNN 能处理任意长度的序列,且参数量不随序列变长。

把循环边按时间"摊开",RNN 就变成一个**很深的前馈网络**(深度 = 序列长度),每层共享权重。训练它就是在这个展开图上跑反向传播,只是反向要沿时间一路传回去——这就是 **BPTT(Backpropagation Through Time)**。

## 例子

参数极简:标量隐状态,$W_{hh}=0.5,\ W_{xh}=1,\ b=0$,激活用 $\tanh$。输入序列 $x=[1,\ 0,\ 0]$,初始 $h_0=0$。

逐步算 $h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t)$:
- $h_1=\tanh(0.5\cdot0+1\cdot1)=\tanh(1)\approx0.762$
- $h_2=\tanh(0.5\cdot0.762+1\cdot0)=\tanh(0.381)\approx0.364$
- $h_3=\tanh(0.5\cdot0.364+0)=\tanh(0.182)\approx0.180$

注意:**输入只在 $t=1$ 出现一次,记忆却被一路带到 $t=3$**(0.762→0.364→0.180),这就是 RNN"记住过去"的能力。同时也看到记忆在 $W_{hh}<1$ 时**逐步衰减**——这正是 [[55 长依赖与梯度消失、爆炸|长依赖问题]]的根。

**BPTT 一步反向手算(对上代码)**。设损失只看最后一步 $\mathcal L=h_3$,求 $\frac{\partial\mathcal L}{\partial W_{hh}}$。沿时间链回传,每步过 $\tanh'=1-h^2$:
- $t=3$:$\frac{\partial\mathcal L}{\partial h_3}=1$;过 tanh':$\delta_3=1\cdot(1-h_3^2)=1-0.180^2=0.9676$;对 $W_{hh}$ 的本步贡献 $\delta_3\cdot h_2=0.9676\times0.364=0.3522$。误差继续回传:$\frac{\partial\mathcal L}{\partial h_2}=\delta_3\cdot W_{hh}=0.9676\times0.5=0.4838$。
- $t=2$:$\delta_2=0.4838\cdot(1-h_2^2)=0.4838\times(1-0.364^2)=0.4838\times0.8675=0.4197$;贡献 $\delta_2\cdot h_1=0.4197\times0.762=0.3198$。回传:$\frac{\partial\mathcal L}{\partial h_1}=0.4197\times0.5=0.2099$。
- $t=1$:$\delta_1=0.2099\cdot(1-h_1^2)=0.2099\times(1-0.762^2)=0.2099\times0.4194=0.0880$;贡献 $\delta_1\cdot h_0=0.0880\times0=0$。
- **累加**:$\frac{\partial\mathcal L}{\partial W_{hh}}=0.3522+0.3198+0=0.672$。注意三步贡献逐步变小($0.35\to0.32\to0$),这就是梯度沿时间衰减的雏形。代码里的 BPTT 和数值梯度都会得到 $\approx0.672$。

![[rnn-时间展开.png]]

## 原理

**前向(隐状态递推)**,核心公式:
$$h_t=\tanh\big(W_{hh}\,h_{t-1}+W_{xh}\,x_t+b_h\big)$$
$$\hat y_t=W_{hy}\,h_t+b_y\quad(\text{分类再过 }\text{softmax})$$
三组权重 $W_{hh},W_{xh},W_{hy}$ 在**所有时间步共享**。$h_t$ 同时是"本步输出的依据"和"传给下一步的记忆"。

**BPTT(沿时间反向传播)**:总损失 $\mathcal L=\sum_t \ell_t$。共享权重 $W_{hh}$ 的梯度要把**每个时间步的贡献都累加**:
$$\frac{\partial \mathcal L}{\partial W_{hh}}=\sum_{t}\frac{\partial \ell_t}{\partial W_{hh}}=\sum_{t}\sum_{k\le t}\frac{\partial \ell_t}{\partial h_t}\Big(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\Big)\frac{\partial h_k}{\partial W_{hh}}$$
关键在中间那串**雅可比连乘** $\prod \frac{\partial h_j}{\partial h_{j-1}}$,其中
$$\frac{\partial h_j}{\partial h_{j-1}}=\text{diag}\big(1-h_j^2\big)\,W_{hh}\quad(\tanh'=1-\tanh^2)$$
连乘 $t-k$ 项 → 时间跨度越大,这串积越**指数地**趋于 0 或发散,埋下梯度消失/爆炸的雷(下一篇)。本质上 BPTT 就是把 [[16 链式法则|链式法则]]沿时间轴展开,再用 [[20 反向传播的数学推导|反向传播]]。

**TBPTT(截断 BPTT)**:序列很长时,反向只回传固定 $k$ 步(如 $k=35$)就截断,省显存、防爆炸,代价是学不到超过 $k$ 的依赖。常见做法是 $k_1/k_2$ 截断:每 $k_1$ 步做一次反向、每次往回传 $k_2$ 步;隐状态在前向时**跨段保留**(detach 切断梯度但保留数值),所以信息能跨段流动、只是梯度不跨段。

**RNN 的几种输入输出结构(任务对应)**:
- **多对一**(many-to-one):读完整序列只输出一个结果——情感分类、序列打分。
- **一对多**(one-to-many):一个输入生成一串——看图说话、音乐生成。
- **多对多等长**(同步):每步都有输出——词性标注、逐帧标注。
- **多对多不等长**(异步):读完再生成——机器翻译,即 [[58 Seq2Seq 编码器与解码器|Seq2Seq]]。
- **双向 RNN**:正反各跑一遍拼接,每个位置同时看左右上下文(非生成任务用,见 [[57 GRU 与变体|GRU 与变体]])。

**深层(堆叠)RNN**:把第 $\ell$ 层每步的 $h^{(\ell)}_t$ 当第 $\ell+1$ 层的输入,纵向加深表达力;每层有独立权重,但同层跨时间仍共享。实际常用 2–4 层。

![[rnn-梯度沿时间.png]]

## 代码

```python
import numpy as np
np.random.seed(0)

# 手写一个标量/小维 RNN 的前向 + BPTT,验证手算
Whh, Wxh, b = 0.5, 1.0, 0.0
xs = [1.0, 0.0, 0.0]
h = 0.0; hs = [h]
for x in xs:                      # 前向:隐状态递推
    h = np.tanh(Whh * h + Wxh * x + b)
    hs.append(h)
print("h1,h2,h3 =", [round(v, 3) for v in hs[1:]])   # ~0.762, 0.364, 0.180 ✅ 对上手算

# BPTT:设损失只看最后一步 L = h3,反向沿时间求 dL/dWhh
dh = 1.0; dWhh = 0.0
for t in range(len(xs), 0, -1):           # t = 3,2,1
    dz = dh * (1 - hs[t] ** 2)             # 过 tanh':1-h^2
    dWhh += dz * hs[t - 1]                 # 共享权重:梯度累加
    dh = dz * Whh                          # 误差继续沿时间回传
print("dL/dWhh (BPTT) =", round(dWhh, 5))

# 数值梯度校验(见 21 数值梯度检验)
def fwd(w):
    h = 0.0
    for x in xs: h = np.tanh(w * h + Wxh * x + b)
    return h
eps = 1e-5
num = (fwd(Whh + eps) - fwd(Whh - eps)) / (2 * eps)
print("dL/dWhh (数值) =", round(num, 5))   # 与 BPTT 一致

# ❌ 用 for 循环但忘了"梯度沿时间累加"(只取最后一步)→ 梯度错
# ✅ 共享权重的梯度必须 sum over time:dWhh += ...,不能覆盖
```

代码同时验证了**前向递推**(对上手算 0.762/0.364/0.180)和 **BPTT 梯度**(与数值梯度一致)。务必记住:共享权重的梯度是**所有时间步贡献之和**。

```python
# PyTorch:nn.RNN 一行搞定前向 + 自动 BPTT;并演示 TBPTT 的 detach 技巧
import torch, torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)  # 2层堆叠
x = torch.randn(4, 35, 10)        # (batch, seq_len, input)
out, h_n = rnn(x)                 # out: 每步隐状态; h_n: 最后一步(各层)
print("输出:", out.shape, " 末隐:", h_n.shape)  # [4,35,20] [2,4,20]

# TBPTT:长序列分段,段间 detach 隐状态(保数值、断梯度),省显存
def tbptt(rnn, long_seq, chunk=35):
    h = None
    for i in range(0, long_seq.size(1), chunk):
        seg = long_seq[:, i:i+chunk]
        if h is not None:
            h = h.detach()        # ✅ 关键:切断跨段梯度,保留隐状态数值
        out, h = rnn(seg, h)
        # loss = ...; loss.backward(); opt.step()   # 每段各自反向
    return out

# 双向 RNN:每位置同时看左右(非生成任务)
birnn = nn.RNN(10, 20, bidirectional=True, batch_first=True)
o, _ = birnn(x); print("双向输出:", o.shape)   # [4,35,40] 正反拼接 → 维度翻倍

# ❌ 易错:把 h 一直带着不 detach → 计算图无限增长、显存爆 / 二次 backward 报错
# ✅ 长序列必须 detach 隐状态做截断
```

## 面试高频

- **"写出 RNN 的前向公式。"** $h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)$,$\hat y_t=W_{hy}h_t+b_y$;三组权重跨时间步共享。
- **"BPTT 和普通反向传播有什么区别?"** BPTT = 把 RNN 沿时间展开成深层网络后做反向传播;特殊点在共享权重的梯度要对所有时间步**累加**,且要沿时间链一路回传。
- **"RNN 为什么参数量和序列长度无关?"** 权重在每个时间步复用(参数共享),展开只增加计算图深度,不增加参数。
- **"截断 BPTT 解决什么、代价是什么?"** 省显存、缓解梯度爆炸;代价是无法学习超过截断窗口的长依赖。
- **"RNN 训练为什么难?"** 雅可比连乘导致梯度沿时间指数消失/爆炸,长依赖学不动——引出 LSTM/GRU。
- **"RNN 能处理哪些任务结构?"** 多对一(分类)、一对多(看图说话)、多对多等长(序列标注)、多对多不等长(翻译=Seq2Seq);双向 RNN 用于非生成任务。
- **"为什么 TBPTT 要 detach 隐状态?"** 不 detach 则计算图随序列无限增长,显存爆、二次 backward 报错;detach 保留隐状态数值(信息跨段流)但切断梯度(只回传 $k$ 步)。
- **"BPTT 里 tanh' 是多少?为什么重要?"** $\tanh'=1-h^2\in(0,1]$;它出现在每步雅可比 $\text{diag}(1-h^2)W_{hh}$ 里,$\le1$ 使梯度连乘天然偏向衰减(消失),是长依赖难学的直接推手。
- **"RNN 和 1D 卷积/Transformer 处理序列的区别?"** RNN 顺序递推、不可并行、理论无限上下文但实际受梯度限制;1D 卷积局部、可并行但感受野有限;Transformer 自注意力全局、可并行,已基本取代 RNN。

## 关键事实

- **现代 RNN 雏形**:Elman, *Finding Structure in Time*,Cognitive Science 14(2), 1990——提出带上下文单元的"简单循环网络"(Elman 网络)。Jordan 网络(1986)是更早的变体。
- **BPTT 算法**:Werbos, *Backpropagation Through Time: What It Does and How to Do It*,Proc. IEEE 78(10), 1990。
- **RNN 语言模型**:Mikolov et al., *Recurrent Neural Network Based Language Model*,INTERSPEECH 2010——把 RNN 用于 LM 并显著降低困惑度。
- **双向 RNN**:Schuster & Paliwal(IEEE Trans. Signal Processing, 1997)。
- **TBPTT($k_1/k_2$ 截断)** 的标准描述见 Sutskever 博士论文(2013)与 Williams & Peng(1990)。
- **RNN 实战教程**:Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks*(2015 博客)——char-RNN 的经典演示。
- RNN 训练的梯度消失/爆炸理论分析见 Bengio, Simard & Frasconi(1994)与 Pascanu, Mikolov & Bengio(2013),详见 [[55 长依赖与梯度消失、爆炸]]。
