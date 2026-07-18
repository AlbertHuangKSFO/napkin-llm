[[56 LSTM 门控机制|长短期记忆网络(LSTM)]]给 RNN 加了一条**加性的 cell 状态高速路**和三道**门**(遗忘、输入、输出);它提供一条可学习为近恒等的 CEC 直通支路,从而显著缓解 [[55 长依赖与梯度消失、爆炸|梯度消失]],但不保证完整梯度始终无损。

## 直觉

朴素 RNN 的记忆 $h_t$ 每步都要被 $W_{hh}$ 乘一遍再过 $\tanh$;时间雅可比连乘会随权重、激活状态和谱半径而**消失或爆炸**。LSTM 的关键洞见:**为记忆额外保留一条加性直通支路,而不是让所有信息都反复经过矩阵乘和非线性**。

它引入一条**细胞状态 $c_t$**的加性主干,像一条传送带穿过所有时间步,沿途只做两件温和的操作:**逐元素相乘(遗忘旧的)**和**逐元素相加(写入新的)**。门和候选仍会带来间接梯度路径,但当遗忘门学到接近 1 时,直接 CEC 支路可近似恒等(论文称 **Constant Error Carousel, CEC**)。

三道**门**(都是 $\sigma$ 输出 0~1 的"水龙头",控制信息流量):
- **遗忘门 $f_t$**:旧记忆 $c_{t-1}$ 该保留多少?(1=全留,0=全忘)
- **输入门 $i_t$**:新候选信息该写入多少?
- **输出门 $o_t$**:cell 里的内容该暴露多少给外部 $h_t$?

记忆(长期 $c$)和输出(短期 $h$)**解耦**——这就是"长短期记忆"名字的由来。

## 例子

一维(标量)LSTM,看 cell 高速路如何"无损保存"。设某时刻要**长期记住一个值** 0.8。

若遗忘门学会输出 $f_t=1$、输入门 $i_t=0$:
$$c_t=f_t\cdot c_{t-1}+i_t\cdot\tilde c_t=1\cdot 0.8+0\cdot(\dots)=0.8$$
**一连 50 步都这样,$c$ 始终是 0.8,纹丝不动**——信息被完美保存。若只取不经门和候选依赖的**直接 CEC 支路**,有 $\left.\frac{\partial c_t}{\partial c_{t-1}}\right|_{\text{direct}}=f_t=1$,该支路的梯度连乘 50 次仍是 $1^{50}=1$,**零衰减**。对比朴素 RNN $0.9^{50}\approx0.005$ 已几乎消失。这里说的是可学习的直通支路,不是完整网络的总雅可比恰好等于 1。

再看"该忘就忘":遇到句号、新主语时,网络可学到 $f_t\approx0$,把 $c$ 清空重置——这正是 Gers 2000 补上遗忘门要解决的"记忆无限增长"问题。

**写入一个新值的完整一步手算**。设上一时刻 $c_{t-1}=0.5$,这步网络想"写入新信息"。假设算得 $f_t=0.3$(忘掉大部分旧的)、$i_t=0.8$(大量写入)、$\tilde c_t=1.0$(候选内容)、$o_t=0.6$。
- cell 更新:$c_t=f_t c_{t-1}+i_t\tilde c_t=0.3\times0.5+0.8\times1.0=0.15+0.8=0.95$。
- 隐状态:$h_t=o_t\tanh(c_t)=0.6\times\tanh(0.95)=0.6\times0.7398=0.444$。
看清三件事:① 旧记忆保留了 $0.3$ 比例;② 新内容按 $0.8$ 比例写入;③ 对外暴露的 $h$ 又被输出门 $o=0.6$ 和 $\tanh$ 压了一道。**长期记忆 $c$(0.95)和对外短期表示 $h$(0.444)是两个不同的量**,这就是"长短期"解耦。

**梯度沿 cell 跨 3 步手算**。只沿门值固定时的直接 CEC 支路,若遗忘门连续三步是 $f=[1.0,\,0.9,\,1.0]$,则该支路的传播因子为 $1.0\times0.9\times1.0=0.9$ —— 跨 3 步梯度只掉到 0.9。对比朴素 RNN 同样三步 $\lambda=0.5$ 则 $0.5^3=0.125$。**遗忘门接近 1 时,这条直通支路几乎无损**,这就是 LSTM 缓解梯度消失的数值证据;完整总梯度还会叠加经门和候选的路径。

![[rnn-LSTM门控.png]]

## 原理

时刻 $t$,先拼接输入 $[\,h_{t-1},\,x_t\,]$,过四组带权重的变换。三门用 $\sigma$(0~1 闸门),候选用 $\tanh$(-1~1 内容):
$$f_t=\sigma(W_f[h_{t-1},x_t]+b_f)\quad\text{遗忘门}$$
$$i_t=\sigma(W_i[h_{t-1},x_t]+b_i)\quad\text{输入门}$$
$$\tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c)\quad\text{候选记忆}$$
$$o_t=\sigma(W_o[h_{t-1},x_t]+b_o)\quad\text{输出门}$$

**cell 状态更新(核心,加性)**:
$$\boxed{\,c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t\,}$$

**隐状态(对外输出)**:
$$h_t=o_t\odot\tanh(c_t)$$

**为什么能缓解而非保证“不消失”**——先看 cell 的直接 CEC 支路。将本步门和候选暂时视为固定,有
$$\left.\frac{\partial c_t}{\partial c_{t-1}}\right|_{\text{direct}}=\operatorname{diag}(f_t)$$
它**逐元素、没有 $W$ 矩阵相乘或新的激活导数**。当遗忘门 $f_t\approx1$,跨 $T$ 步这条支路的传播因子 $\approx\prod f\approx1$,不会像朴素 RNN 的 $\text{diag}(1-h^2)W_{hh}$ 那样被每步矩阵连乘主导。

但它不是完整网络的总雅可比:门和候选都依赖 $h_{t-1}$,而 $h_{t-1}=o_{t-1}\odot\tanh(c_{t-1})$ 又依赖 $c_{t-1}$。因此总导数还包括这些间接项:
$$\frac{d c_t}{d c_{t-1}}=\operatorname{diag}(f_t)+\operatorname{diag}(c_{t-1})\frac{d f_t}{d c_{t-1}}+\operatorname{diag}(\tilde c_t)\frac{d i_t}{d c_{t-1}}+\operatorname{diag}(i_t)\frac{d\tilde c_t}{d c_{t-1}}$$
CEC 的价值是总雅可比中始终有一条可学成近恒等的加性支路,其余门控路径是额外项;所以它**大幅缓解**梯度消失,并不承诺每个输入上总梯度都等于 1。这与 [[52 残差连接与深度可训练性|残差连接]] $x+F(x)$ 的思想相近:加法提供稳定的直通项。

**遗忘门偏置技巧**:实践常把 $b_f$ 初始化为正(如 1),让 LSTM 默认"先记着",训练初期更易学长依赖(Jozefowicz 2015)。原理:$\sigma(1)\approx0.73$、$\sigma(2)\approx0.88$,让初始遗忘门偏大、记忆不容易一开始就被冲掉,给优化器时间学到"何时该忘"。

**参数量与计算量**。LSTM 一个 cell 有 4 组权重(三门 + 候选),每组矩阵形状 $H\times(H+D)$,故参数量 $=4H(H+D)$(忽略 bias),约是朴素 RNN($H(H+D)$)的 **4 倍**。这是 LSTM 表达力强但也更重的代价;GRU(3 组)介于二者之间。

**完整的反向直觉(为什么"加性"是关键)**。把 cell 更新 $c_t=f_t c_{t-1}+i_t\tilde c_t$ 和朴素 RNN $h_t=\tanh(Wh_{t-1}+\dots)$ 对比:朴素 RNN 沿时间的雅可比包含 $\text{diag}(1-h^2)W$(**矩阵相乘 + 非线性挤压**)的连乘;LSTM 总雅可比也有经门、候选和 $h_{t-1}$ 的复杂间接项,但其中的直接主干项是 $\text{diag}(f_t)$(**逐元素、无矩阵乘、无挤压**)。“危险的矩阵连乘”不再垄断梯度路径,门控还可学到接近 1;这正是 CEC 的本质,也和 [[52 残差连接与深度可训练性|残差连接]] 的 $+1$ 高速路同源。

**窥孔连接(peephole)**:Gers & Schmidhuber(2000)让三个门除了看 $[h_{t-1},x_t]$,还能直接"窥视"$c_{t-1}$(门的输入加上 $c_{t-1}$ 项)。这让门能基于精确的 cell 内容做开合,在需要精确计时的任务(如数节拍)上更好;但大多数任务非必需,主流实现常省略。

![[rnn-LSTM窥孔连接.png]]

![[rnn-梯度沿时间.png]]

## 代码

```python
import numpy as np
def sigmoid(x): return 1 / (1 + np.exp(-x))

def lstm_step(x, h, c, W, b):
    z = W @ np.concatenate([h, x]) + b      # 一次性算四门(拼接后做大矩阵乘)
    H = h.size
    f = sigmoid(z[0:H]); i = sigmoid(z[H:2*H])
    g = np.tanh(z[2*H:3*H]); o = sigmoid(z[3*H:4*H])
    c = f * c + i * g                        # ✅ cell 加性更新:遗忘旧 + 写入新
    h = o * np.tanh(c)                       # 输出门控制暴露
    return h, c

H, D = 1, 1
W = np.zeros((4*H, H+D)); b = np.zeros(4*H)
b[0:H] = 12.0         # f=σ(12)≈0.9999939 → 50 步后仍近乎全留
b[H:2*H] = -12.0      # i=σ(-12)≈0.0000061 → 几乎不写
h, c = np.zeros(H), np.array([0.8])         # 把 0.8 放进 cell
for t in range(50):                          # 跑 50 步无新输入
    h, c = lstm_step(np.zeros(D), h, c, W, b)
print("50 步后 cell =", round(c[0], 4))      # 0.7998:接近 0.8,但并非精确无损 ✅

# 对比朴素 RNN:同样 50 步,记忆指数衰减
hr = 0.8
for t in range(50): hr = np.tanh(0.9 * hr)   # 反复乘+挤压
print("朴素 RNN 50 步后 =", round(hr, 6))    # ≈0 → 长期记忆丢失 ❌

# ❌ 误区:以为 LSTM "完全不会"梯度消失 —— 错。遗忘门 f<<1 时仍会衰减
# ✅ 正解:门值固定时,直接 CEC 支路有 ∂cₜ/∂cₜ₋₁≈fₜ≈1;总雅可比还含经门/候选的间接项,因此它提供近恒等通道、大幅缓解但非根除

# PyTorch:实际工程一行用 nn.LSTM,并演示遗忘门 bias 初始化技巧
import torch.nn as nn
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
# 遗忘门 bias 初始化为正(PyTorch 把 4 门 bias 拼在一起,f 在第 2 段 [H:2H])
for name, p in lstm.named_parameters():
    if "bias" in name:
        n = p.size(0); p.data[n//4:n//2].fill_(1.0)   # ✅ 让初始 f≈σ(1)≈0.73
import torch
out, (h_n, c_n) = lstm(torch.randn(4, 35, 10))
print("输出:", out.shape, " 末隐 h:", h_n.shape, " 末 cell c:", c_n.shape)

# 参数量对比:LSTM 约是朴素 RNN 的 4 倍
H, D = 20, 10
print("LSTM 参数 ≈", 4*H*(H+D), " 朴素RNN ≈", H*(H+D))   # 2400 vs 600
```

代码直接验证 cell 高速路:遗忘门 $\sigma(12)\approx0.9999939$、输入门 $\sigma(-12)\approx0.0000061$ 时,0.8 在 cell 里 50 步后为约 $0.7998$;这是**近乎保持而非精确无损**。同样设定的朴素 RNN 记忆已衰减到 0。

## 面试高频

- **"LSTM 三个门各管什么?"** 遗忘门 $f$ 控制留多少旧 cell;输入门 $i$ 控制写多少新候选;输出门 $o$ 控制暴露多少 cell 给 $h$。候选 $\tilde c$ 用 $\tanh$ 生成待写内容。
- **"LSTM 为什么能缓解梯度消失?"** cell 更新是**加性**的 $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$。门值固定时,其中的直接 CEC 支路传播因子是 $f_t$(逐元素、无 $W$ 相乘),$f_t\approx1$ 时近恒等;完整总雅可比还含经门和候选的间接项,因此是缓解而非保证。
- **"cell 状态 $c$ 和隐状态 $h$ 区别?"** $c$ 是长期记忆主干、走加性高速路;$h=o\odot\tanh(c)$ 是经输出门筛选后对外的"短期/工作"表示。
- **"门为什么用 sigmoid、候选为什么用 tanh?"** $\sigma\in(0,1)$ 天然是"流量阀";$\tanh\in(-1,1)$ 给候选记忆带正负方向、零中心。
- **"遗忘门偏置为什么常初始化为正?"** 让初始 $f\approx1$,默认保留记忆,训练初期更容易学到长依赖。
- **"LSTM 彻底解决梯度消失了吗?"** 没有,只是大幅缓解;$f$ 远小于 1 时仍衰减。真正"去循环"的并行长程方案是注意力 / Transformer。
- **"LSTM 参数量是朴素 RNN 的几倍?"** 4 倍(三门 + 候选,4 组 $H\times(H+D)$ 权重)。GRU 是 3 倍,所以 GRU 更轻。
- **"遗忘门 cell 雅可比为什么这么关键?"** 严格说 $\operatorname{diag}(f_t)$ 是门值固定时的**直接 CEC 支路**,不是完整总雅可比。它逐元素、无矩阵乘、无新的激活导数,$f\approx1$ 时跨多步连乘仍 $\approx1$;总导数还叠加经门和候选的路径。这条稳定支路是 CEC 的数学核心。
- **"窥孔连接(peephole)是什么?"** 让门也能直接看 $c_{t-1}$,精确计时任务更好(Gers 2000);大多数任务非必需,主流实现常省。
- **"LSTM 和残差连接的联系?"** 都是"加性旁路绕开连乘":LSTM 加性更新 cell,残差 $x+F(x)$ 加恒等捷径,都给梯度不衰减的高速路。
- **"为什么门用 sigmoid 不用 ReLU?"** 门是"流量阀"需要输出严格在 $[0,1]$ 表示"保留多少比例";ReLU 无上界做不了比例阀。

## 关键事实

- **原始 LSTM**:Hochreiter & Schmidhuber, *Long Short-Term Memory*,Neural Computation 9(8), pp. 1735-1780(1997)——提出 cell + 输入/输出门 + Constant Error Carousel,可学跨 1000+ 步的依赖。
- **遗忘门是后加的**:原版只有输入、输出两门;Gers, Schmidhuber & Cummins, *Learning to Forget*(Neural Computation, 2000)补上遗忘门,解决 cell 无限增长,才成为今天的标准 LSTM。
- **窥孔连接(peephole)**:Gers & Schmidhuber(2000)让门也能看到 $c_{t-1}$,精确计时任务更好;非必需。
- **变体大比拼**:Greff et al., *LSTM: A Search Space Odyssey*(2017)系统比较各变体,结论是标准 LSTM 已足够好、遗忘门最关键;Jozefowicz et al.(ICML 2015)建议 $b_f$ 初始化为正。
