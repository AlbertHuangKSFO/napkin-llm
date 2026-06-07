[[061 优化器与超参(AdamW)|优化器与超参(AdamW)]] 是 LLM 预训练的默认更新算法:在 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|Adam]] 的「动量 + 自适应学习率」之上,把**权重衰减(weight decay)从梯度里解耦出来**直接作用于参数,配上一套近乎行业标准的超参(β=(0.9, 0.95)、wd=0.1、ε=1e-8)。

## 直觉

先回忆 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|Adam]] 干两件事:① 用**一阶矩 m**(动量)平滑梯度方向,别被单步噪声带偏;② 用**二阶矩 v**(梯度平方的滑动平均)给每个参数算一个**自适应步长**——经常更新的方向走小步、稀疏的方向走大步。这让 Adam 对学习率不那么敏感,几乎成了 Transformer 训练的默认选择(SGD 在这种损失面上很难调)。

那 AdamW 改了什么?只改一处:**权重衰减怎么加**。

- **L2 正则(老 Adam 的做法)**:把 $\lambda\theta$ 这一项加进梯度 $g$ 里,然后整条梯度被二阶矩 $\sqrt{\hat v}$ 自适应缩放。后果:**梯度大的参数,衰减反而被削弱**;而且衰减强度和学习率纠缠在一起,调参时按下葫芦浮起瓢。
- **解耦权重衰减(AdamW)**:把 $\lambda\theta$ **抽出来**,不进 $m$、不进 $v$,在更新末尾直接从参数里减掉。于是「自适应步」和「参数收缩」各算各的,衰减强度与学习率、与自适应缩放都**解耦**,正则行为干净可控。

一句话:**AdamW = Adam(动量 + 自适应) + 把权重衰减从梯度里拎出来直接缩参数**。LLM 预训练里它就是默认值,几乎不用想别的。

## 例子

取一个参数 $\theta=0.5$,学习率 $\eta=0.001$,衰减 $\lambda=0.1$,某步自适应步算出来是 $\eta\hat m/(\sqrt{\hat v}+\varepsilon)=0.0008$。

- **解耦衰减项** $=\eta\lambda\theta = 0.001\times0.1\times0.5 = 0.00005$。
- 更新:$\theta \leftarrow 0.5 - 0.0008 - 0.00005 = 0.49915$。

注意衰减项只跟 $\theta$ 自己的大小有关($0.5$ 越大、衰减越多),**完全不被 $\sqrt{\hat v}$ 影响**。换成老 Adam 的 L2,这 $0.00005$ 会先混进梯度、再被 $\sqrt{\hat v}+\varepsilon$ 除一遍——梯度大的参数那一除,衰减就被稀释了,这正是 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|Adam]] 泛化不如 AdamW 的根因。

![[train-AdamW更新.png]]

## 原理

**第 $t$ 步**,设梯度 $g_t=\nabla_\theta \mathcal L$。一阶、二阶矩的滑动平均:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2$$

初值 $m_0=v_0=0$,早期会偏向 0,做**偏差校正**:

$$\hat m_t = \frac{m_t}{1-\beta_1^{\,t}},\qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}}$$

**AdamW 更新**(关键在两项相减、互不干扰):

$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon} + \lambda\,\theta_{t-1}\right)$$

对照**原始 Adam + L2** 的写法——它把衰减塞进梯度 $g_t \leftarrow g_t + \lambda\theta_{t-1}$,于是衰减项也经过 $m,v$ 和 $\sqrt{\hat v}$:

$$\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t(\text{含 }\lambda\theta)}{\sqrt{\hat v_t}+\varepsilon}$$

差别就是 $\lambda\theta$ 那一项**在不在分母 $\sqrt{\hat v}$ 的作用范围内**。Loshchilov & Hutter 证明:对纯 SGD,L2 与权重衰减等价;但对**自适应**优化器(Adam),二者不等价,解耦版泛化更好。

$\beta_2$ 取 $0.95$(而非 CV 常用的 $0.999$)是 LLM 的经验:$0.999$ 的二阶矩窗口太长(有效平均约 $1/(1-\beta_2)\approx1000$ 步),遇到 [[066 训练不稳定：loss spike 与对策|loss spike]] 时 $v$ 反应迟钝、容易被一个坏 batch 带飞;$0.95$ 窗口约 20 步,对梯度尺度突变更敏感、更稳。

**完整三步数值更新(带偏差校正)。** 设 $\eta=0.001,\beta_1=0.9,\beta_2=0.95,\varepsilon=10^{-8},\lambda=0$(先看主更新),某参数连续三步梯度 $g_1=g_2=g_3=0.1$:

- **第 1 步**:$m_1=0.1\times0.1=0.01$,$v_1=0.05\times0.01=0.0005$。偏差校正:$\hat m_1=0.01/(1-0.9)=0.1$,$\hat v_1=0.0005/(1-0.95)=0.01$。更新量 $=\eta\hat m_1/(\sqrt{\hat v_1}+\varepsilon)=0.001\times0.1/0.1=0.001$。**注意**:校正后第一步的有效步长 $\approx\eta$,正好等于学习率——这就是偏差校正的意义(不校正的话第一步会因 $m,v$ 偏 0 而步子极小)。
- **第 2 步**:$m_2=0.9\times0.01+0.1\times0.1=0.019$,$v_2=0.95\times0.0005+0.05\times0.01=0.000975$;$\hat m_2=0.019/(1-0.81)=0.1$,$\hat v_2=0.000975/(1-0.9025)=0.01$;更新量仍 $\approx0.001$。
- **规律**:梯度恒定时,Adam 的有效步长稳定在 $\approx\eta$(与梯度大小近似无关,因为分子分母都含同尺度的 $g$)——这正是「自适应、对 lr 不敏感」的来源:Adam 近似把每个方向归一化成单位步长。

**有效步长上界。** 由 $\hat m/\sqrt{\hat v}$ 的形式可证 $|\Delta\theta|\lesssim\eta$(梯度方向一致时趋近 $\eta$,噪声大时更小),所以 Adam 的单步更新幅度天然被 $\eta$ 框住,比 SGD 更不易爆——但这不替代梯度裁剪(见 [[066 训练不稳定：loss spike 与对策|loss spike]])。

**显存账(16–18 字节/参数)。** 混合精度 + AdamW 的标准开销:FP32 主权重 4 + FP16/BF16 权重副本 2 + FP16/BF16 梯度 2(或梯度也存 FP32 共 4)+ 优化器 $m$ 4 + $v$ 4 = **约 16–18 字节/参数**(不含激活)。其中**优化器状态 $m,v$ 各 4 字节、合计 8 字节是大头**。例:7B 模型仅优化器状态就 $7\times10^9\times8=56\text{GB}$,加权重/梯度共约 112–126GB——单张 80GB 卡放不下,这正是 [[070 ZeRO 与 FSDP|ZeRO]] 要分片的首要对象,详见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]]。

## 代码

```python
import torch

model = torch.nn.Linear(4096, 4096)

# ❌ 错误一:用 Adam 并以为 weight_decay 就是 AdamW
#   torch.optim.Adam 的 weight_decay 走的是 L2(混进梯度),不是解耦衰减
opt_bad = torch.optim.Adam(model.parameters(),
                           lr=3e-4, weight_decay=0.1)  # 实为 L2,效果打折

# ❌ 错误二:对所有参数无脑加 wd —— bias、LayerNorm 的 gain/bias 不该衰减
#   (它们维度低、衰减会伤害稳定性,几乎所有 LLM 代码都会排除)

# ✅ 正确:AdamW + 参数分组(只对 2D 权重做 weight decay)
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.ndim >= 2:               # 矩阵权重 → 衰减
        decay.append(p)
    else:                         # bias / norm 的 1D 参数 → 不衰减
        no_decay.append(p)

opt = torch.optim.AdamW(
    [{"params": decay,    "weight_decay": 0.1},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
)

# 一步训练
loss = model(torch.randn(8, 4096)).sum()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 配合梯度裁剪,见 066
opt.step()
opt.zero_grad()
```

```python
# —— 从零手写 AdamW 一步,看清「解耦衰减」和「偏差校正」 ——
def adamw_step(theta, g, m, v, t, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g*g
    m_hat = m / (1 - b1**t)            # 偏差校正:抵消 m,v 初值为 0 的早期偏置
    v_hat = v / (1 - b2**t)
    adaptive = lr * m_hat / (v_hat**0.5 + eps)   # 自适应步,经过 √v 缩放
    decay    = lr * wd * theta                    # ✅ 解耦:衰减只看 θ 自己,不经 √v
    theta = theta - adaptive - decay              # 两项各算各的、相减
    return theta, m, v

theta, m, v = 0.5, 0.0, 0.0
for t in range(1, 4):
    theta, m, v = adamw_step(theta, g=0.1, m=m, v=v, t=t)
    print(f"t={t}  theta={theta:.6f}")
# 梯度恒定时有效自适应步 ≈ lr(对梯度大小不敏感);衰减项独立缩参

# —— 显存账:AdamW 优化器状态是大头 ——
for N in [1e9, 7e9, 70e9]:
    opt_state = N * 8                 # m + v 各 4 字节(FP32)
    total = N * 16                    # +主权重4 +副本2 +梯度2 +m,v 8 = 16~18
    print(f"{N/1e9:>4.0f}B 参数: 仅优化器 {opt_state/1e9:.0f}GB, 训练共 ~{total/1e9:.0f}GB")
# 7B → 仅优化器 56GB;故 ZeRO 首先分片 m,v
```

## 面试高频

- **Adam 和 AdamW 区别?** 唯一区别是权重衰减:Adam(+L2)把 $\lambda\theta$ 加进梯度,会被自适应分母 $\sqrt{\hat v}$ 缩放、且与 lr 纠缠;AdamW 把它解耦,直接从参数减去,正则更干净、泛化更好。
- **为什么不用 SGD 训 LLM?** Transformer 损失面各方向尺度差异大、稀疏更新多,Adam 的逐参数自适应步长几乎是必需;SGD+momentum 在这上面极难调且收敛慢。
- **β=(0.9, 0.95) 里 0.95 为何不是 0.999?** 长窗口二阶矩对梯度突变迟钝,大规模训练时一个坏 batch 容易引发 [[066 训练不稳定：loss spike 与对策|loss spike]];0.95 更敏感、更稳。
- **哪些参数不加 weight decay?** bias 和 [[010 层归一化：Pre-LN 与 Post-LN|LayerNorm]]/RMSNorm 的 gain、bias(1D 参数)通常排除,否则伤稳定性。
- **AdamW 的显存代价?** 每个参数额外存 $m$ 和 $v$ 两份(各与参数同形)。FP32 下优化器状态约 $8$ 字节/参数;混合精度训练总开销约 **16–18 字节/参数**(主权重4+副本2+梯度2+m,v 8),是 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存占用]]的大头。7B 模型仅优化器就 56GB。
- **偏差校正(bias correction)为什么必要?** $m,v$ 初值为 0,早期被严重低估;除以 $1-\beta^t$ 把它放大回无偏估计。否则**第一步步长几乎为 0**、训练前期严重偏慢。$t$ 大时 $\beta^t\to0$,校正自动消失。
- **梯度恒定时 Adam 步长多大?** 约等于 $\eta$——分子 $\hat m$、分母 $\sqrt{\hat v}$ 都含同尺度的 $g$,相除把方向归一化成近似单位步长,这就是「对梯度尺度和 lr 不敏感」的来源,有效步长上界 $\lesssim\eta$。
- **省优化器显存的替代品?** Adafactor(因子分解 $v$、省一半状态)、8-bit Adam(bitsandbytes,$m,v$ 量化到 8 位)、Lion(只存 $m$、符号更新更省)、CAME/Sophia(二阶信息提速)。AdamW 仍是 2020–2025 默认基线。
- **哪些参数不加 weight decay?** bias 和 [[010 层归一化：Pre-LN 与 Post-LN|LayerNorm]]/RMSNorm 的 gain、bias(1D 参数)通常排除,否则伤稳定性;embedding 是否衰减各家不一。

## 关键事实

- AdamW 出自 Loshchilov & Hutter《Decoupled Weight Decay Regularization》(2017,arXiv:1711.05101,ICLR 2019);证明 L2 正则与权重衰减对 SGD 等价、对 Adam 不等价。
- LLM 预训练常用配置:$\beta=(0.9, 0.95)$、weight decay $=0.1$、$\varepsilon=10^{-8}$、grad clip $=1.0$;见 GPT-3(Brown et al., 2020)、LLaMA(Touvron et al., 2023)等技术报告。
- 优化器状态($m,v$)是显存第二大消耗(仅次于激活/参数);这也是 [[070 ZeRO 与 FSDP|ZeRO]] 首先要分片的对象。
- 进阶:Adafactor、Lion、Sophia 等想省状态显存或提速,但 AdamW 仍是 2020–2025 主流默认。
