[[40 学习率调度与 warmup、cosine|学习率调度]]是在训练过程中**按计划改变学习率** $\eta$,而不是从头到尾用一个固定值。典型套路是 **warmup(先慢慢升温)+ cosine(再平滑退火)**,这是 Transformer/LLM 训练的标配。它直接决定了 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|优化器]] 每步迈多大。

## 直觉

学习率是 [[37 梯度下降：BGD、SGD、Mini-batch|梯度下降]] 最关键的旋钮:太大震荡/发散,太小慢到天荒地老。**而且最优学习率在训练不同阶段是不同的**:

- **开头(warmup)**:参数刚从 [[41 权重初始化(Xavier、He、正交)|随机初始化]] 出发,网络还没"暖机",梯度统计很不稳。一上来就用大学习率容易把刚学到的东西冲垮(尤其 Adam 早期二阶矩估计不准、大批量训练时)。所以**前几百~几千步把 $\eta$ 从 0 线性升到峰值**,给模型一个平稳起步。
- **中后段(退火/decay)**:已经下到谷地附近,大步长会在谷底反复横跳停不下来。**逐渐调小 $\eta$**,让模型稳稳落进极小。cosine 退火用半个余弦曲线把 $\eta$ 平滑地从峰值降到接近 0——比阶梯式 step decay 更顺、收尾更稳。

一句话:**warmup 防止"开局翻车",cosine 退火保证"收尾平稳"**。合起来是一条先升后缓降的曲线。

## 例子

设峰值 $\eta_{\max}=10^{-3}$,warmup 步数 $T_w=1000$,总步数 $T=10000$。

**Warmup 段(线性升)**:第 $t$ 步($t\le T_w$)$\eta_t=\eta_{\max}\cdot\frac{t}{T_w}$。
- $t=100$:$\eta=10^{-3}\times0.1=10^{-4}$
- $t=500$:$\eta=5\times10^{-4}$
- $t=1000$:$\eta=10^{-3}$(到达峰值)

**Cosine 退火段**($T_w<t\le T$):令进度 $p=\frac{t-T_w}{T-T_w}\in[0,1]$,
$$\eta_t=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos(\pi p)\big)$$
取 $\eta_{\min}=0$:
- 退火刚开始 $p=0$:$\cos0=1$,$\eta=\eta_{\max}=10^{-3}$
- 中点 $p=0.5$:$\cos\frac\pi2=0$,$\eta=\tfrac12\eta_{\max}=5\times10^{-4}$
- 结尾 $p=1$:$\cos\pi=-1$,$\eta=0$

曲线:从 0 直线升到 $10^{-3}$,再走半个余弦平滑降到 0。**前快(陡升)、中匀、尾慢(余弦在两端导数为 0,降得轻)**。

![[nn-学习率调度曲线.svg]]

## 原理

**Warmup(线性)**。$t\le T_w$:
$$\eta_t=\eta_{\max}\cdot\frac{t}{T_w}$$
为什么有效(三条一起说):
1. **Adam 二阶矩没暖机**:前几步 $v_t$ 由极少样本估计、方差巨大,$\frac{\hat m}{\sqrt{\hat v}}$ 可能给出离谱的大步长把模型带飞;warmup 用小步长熬过这段。
2. **初始化处梯度统计不稳**:刚从随机初始化出发,损失面陡、梯度噪声大,大步一冲就可能跳进坏区域甚至发散。
3. **大批量训练**:配合"学习率随批量线性放大"时峰值很高,必须用 warmup 渐增地达到,否则一上来的大步长直接 NaN(Goyal 2017)。

**Cosine 退火(SGDR,Loshchilov & Hutter 2017)**。退火段:
$$\eta_t=\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})\Big(1+\cos\big(\pi\,\tfrac{t-T_w}{T-T_w}\big)\Big)$$
余弦的好处:① 两端($p=0,1$)导数为 0 → 退火起步和收尾都平缓,避免突变;② 全程平滑无阶梯跳变。原论文的 **warm restarts** 还会周期性把 $\eta$ 重置回峰值再退火,帮助跳出局部极小(单周期版本即"cosine decay")。

**Transformer 原版调度(Vaswani 2017)**。注意力论文用的是 **warmup + 逆平方根衰减**(不是 cosine):
$$\eta_t=d_{\text{model}}^{-0.5}\cdot\min\big(t^{-0.5},\ t\cdot T_w^{-1.5}\big)$$
- $t<T_w$:$t\cdot T_w^{-1.5}$ 占优 → **线性升**;
- $t>T_w$:$t^{-0.5}$ 占优 → 按 $1/\sqrt t$ **慢降**。
在 $t=T_w$ 两段相等、达到峰值。后来的 BERT/GPT 等多改用 **warmup + 线性 / cosine 衰减**,但"warmup 必备"这一点是共识。

**其它常见调度**:
- **step decay**:每隔若干 epoch 把 $\eta$ ×0.1;简单但跳变点扰动训练。
- **指数衰减**:$\eta_t=\eta_0\gamma^t$,平滑但全程同速率。
- **ReduceLROnPlateau**:验证指标停滞就降 $\eta$;自适应、无需预设总步数。
- **One-Cycle(Smith)**:单峰——先升到峰再降到远低于初值,配合动量反向调度,常能"超收敛"快速训完。
- **WSD(Warmup-Stable-Decay)**:warmup 后**长时间恒定**、最后短促衰减;便于中途加数据继续训(不必预知总步数),近年大模型常用。
现代大模型基本是 **warmup + cosine(到 $\eta_{\min}$)** 或 **warmup + 线性到 0**;$\eta_{\min}$ 常取峰值的 $10\%$ 而非 0(留一点末段学习能力)。

**线性 warmup + 线性衰减(BERT/很多 LLM 的实际选择)**。退火段 $\eta_t=\eta_{\max}\cdot\frac{T-t}{T-T_w}$,从峰值直线降到 0。和 cosine 的区别只是"直线 vs 余弦曲线",cosine 两端更缓;实践中差别不大,都优于无调度。

![[nn-warmup机制.svg]]

## 代码

```python
import numpy as np

def lr_at(t, eta_max=1e-3, eta_min=0.0, warmup=1000, total=10000):
    if t < warmup:                                   # 线性 warmup
        return eta_max * t / warmup
    p = (t - warmup) / (total - warmup)              # 退火进度 ∈[0,1]
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * p))

for t in [0, 100, 500, 1000, 5500, 10000]:
    print(f"step {t:5d}: lr = {lr_at(t):.2e}")
# step     0: 0.00e+00   step  1000: 1.00e-03(峰值)
# step  5500: 5.00e-04(退火中点)   step 10000: 0.00e+00

# Transformer 原版:warmup + 逆平方根(Vaswani 2017)
def lr_transformer(t, d_model=512, warmup=4000):
    t = max(t, 1)
    return d_model ** -0.5 * min(t ** -0.5, t * warmup ** -1.5)
print("transformer peak at t=warmup:", lr_transformer(4000))   # 峰值

# ❌ 错误:从头到尾固定大学习率 —— 早期把初始化冲垮 / 后期谷底横跳停不下来
# opt = SGD(lr=1e-2)  # 全程不变
# ✅ 正确:用调度器,每步更新学习率
# for t, (xb, yb) in enumerate(loader):
#     for g in opt.param_groups: g["lr"] = lr_at(t)   # 或用 torch 的 LambdaLR/CosineAnnealingLR
#     ... loss.backward(); opt.step()
```

```python
# PyTorch 写法(等价):
# sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
# 配合手写 warmup 或 transformers.get_cosine_schedule_with_warmup(...)
```

手算对照:$t=1000$ 时 warmup 结束、$\eta=10^{-3}$(峰值);退火中点 $t=5500$(进度 0.5)$\cos\frac\pi2=0$ → $\eta=5\times10^{-4}$;$t=10000$(进度 1)$\cos\pi=-1$ → $\eta=0$,与曲线和代码输出一致。

## 面试高频

- **"为什么要用 warmup?"** 训练初期参数随机、梯度噪声大、Adam 二阶矩估计不准,大学习率会把模型带偏甚至发散;warmup 用渐增的小步长平稳起步。大批量训练几乎必须。
- **"cosine 退火比 step decay 好在哪?"** 平滑无突变、两端导数为 0(起步收尾都缓),通常收敛更稳、最终精度略高;step decay 在跳变点会扰动训练。
- **"Transformer 原论文用的是 cosine 吗?"** 不是,是 **warmup + 逆平方根衰减** $\eta\propto\min(t^{-0.5}, t\cdot T_w^{-1.5})$;cosine 是后来(SGDR)更常用的退火方式,但"必须 warmup"是共识。
- **"学习率太大/太小各会怎样?"** 太大:loss 震荡甚至 NaN(发散);太小:收敛极慢、可能卡在平坦区/鞍点。调度让两阶段各取所需。
- **"warmup 步数 / 峰值怎么定?"** 经验:warmup 取总步数的 1%~10%(常见几百~几千步);峰值用 **LR range test**(从极小线性升、看 loss 何时开始上翘,取拐点前)或经验值($10^{-3}$ for AdamW@Transformer)。批量放大时峰值随之线性放大。
- **"为什么后期要把学习率降到接近 0?"** 谷底附近大步长会反复横跳停不下来(SGD 噪声不收敛);退火到 ~0(或峰值的 10%)让参数稳稳落进极小。
- **"warm restarts 和单周期 cosine 区别?"** restarts 周期性把 $\eta$ 重置回峰再退火(帮助跳出局部极小、可做快照集成);单周期(cosine decay)只退一次,LLM 预训练常用后者。
- **"One-Cycle / WSD 是什么?"** One-Cycle 单峰先升后大降,常"超收敛";WSD 是 warmup-恒定-末段衰减,便于不预知总步数地续训,近年大模型常用。
- **"linear decay 和 cosine decay 哪个好?"** 实践相近,都远胜无调度;cosine 两端导数为 0 更平滑,linear 实现更简单(BERT 用 linear)。
- **"调度作用在所有参数组上吗?"** 通常是;但常对 bias/LayerNorm 等不做 weight decay,学习率调度一般统一作用。

## 关键事实

- **Cosine 退火 / Warm Restarts(SGDR)**:Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*(arXiv:1608.03983,ICLR 2017),公式 $\eta=\eta_{\min}+\frac12(\eta_{\max}-\eta_{\min})(1+\cos(\pi\frac{T_{cur}}{T_i}))$。
- **Transformer warmup + 逆平方根**:Vaswani et al., *Attention Is All You Need*(NeurIPS 2017),$\eta=d_{\text{model}}^{-0.5}\min(t^{-0.5}, t\cdot T_w^{-1.5})$,默认 $T_w=4000$。
- **大批量训练需 warmup + 线性放大学习率**:Goyal et al., *Accurate, Large Minibatch SGD*(2017)。
- 学习率是最重要的超参数之一,见 Goodfellow, Bengio & Courville《Deep Learning》(2016)第 8.3、11.4 节;预热与调度详见各大 LLM 训练报告(GPT、BERT、Llama)。
