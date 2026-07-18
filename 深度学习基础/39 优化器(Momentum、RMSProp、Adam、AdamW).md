[[39 优化器(Momentum、RMSProp、Adam、AdamW)|优化器]]是给 [[37 梯度下降：BGD、SGD、Mini-batch|梯度下降]] 加"导航和悬挂"的升级版:朴素 SGD 只会"当前梯度 × 学习率",优化器则用**历史梯度的一阶矩(动量)和二阶矩(自适应步长)**让下降更快更稳。主流谱系:SGD → Momentum → RMSProp → Adam → AdamW。

## 直觉

朴素 SGD 有两个毛病:① 在狭长山谷里**来回横跳**(陡方向震荡、缓方向爬得慢);② 所有参数共用一个学习率,但不同参数该走的步长差别很大。

三个升级思路:
- **动量(Momentum)**:像下坡的球带惯性。把历史梯度做指数平均,**同向梯度累积加速、反向震荡相互抵消**——横跳被抚平,沿谷底方向越滚越快。
- **自适应步长(RMSProp)**:给每个参数**单独**调步长。梯度一直很大的参数(陡)自动缩小步长,梯度很小的参数(平)自动放大步长——用"梯度平方的滑动平均"做分母归一化。
- **Adam = 动量 + RMSProp**:一阶矩给方向惯性,二阶矩给逐参数自适应,再加"偏差修正"。它是一个常用起点,不是所有任务的默认最优。
- **AdamW**:把权重衰减**从梯度里解耦出来**,直接作用在参数上;对使用自适应优化器的训练,这让正则强度更可解释([[LLM/061 优化器与超参(AdamW)|LLM 预训练]] 常见)。

一句话:**动量管方向,RMSProp 管步长,Adam 两者兼得,AdamW 把正则修对**。

## 例子

一维狭长谷,某参数连续几步梯度都是 $g=0.1$(同向),$\eta=1$。

**SGD**:每步走 $-\eta g=-0.1$,匀速。

**Momentum**($\beta=0.9$):速度 $v_t=\beta v_{t-1}+g$。
$v_1=0.1,\ v_2=0.19,\ v_3=0.271,\dots\to$ 稳态 $v_\infty=\frac{g}{1-\beta}=\frac{0.1}{0.1}=1$。
**同向梯度让有效步长放大到 10 倍**($\frac{1}{1-\beta}$)——这就是动量"越滚越快"。若梯度反向震荡($+0.1,-0.1,+0.1\dots$),$v$ 会相互抵消、接近 0,**横跳被压住**。

**RMSProp**:某参数梯度一直是 $0.1$,二阶矩 $s\to 0.1^2=0.01$,更新 $\frac{g}{\sqrt s}=\frac{0.1}{0.1}=1$;另一参数梯度一直是 $10$,$s\to100$,更新 $\frac{10}{10}=1$。**不管原始梯度大小,都被归一化到 ~1 量级**——每个参数自适应。

![[nn-优化器对比.png]]

## 原理

设第 $t$ 步梯度 $g_t=\nabla_\theta\mathcal L$,学习率 $\eta$,小常数 $\epsilon$ 防除零。

**① SGD**:
$$\theta_{t}=\theta_{t-1}-\eta\,g_t$$

**② Momentum**(指数平均的一阶矩 = 速度):
$$v_t=\beta v_{t-1}+g_t,\qquad \theta_t=\theta_{t-1}-\eta\,v_t\quad(\beta=0.9)$$
有效步长在稳态放大 $\frac{1}{1-\beta}$ 倍;震荡方向抵消。**Nesterov 加速(NAG)**变体在"前瞻点"$\theta-\eta\beta v$ 处算梯度(先按惯性走一步再看坡度),相当于带"预判刹车",过冲更少、收敛更稳;凸情形下把收敛率从 $O(1/t)$ 提到 $O(1/t^2)$。

**②.5 AdaGrad**(RMSProp 的前身,累加全部历史平方):
$$s_t=s_{t-1}+g_t^2,\qquad \theta_t=\theta_{t-1}-\eta\,\frac{g_t}{\sqrt{s_t}+\epsilon}$$
分母无遗忘地累加,**步长单调衰减**——对稀疏特征(NLP 早期)很好,但后期分母过大、学习率趋 0、过早停止学习。RMSProp 正是把"累加"改成"指数平均"来修这个毛病。

**③ RMSProp**(指数平均的二阶矩,逐参数归一化):
$$s_t=\rho s_{t-1}+(1-\rho)g_t^2,\qquad \theta_t=\theta_{t-1}-\eta\,\frac{g_t}{\sqrt{s_t}+\epsilon}\quad(\rho=0.9)$$
分母是该参数近期梯度的 RMS(不再无限累加,故不会过早停),**陡参数除大数(步小)、平参数除小数(步大)**。

**④ Adam**(动量 + RMSProp + 偏差修正):
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t\quad(\text{一阶矩,方向})$$
$$v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2\quad(\text{二阶矩,步长})$$
$m_0=v_0=0$,初期被偏向 0,故**偏差修正**:
$$\hat m_t=\frac{m_t}{1-\beta_1^{t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{t}}$$
$$\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$
默认 $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$。偏差修正在前几十步显著抬高有效步长(否则 $m_t,v_t$ 偏小、起步太慢)。

**偏差修正手算($t=1$,看它救回多少)**。设首步梯度 $g_1=0.1$。未修正时
$$m_1=(1-0.9)\cdot0.1=0.01,\qquad v_1=(1-0.999)\cdot0.1^2=10^{-5},$$
$m_1$ 只有真值 $0.1$ 的 $1/10$、$v_1$ 只有 $g^2=0.01$ 的 $1/1000$——严重偏 0。修正:
$$\hat m_1=\frac{0.01}{1-0.9}=0.1,\qquad \hat v_1=\frac{10^{-5}}{1-0.999}=0.01,$$
恰好**还原成 $g_1$ 和 $g_1^2$**。对比有效更新方向 $\hat m_1/\sqrt{\hat v_1}=0.1/0.1=1$;若不修正则是 $0.01/\sqrt{10^{-5}}\approx3.16$——量级全错。可见 $t$ 小、尤其 $\beta_2=0.999$ 时,$1-\beta_2^t$ 极小,修正把缩了千倍的二阶矩拉回正确量级,这正是 Adam 起步不"瘸腿"的原因。

**⑤ AdamW(解耦权重衰减,Loshchilov & Hutter 2017)**。问题:在 Adam 里把 L2 正则($+\lambda\theta$ 加进梯度)和自适应步长混在一起——大梯度参数的衰减被 $\sqrt{\hat v_t}$ 除小了,**正则力度被自适应步长扭曲**,L2 不再等价于权重衰减。修正:**把衰减项从梯度里拿出来,直接乘到参数上**:
$$\theta_t=\theta_{t-1}-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\,\theta_{t-1}\Big)$$
即正则项 $\lambda\theta$ **不经过** $\sqrt{\hat v_t}$ 缩放,所有参数被均匀地往 0 拉。这让 [[42 正则化(L2、Dropout、早停、标签平滑)|权重衰减]] 行为可控、泛化更好,是当今 Transformer/LLM 的默认优化器。

![[nn-AdamW解耦.png]]

![[nn-adam自适应.png]]

**⑥ Lion(EvoLved Sign Momentum,Chen et al. 2023)**。用符号搜索发现的新优化器:只维护**一个动量**、更新方向取**符号**(类似 sign-SGD + 动量),不存二阶矩 → 显存约为 Adam 的一半。
$$c_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad \theta_t=\theta_{t-1}-\eta\big(\text{sign}(c_t)+\lambda\theta_{t-1}\big),\quad m_t=\beta_2 m_{t-1}+(1-\beta_2)g_t$$
因更新方向取符号,学习率、batch size 与 weight decay 必须重新搜索;Lion 论文在其配方中使用比 AdamW 更小的学习率,不能把“约 10 倍”当作通用换算。它少维护二阶矩,因而可减少优化器状态内存,但质量—成本取舍仍须在目标任务上复测。

**怎么选(条件化选型卡)**:

| 起点条件 | 可先试 | 必须同时核验 |
|---|---|---|
| 已有成熟 CV 配方、追求最终验证泛化 | SGD + Momentum | 学习率/调度、数据增强、训练步数相同后再比较 |
| Transformer 或稀疏/异尺度梯度、需要稳定基线 | AdamW | 学习率、weight decay、warmup 与裁剪共同调参 |
| 优化器状态显存是瓶颈 | Lion 或其他低状态优化器 | 重搜学习率/weight decay,以验证质量和总成本而非显存单项决策 |

这张表是实验起点,不构成跨任务排名;优化器、调度、batch 和正则是耦合的。

**Adam 的有效学习率直觉**。更新 $\frac{\hat m}{\sqrt{\hat v}}$ 大致是"梯度的信噪比":分子是平均方向、分母是波动幅度。梯度稳定一致(信号强)→ 接近 $\pm1$ 步长大;梯度乱抖(噪声大)→ 被压小。所以 Adam 的每参数有效步长被**归一化到 $\sim\eta$ 量级**,这也是它对学习率不那么敏感、好调的原因。

## 代码

```python
import numpy as np

def make_opt(kind, dim, eta=0.01):
    m = np.zeros(dim); v = np.zeros(dim); t = [0]
    b1, b2, eps, beta, lam = 0.9, 0.999, 1e-8, 0.9, 0.01
    def step(theta, g):
        t[0] += 1
        if kind == "sgd":
            return theta - eta * g
        if kind == "momentum":                      # m 当速度
            m[:] = beta * m + g
            return theta - eta * m
        if kind == "rmsprop":
            v[:] = 0.9 * v + 0.1 * g**2
            return theta - eta * g / (np.sqrt(v) + eps)
        if kind in ("adam", "adamw"):
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g**2
            mh = m / (1 - b1**t[0]); vh = v / (1 - b2**t[0])   # 偏差修正
            upd = mh / (np.sqrt(vh) + eps)
            if kind == "adamw":                     # 解耦权重衰减:λθ 不过 √v
                return theta - eta * (upd + lam * theta)
            return theta - eta * upd                # 普通 Adam(无解耦)
    return step

# 在病态二次型 L = 0.5*(100*x0^2 + x1^2) 上比较(狭长山谷)
A = np.array([100., 1.])
loss = lambda x: 0.5 * (A * x**2).sum()
grad = lambda x: A * x
for kind in ["sgd", "momentum", "rmsprop", "adam", "adamw"]:
    x = np.array([1., 1.]); step = make_opt(kind, 2, eta=0.01)
    for _ in range(200): x = step(x, grad(x))
    print(f"{kind:9s} 终点={x.round(4)}  loss={loss(x):.4e}")

# ❌ 错误理解:Adam 里 weight decay 等价于在 loss 加 L2 —— 在自适应步长下不成立
# ✅ 正确:AdamW 把 λθ 从梯度里解耦,直接作用于参数,正则才不被 √v 扭曲
```

手算对照:Momentum 同向梯度稳态速度 $v_\infty=g/(1-\beta)=0.1/0.1=1$(有效步长 ×10);RMSProp 把梯度归一化到 ~1 量级,与例子一致。代码在病态山谷上,SGD 收敛最慢(陡方向被迫用小学习率),RMSProp/Adam/AdamW 因逐参数自适应明显更快到达极小。

## 面试高频

- **"Momentum 解决什么?$\beta$ 的物理意义?"** 抚平狭长谷里的横向震荡、沿谷底加速;$\beta$ 是历史的保留比例,稳态有效步长放大 $\frac{1}{1-\beta}$($\beta=0.9$ → ×10)。
- **"RMSProp / Adam 为什么要除以梯度平方的均方根?"** 实现**逐参数自适应学习率**:陡参数(梯度大)步长自动变小防震荡,平参数(梯度小)步长自动变大加速。
- **"Adam 的偏差修正为什么必要?"** $m,v$ 初始化为 0,前几步严重偏向 0,导致有效步长偏小、起步慢;除以 $1-\beta^t$ 抵消这个偏差。
- **"AdamW 和 Adam 的区别?何时优先试 AdamW?"** AdamW 把 weight decay 从梯度中**解耦**,直接作用于参数;在自适应优化器中这避免正则项随 $\sqrt{\hat v}$ 被逐参数缩放。它常是 Transformer 的合理基线，但泛化是否改善仍需以目标验证集比较。
- **"Adam 和 SGD+Momentum 怎么选?"** 先固定模型、总训练 token/epoch、调度和增强。Transformer 可从 AdamW 基线开始;成熟 CV 配方可把 SGD+Momentum 与 AdamW 在相同预算下比验证集。没有脱离任务的“首选”。
- **"AdaGrad 的问题是什么?RMSProp 怎么修?"** AdaGrad 无遗忘地累加历史梯度平方,分母只增不减,学习率单调趋 0、后期学不动;RMSProp 改成指数移动平均(有遗忘),分母不爆。
- **"Nesterov 和普通动量区别?"** NAG 在"前瞻点"算梯度(先按惯性走再看坡),带预判刹车、过冲更小;凸情形把 $O(1/t)$ 提到 $O(1/t^2)$。
- **"Lion 优化器了解吗?"** 2023 的符号搜索优化器,只维护一阶动量并取符号更新,可减少优化器状态;迁移时必须重搜学习率、weight decay 与 batch,不能套用固定倍数。
- **"为什么 Adam 对学习率不敏感?"** 更新 $\hat m/\sqrt{\hat v}$ 是逐参数信噪比、被归一化到 $\sim\eta$ 量级,各参数有效步长接近,好调。
- **常见起始超参** $\beta_1{=}0.9,\beta_2{=}0.999,\epsilon{=}10^{-8}$ 可作为 Adam 原论文的起点。是否配 [[40 学习率调度与 warmup、cosine|warmup + 衰减]] 或 [[44 梯度消失、爆炸与梯度裁剪|梯度裁剪]]，取决于模型、batch 与训练曲线。

## 关键事实

- **Adam**:Kingma & Ba, *Adam: A Method for Stochastic Optimization*(ICLR 2015,arXiv:1412.6980),默认 $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$,含偏差修正。
- **AdamW / 解耦权重衰减**:Loshchilov & Hutter, *Decoupled Weight Decay Regularization*(arXiv:1711.05101,2017;ICLR 2019)——指出 Adam 中 L2 正则 ≠ 权重衰减,提出把衰减解耦。
- **RMSProp**:Tieleman & Hinton, Coursera《Neural Networks for ML》讲义(2012,未正式发表);**AdaGrad**:Duchi, Hazan & Singer(JMLR 2011);**Momentum**:Polyak(1964)的 heavy-ball 法,Nesterov 加速(1983)。
- **Lion**:Chen et al., *Symbolic Discovery of Optimization Algorithms*(arXiv:2302.06675,ICLR 2023),EvoLved Sign Momentum,只维护一阶动量。其论文中的学习率设置属于该实验配方,迁移到新模型须重新调参。
- 自适应优化器综述与权衡,见 Goodfellow, Bengio & Courville《Deep Learning》(2016)第 8.5 节。
