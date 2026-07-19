[[048 路由稳定性：router z-loss]]:MoE 训练里用于稳定 [[043 门控路由与 top-k 选择|门控]] 的辅助项——惩罚路由 logits 的 log-sum-exp 平方,约束 log-partition 的漂移并改善训练稳定性。它既不是“保证 bf16 不溢出”的开关,也不等同于 [[044 专家容量、丢弃与负载均衡损失|负载均衡损失]]；后者主要提供负载均衡压力。

## ① 直觉:均衡管"分得匀不匀",z-loss 管"数值会不会炸"

[[044 专家容量、丢弃与负载均衡损失|负载均衡损失]] 解决的是"token 有没有均摊到各专家"(分布问题)。但 MoE 还有另一类独立的麻烦:**路由 logits 的绝对数值会在训练中悄悄变大**。

为什么会变大?门控 $\ell = x\,W_g^\top$ 没有直接约束其 log-partition 的项;训练中 $W_g$ 和激活值可能使 logits 进入极端区间,并伴随 loss spike、路由不稳定或梯度异常。工程上不应把这简化成“bf16 指数范围小”：bf16 的指数范围接近 fp32，且正确实现的 softmax / `logsumexp` 会先减去最大值、通常在更高精度累积。

**router z-loss** 给 log-partition 加正则：惩罚 $\log\sum_j e^{\ell_j}$(即 logits 的 log-sum-exp,在最大值明显占优时接近最大 logit)的平方。它会选择 softmax 平移不变性留下的“绝对坐标”，让 $\log Z$ 不要无约束漂移；ST-MoE 报告这能改善其训练稳定性，但实际质量与系数仍须在目标模型、数据与精度配置上验证。

一句话:**均衡损失让专家"雨露均沾",z-loss 让数值"别上天"**;两者正交,大规模 MoE 通常一起用。

## ② 例子:logits 漂移与 z-loss 的拉回

设某 token 4 专家,训练早期 logits $\ell=[1.0,\,2.0,\,0.5,\,-0.5]$。

- $\text{LSE}=\log\sum e^{\ell_j}=\log(e^{1}+e^{2}+e^{0.5}+e^{-0.5})\approx \log(2.72+7.39+1.65+0.61)\approx \log 12.37\approx 2.52$。
- z-loss 贡献 $\propto \text{LSE}^2 \approx 2.52^2 = 6.3$,很小。

训练几万步后,若 $\log Z$ 漂移,logits 可能到 $\ell=[40,\,50,\,38,\,30]$:

- $\text{LSE}\approx 50$(被最大值主导),$\text{LSE}^2=2500$ → z-loss 巨大。
- $e^{50}\approx 5\times10^{21}$ **尚未超过** bf16 约 $3.4\times10^{38}$ 的最大有限值；若直接计算 $e^{90}$ 左右才会接近溢出。正确的 `logsumexp` 会用减最大值规避这类中间量溢出，因此不能把 $e^{50}$ 当作 bf16 overflow 的证据。

z-loss 会在训练中给这类漂移额外梯度，让极端的 $\log Z$ 变得昂贵；它是稳定性工具的一环，还应配合稳定的 `logsumexp`、合适的累积精度、梯度/路由监控和回归评测。不要承诺“加了就一定不掉点”。

**验证平移不变性与 z-loss 的边界。** $[1,2,0.5,-0.5]$ 和 $[41,42,40.5,39.5]$(每个都 +40)的 softmax 完全相同，因为

$$
\operatorname{softmax}(\ell+c\mathbf1)=\operatorname{softmax}(\ell),\qquad
\operatorname{LSE}(\ell+c\mathbf1)=\operatorname{LSE}(\ell)+c.
$$

这说明仅凭主任务 softmax 存在一个平移自由度，而 z-loss 会固定这个自由度。**但它不只是“整体缩放且绝不改相对路由”**：令 $z=\operatorname{LSE}(\ell)$，则

$$
\frac{\partial z^2}{\partial \ell_j}=2z\cdot\operatorname{softmax}(\ell)_j.
$$

不同专家收到的梯度一般不同，所以它可能影响相对 logits、top-$k$ 边界和最终质量；这正是要做 ablation 与终态评估的原因。

![[moe-zloss.png]]

## ③ 原理:router z-loss 公式与总损失

对一个 batch 的 $B$ 个 token、每 token 的路由 logits $x^{(i)}\in\mathbb{R}^N$,router z-loss 定义为:

$$
L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log\sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^{2}
$$

括号里就是 $\text{logsumexp}$(softmax 分母的对数),近似于"最大 logit"。平方惩罚让大 logits 受到二次压制。

**为什么是 log-sum-exp 的平方,而不是直接惩罚 $\sum \ell^2$?** LSE 是 softmax 分母的对数，正好正则化 log-partition；直接 L2 会对每个 logit 对称施压、目标不同。两者都会影响相对关系，不能把 router z-loss 解释成“数学上完全不碰路由分布”。选择哪一个应以论文复现与目标任务的稳定性/质量评测为准。

并入总损失(三项线性加权):

$$
L = \underbrace{L_{\text{CE}}}_{\text{主任务 [[30 交叉熵与负对数似然|交叉熵]]}} + \underbrace{\alpha\,B_{\text{aux}}}_{\text{[[044 专家容量、丢弃与负载均衡损失|负载均衡]]}} + \underbrace{c_z\,L_z}_{\text{router z-loss}}
$$

ST-MoE 的实验使用过 $c_z=10^{-3}$；这是论文配置，不是跨模型默认值。系数太大可能损害主任务或路由，太小则可能没有稳定效果，应连同模型、数据、精度、batch 和 router 指标记录下来。

**与 jitter 抖动的关系**:[[043 门控路由与 top-k 选择|门控抖动]](给输入或 logits 加噪)解决的是"路由探索/避免僵化",属于**正则**;z-loss 解决的是"数值稳定"。两者目的不同但都作用在路由上,可同时启用(推理时都关掉)。z-loss 出自 **ST-MoE**(Zoph 2022),该工作系统研究了 MoE 的稳定性与可迁移性,把 [[045 Switch Transformer 与 GShard|Switch]] 的 selective-precision 经验升级成显式损失项。

**MoE 路由的「三类正交辅助手段」一表理清**。零基础最容易把它们搅成一团,记住「各管一件事」:

| 手段 | 管什么 | 公式/做法 | 出处 |
|---|---|---|---|
| 负载均衡统计 | 对过载路由施加压力 | 总目标中的 $\alpha N\sum f_iP_i$ | GShard/Switch |
| router z-loss | 稳定 log-partition 漂移 | $c_z\cdot\frac1B\sum(\text{LSE})^2$ | ST-MoE |
| jitter 抖动 | 路由**探索**(防僵化) | logits/输入加噪 | Switch/ST-MoE |
| 无辅助偏置 | 均匀(替代均衡损失) | 可学习 $b_i$ 调 top-k | DeepSeek-V2/V3 |

它们解决的失效模式不同、可在同一训练配方中出现；噪声类 jitter 通常只在训练开启。**别把「均衡」和「稳定」混为一谈**是这块最高频的考点。

**和 LLM 主干的 z-loss 区别**。PaLM 等也用一个 **logit z-loss** 稳定**最终输出 softmax**(词表上),公式同样是 $\log Z$ 的平方;router z-loss 则作用在 **MoE 路由 logits**(专家上)。同一招、两个地方用,别张冠李戴。

![[moe-门控topk.png]]

## ④ 代码:router z-loss 实现与总损失组合

```python
import torch, torch.nn.functional as F

def router_z_loss(logits):
    # logits:(B, N) 每 token 对 N 个专家的路由打分(softmax 之前)
    lse = torch.logsumexp(logits, dim=-1)             # (B,) = log Σ exp(logit)
    return (lse ** 2).mean()                           # 平方再对 batch 取平均

def load_balance_statistic(logits):
    g = F.softmax(logits, dim=-1)                      # (B, N)
    assign = g.argmax(-1)
    N = logits.size(-1)
    f = torch.bincount(assign, minlength=N).float() / logits.size(0)  # 实际占比
    P = g.mean(0)                                       # 平均门控概率
    return N * (f.detach() * P).sum()              # B_aux：未加权，系数只在总目标加一次

# ❌ 错:把 z-loss 当作稳定 softmax 的替代品，或把 alpha 计两次
def total_loss_wrong(ce, logits):
    return ce + 1e-2 * (1e-2 * load_balance_statistic(logits))

# ✅ 对:稳定算子 + 一次 alpha * B_aux + 经实验选定的 c_z * z-loss
def total_loss(ce, logits, c_z=1e-3):
    alpha = 1e-2
    return ce + alpha * load_balance_statistic(logits) + c_z * router_z_loss(logits)

logits = torch.randn(8, 4) * 5        # 模拟偏大的 logits
print("z-loss:", router_z_loss(logits).item())   # logits 越大,这个值越大
```

## 面试高频

- **Q:router z-loss 解决什么问题?和负载均衡损失有何不同?** z-loss 正则路由 logits 的 log-partition、改善训练稳定性；负载均衡统计对当前过载专家施加路由压力。它们针对不同失效模式，可以一起评估，但没有哪个单独保证稳定或均衡。
- **Q:z-loss 的公式?** $L_z=\frac1B\sum_i(\log\sum_j e^{x^{(i)}_j})^2$,即每 token 路由 logits 的 log-sum-exp 平方、对 batch 平均。
- **Q:为什么惩罚 LSE 而不是直接惩罚 logits 平方?** LSE 正则化的是 softmax 的 log-partition，和直接 L2 是不同目标。二者都可能改变相对 logits；z-loss 的取舍应复现论文并验证路由、损失曲线和下游质量。
- **Q:为什么 logits 会漂移变大?** 门控 $x W_g^\top$ 的尺度无任何约束,训练越久 $W_g$/激活越可能让 logits 整体增大;bf16 动态范围/精度有限,容易触发数值问题。
- **Q:z-loss 会损害模型质量吗?** 可能。ST-MoE 报告其配置下的稳定性/质量结果，但 z-loss 梯度一般会改变相对 logits；$c_z=10^{-3}$ 是该论文的实验值，迁移时需做系数扫描与 held-out 评测。
- **Q:它和 jitter 抖动是一回事吗?** 不是。jitter 给路由加噪是为**探索/防僵化**(正则);z-loss 是为**数值稳定**。两者可同时用,推理时都关闭。
- **Q:softmax 加常数会变吗?这和 z-loss 有何关系?** softmax 不变，但 LSE 会加上该常数。z-loss 因此消除一个未约束的平移自由度；其梯度 $2\,\mathrm{LSE}\cdot\mathrm{softmax}$ 非均匀，不能据此承诺路由决策完全不变。
- **Q:MoE 路由的三类辅助手段各管什么?** 负载均衡=分布均匀、z-loss=数值稳定、jitter=路由探索;三者正交,别混(尤其「均衡 ≠ 稳定」)。
- **Q:router z-loss 和 LLM 输出层的 logit z-loss 一样吗?** 同一招(惩罚 $\log Z$ 平方),但作用位置不同:前者在 MoE 路由 logits(专家上),后者在最终词表 softmax 上(PaLM 用)。
- **Q:bf16 为什么仍要谨慎?** bf16 尾数较短，但指数范围接近 fp32；`e^{50}` 不是溢出。应使用稳定的 `logsumexp`/softmax 与合适累积精度，再用 z-loss 作为经验证的训练稳定化项，而非把它当数值实现的替代品。

## 关键事实

- 出处:Zoph et al.,*ST-MoE: Designing Stable and Transferable Sparse Expert Models*,2022,arXiv:2202.08906。提出 **router z-loss** $L_z=\frac1B\sum_i(\log\sum_j e^{x_j^{(i)}})^2$，并在其模型、数据与训练配方中报告稳定性结果。
- 系数必须有实验上下文：ST-MoE 的实验值包含 $c_z=10^{-3}$；和一次 $\alpha B_{\text{aux}}$ 一并加入 $L=L_{\text{CE}}+\alpha B_{\text{aux}}+c_zL_z$，迁移时应重新扫描。
- 它惩罚路由 logits 的 **log-sum-exp(≈最大 logit 的条件近似)** 平方，选择并正则化 log-partition 的绝对坐标；不保证相对路由、质量或所有 bf16 数值问题不受影响。
- 是 [[045 Switch Transformer 与 GShard|Switch]] 门控 selective-precision(fp32 softmax)经验的显式损失化;被 PaLM-2、众多大规模 MoE 采用。
- 同一招两处用:router z-loss 作用于 MoE 路由 logits;PaLM 的 logit/output z-loss 作用于最终词表 softmax,稳定输出层。
- softmax 平移不变性($\text{softmax}(\ell+c)=\text{softmax}(\ell)$)与 $\operatorname{LSE}(\ell+c)=\operatorname{LSE}(\ell)+c$ 解释了 z-loss 为什么能约束一个未定的绝对坐标；其梯度并不保证保持相对路由不变。
- z-loss(稳定)、[[044 专家容量、丢弃与负载均衡损失|负载均衡损失]](均匀)、[[043 门控路由与 top-k 选择|jitter]](探索)是 MoE 路由的三类正交辅助手段;MoE 用专家替代稠密 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]],部署需 [[049 专家并行 EP 与 MoE 部署|专家并行]]。
