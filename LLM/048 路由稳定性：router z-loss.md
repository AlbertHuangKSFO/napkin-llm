[[048 路由稳定性：router z-loss]]:MoE 训练里专门稳住 [[043 门控路由与 top-k 选择|门控]] 的辅助损失——惩罚路由 logits 的 log-sum-exp 平方,压住 logits 不要漂移变大,防止 softmax 在低精度(bf16)下数值溢出、梯度爆炸,从而让大规模 MoE 训得稳;它和 [[044 专家容量、丢弃与负载均衡损失|负载均衡损失]] 解决的是**两个不同问题**。

## ① 直觉:均衡管"分得匀不匀",z-loss 管"数值会不会炸"

[[044 专家容量、丢弃与负载均衡损失|负载均衡损失]] 解决的是"token 有没有均摊到各专家"(分布问题)。但 MoE 还有另一类独立的麻烦:**路由 logits 的绝对数值会在训练中悄悄变大**。

为什么会变大?门控 $\ell = x\,W_g^\top$ 没有任何东西约束它的尺度;训练越久,$W_g$ 和激活值都可能让 logits 整体越来越大。logits 一大,$\text{softmax}$ 里的 $e^{\ell}$ 就可能**溢出**(尤其 bf16 动态范围小)、梯度爆炸或出现 loss 尖刺(spike),严重时训练直接崩。

**router z-loss** 就是给 logits 套一个"软上限":惩罚 $\log\sum_j e^{\ell_j}$(即 logits 的 log-sum-exp,约等于最大 logit)的平方,把 logits 整体往小了压。它**不改变 softmax 的相对大小关系**(因为是对所有 logits 整体加约束),所以**几乎不损质量**,只换来稳定。

一句话:**均衡损失让专家"雨露均沾",z-loss 让数值"别上天"**;两者正交,大规模 MoE 通常一起用。

## ② 例子:logits 漂移与 z-loss 的拉回

设某 token 4 专家,训练早期 logits $\ell=[1.0,\,2.0,\,0.5,\,-0.5]$。

- $\text{LSE}=\log\sum e^{\ell_j}=\log(e^{1}+e^{2}+e^{0.5}+e^{-0.5})\approx \log(2.72+7.39+1.65+0.61)\approx \log 12.37\approx 2.52$。
- z-loss 贡献 $\propto \text{LSE}^2 \approx 2.52^2 = 6.3$,很小。

训练几万步后,若没约束,logits 漂移到 $\ell=[40,\,50,\,38,\,30]$(相对关系没变、softmax 结果几乎一样!):

- $\text{LSE}\approx 50$(被最大值主导),$\text{LSE}^2=2500$ → z-loss 巨大。
- $e^{50}\approx 5\times10^{21}$,在 bf16($\max\approx 3.4\times10^{38}$ 但精度只 8 位)下已开始丢精度,继续涨就溢出。

z-loss 在 logits 还没涨到危险区时就提供了把它们往下拽的梯度,于是 logits 稳在小数值区,softmax 安全。注意:**两组 logits 的 softmax 输出几乎相同**(都由相对差决定),所以压住绝对尺度不伤路由决策——这正是 z-loss "稳定而不掉点"的关键。

**验证「softmax 只看相对差」**。把 $[1,2,0.5,-0.5]$ 和 $[41,42,40.5,39.5]$(每个都 +40)分别 softmax:由于 softmax 对**所有 logits 同时加常数 $c$ 不变**($\frac{e^{\ell_i+c}}{\sum_j e^{\ell_j+c}}=\frac{e^{\ell_i}}{\sum_j e^{\ell_j}}$),两组输出**完全相同**(都约 $[0.21,0.58,0.13,0.08]$)。可见门控的「选谁、信多少」只由相对差决定,绝对尺度对决策毫无用处——却会在 bf16 下惹出溢出。z-loss 正是「砍掉这个无用又危险的绝对尺度」。

![[moe-zloss.png]]

## ③ 原理:router z-loss 公式与总损失

对一个 batch 的 $B$ 个 token、每 token 的路由 logits $x^{(i)}\in\mathbb{R}^N$,router z-loss 定义为:

$$
L_z = \frac{1}{B}\sum_{i=1}^{B}\Big(\log\sum_{j=1}^{N} e^{x^{(i)}_j}\Big)^{2}
$$

括号里就是 $\text{logsumexp}$(softmax 分母的对数),近似于"最大 logit"。平方惩罚让大 logits 受到二次压制。

**为什么是 log-sum-exp 的平方,而不是直接惩罚 $\sum \ell^2$?** 直接惩罚每个 logit 平方会**改变相对关系**、可能压平有用的路由差异;而 LSE 主要由**最大 logit**主导,惩罚它等于"只压整体尺度的上界",对 softmax 的相对分布扰动最小。

并入总损失(三项线性加权):

$$
L = \underbrace{L_{\text{CE}}}_{\text{主任务 [[30 交叉熵与负对数似然|交叉熵]]}} + \underbrace{\alpha\,L_{\text{aux}}}_{\text{[[044 专家容量、丢弃与负载均衡损失|负载均衡]]}} + \underbrace{c_z\,L_z}_{\text{router z-loss}}
$$

ST-MoE 推荐 $c_z=10^{-3}$ 量级(均衡损失 $\alpha=10^{-2}$ 量级)。系数太大伤主任务,太小压不住漂移。

**与 jitter 抖动的关系**:[[043 门控路由与 top-k 选择|门控抖动]](给输入或 logits 加噪)解决的是"路由探索/避免僵化",属于**正则**;z-loss 解决的是"数值稳定"。两者目的不同但都作用在路由上,可同时启用(推理时都关掉)。z-loss 出自 **ST-MoE**(Zoph 2022),该工作系统研究了 MoE 的稳定性与可迁移性,把 [[045 Switch Transformer 与 GShard|Switch]] 的 selective-precision 经验升级成显式损失项。

**MoE 路由的「三类正交辅助手段」一表理清**。零基础最容易把它们搅成一团,记住「各管一件事」:

| 手段 | 管什么 | 公式/做法 | 出处 |
|---|---|---|---|
| 负载均衡损失 | 分布**均匀**(防坍缩) | $\alpha N\sum f_iP_i$ | GShard/Switch |
| router z-loss | 数值**稳定**(防溢出) | $c_z\cdot\frac1B\sum(\text{LSE})^2$ | ST-MoE |
| jitter 抖动 | 路由**探索**(防僵化) | logits/输入加噪 | Switch/ST-MoE |
| 无辅助偏置 | 均匀(替代均衡损失) | 可学习 $b_i$ 调 top-k | DeepSeek-V2/V3 |

三者正交、可同时用;前三个推理时都关闭。**别把「均衡」和「稳定」混为一谈**是这块最高频的考点。

**和 LLM 主干的 z-loss 区别**。PaLM 等也用一个 **logit z-loss** 稳定**最终输出 softmax**(词表上),公式同样是 $\log Z$ 的平方;router z-loss 则作用在 **MoE 路由 logits**(专家上)。同一招、两个地方用,别张冠李戴。

![[moe-门控topk.png]]

## ④ 代码:router z-loss 实现与总损失组合

```python
import torch, torch.nn.functional as F

def router_z_loss(logits):
    # logits:(B, N) 每 token 对 N 个专家的路由打分(softmax 之前)
    lse = torch.logsumexp(logits, dim=-1)             # (B,) = log Σ exp(logit)
    return (lse ** 2).mean()                           # 平方再对 batch 取平均

def load_balance_loss(logits, alpha=1e-2):
    g = F.softmax(logits, dim=-1)                      # (B, N)
    assign = g.argmax(-1)
    N = logits.size(-1)
    f = torch.bincount(assign, minlength=N).float() / logits.size(0)  # 实际占比
    P = g.mean(0)                                       # 平均门控概率
    return alpha * N * (f.detach() * P).sum()

# ❌ 错:只有主任务 + 均衡损失,logits 无尺度约束 → 训久了 softmax 溢出、loss 尖刺
def total_loss_wrong(ce, logits):
    return ce + load_balance_loss(logits)

# ✅ 对:再加 c_z · z-loss,压住 logits 尺度,稳定不掉点
def total_loss(ce, logits, c_z=1e-3):
    return ce + load_balance_loss(logits) + c_z * router_z_loss(logits)

logits = torch.randn(8, 4) * 5        # 模拟偏大的 logits
print("z-loss:", router_z_loss(logits).item())   # logits 越大,这个值越大
```

## 面试高频

- **Q:router z-loss 解决什么问题?和负载均衡损失有何不同?** z-loss 管**数值稳定**(压住 logits 绝对尺度,防 softmax 溢出/梯度爆炸);负载均衡损失管**分布均匀**(token 均摊各专家)。两者正交,通常同时用。
- **Q:z-loss 的公式?** $L_z=\frac1B\sum_i(\log\sum_j e^{x^{(i)}_j})^2$,即每 token 路由 logits 的 log-sum-exp 平方、对 batch 平均。
- **Q:为什么惩罚 LSE 而不是直接惩罚 logits 平方?** LSE 主要由最大 logit 主导,只压整体尺度上界,**不改变 softmax 的相对分布** → 稳定而几乎不掉点;直接压每个 logit 会扰乱有用的路由差异。
- **Q:为什么 logits 会漂移变大?** 门控 $x W_g^\top$ 的尺度无任何约束,训练越久 $W_g$/激活越可能让 logits 整体增大;bf16 动态范围/精度有限,容易触发数值问题。
- **Q:z-loss 会损害模型质量吗?** 基本不会——它压的是绝对尺度而非相对关系,ST-MoE 报告显著提升稳定性而质量不降;系数 $c_z\approx10^{-3}$。
- **Q:它和 jitter 抖动是一回事吗?** 不是。jitter 给路由加噪是为**探索/防僵化**(正则);z-loss 是为**数值稳定**。两者可同时用,推理时都关闭。
- **Q:softmax 加常数会变吗?为什么这保证 z-loss 不掉点?** 不变(平移不变性)。两组只差常数的 logits softmax 完全相同,所以压绝对尺度不动路由决策。
- **Q:MoE 路由的三类辅助手段各管什么?** 负载均衡=分布均匀、z-loss=数值稳定、jitter=路由探索;三者正交,别混(尤其「均衡 ≠ 稳定」)。
- **Q:router z-loss 和 LLM 输出层的 logit z-loss 一样吗?** 同一招(惩罚 $\log Z$ 平方),但作用位置不同:前者在 MoE 路由 logits(专家上),后者在最终词表 softmax 上(PaLM 用)。
- **Q:bf16 为什么特别需要 z-loss?** bf16 指数范围够大但**尾数只 7~8 位**,大 logits 下 $e^\ell$ 精度迅速流失、易溢出/出 NaN;z-loss 把 logits 压在小区间规避。

## 关键事实

- 出处:Zoph et al.,*ST-MoE: Designing Stable and Transferable Sparse Expert Models*,2022,arXiv:2202.08906。提出 **router z-loss** $L_z=\frac1B\sum_i(\log\sum_j e^{x_j^{(i)}})^2$,显著改善大规模 MoE 训练稳定性而不损质量。
- 推荐系数:$c_z\approx 10^{-3}$,与负载均衡损失 $\alpha\approx10^{-2}$ 并列加入总损失 $L=L_{\text{CE}}+\alpha L_{\text{aux}}+c_z L_z$。
- 它惩罚路由 logits 的 **log-sum-exp(≈最大 logit)** 平方,只压绝对尺度、不改相对分布 → 防 softmax 在 bf16 下溢出、梯度爆炸与 loss 尖刺。
- 是 [[045 Switch Transformer 与 GShard|Switch]] 门控 selective-precision(fp32 softmax)经验的显式损失化;被 PaLM-2、众多大规模 MoE 采用。
- 同一招两处用:router z-loss 作用于 MoE 路由 logits;PaLM 的 logit/output z-loss 作用于最终词表 softmax,稳定输出层。
- softmax 平移不变性($\text{softmax}(\ell+c)=\text{softmax}(\ell)$)是 z-loss「压绝对尺度而不改相对分布」的数学基础;故稳定且几乎不掉点。
- z-loss(稳定)、[[044 专家容量、丢弃与负载均衡损失|负载均衡损失]](均匀)、[[043 门控路由与 top-k 选择|jitter]](探索)是 MoE 路由的三类正交辅助手段;MoE 用专家替代稠密 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]],部署需 [[049 专家并行 EP 与 MoE 部署|专家并行]]。
