[[043 门控路由与 top-k 选择]]:MoE 里负责"派活"的小网络——给每个 token 算出它对各专家的**门控分**,选出分最高的 top-k 个 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN 专家]],只让这几个专家干活,再用门控分加权汇总它们的输出。

## ① 直觉:一个"分诊台",把每个 token 派给最合适的少数专家

[[042 MoE 动机：稀疏激活与容量解耦|MoE]] 把一层稠密 FFN 换成 $N$ 个并列的小 FFN(叫"专家")。但不能让每个 token 都过全部 $N$ 个专家——那就退化成 $N$ 倍算力的稠密层了。于是需要一个**门控网络(router / gate)**当分诊台:它看一眼 token,打分,只把这个 token 送进**得分最高的 $k$ 个专家**(通常 $k=1$ 或 $2$),其余专家这一步对它**完全不算**。

这就是"**条件计算**":参数(容量)随专家数线性增长,但单 token 的算力只跟 $k$ 有关、几乎恒定。门控本身极轻——通常就是**一个线性层** $W_g\in\mathbb{R}^{N\times d}$,把 $d$ 维 token 映射成 $N$ 个分数。

关键直觉:门控的输出**既是"选谁"(top-k 索引),也是"信多少"(加权系数)**。选中专家算完后,不是简单求和,而是按门控分加权——分高的专家说话更算数。

## ② 例子:8 专家、top-2 走一遍

设 $N=8$ 个专家,$k=2$。某 token $x$ 进来:

1. 门控算 logits:$\ell = x\,W_g^\top \in \mathbb{R}^8$,比如 $\ell=[-1.2,\,2.3,\,-2.0,\,0.1,\,1.8,\,-1.5,\,-2.4,\,-1.0]$。
2. softmax(见 [[27 Softmax 与温度|Softmax]]):得到概率 $g=[0.05,\,0.41,\,0.03,\,0.08,\,0.32,\,0.04,\,0.02,\,0.05]$。
3. 取 top-2:专家 2(0.41)和专家 5(0.32)。其余 6 个专家**不参与计算**。
4. 输出:$y = \dfrac{0.41}{0.41+0.32}\,\text{FFN}_2(x) + \dfrac{0.32}{0.41+0.32}\,\text{FFN}_5(x)$。

注意第 4 步把两个选中专家的门控分**重新归一化**(0.41 与 0.32 各占 56% 与 44%),让加权系数和为 1——这是 Mixtral 等用的做法。Switch(top-1)则直接用未归一化的 $g$ 当缩放系数。

8 个专家、每个是 7B 的 FFN,总容量近 8 倍;但每 token 只算 2 个 → 算力只有稠密 2-专家级别。这正是 [[046 Mixtral 稀疏 MoE|Mixtral 8×7B]] 的配置。

![[moe-门控topk.png]]

## ③ 原理:门控公式与"为什么是 top-k 而不是软混合"

门控网络对每个 token 独立计算:

$$
g(x) = \text{softmax}\big(x\,W_g^\top\big),\qquad W_g\in\mathbb{R}^{N\times d},\ g(x)\in\mathbb{R}^N
$$

设 $\mathcal{T}=\text{TopK}\big(g(x),k\big)$ 为门控分最大的 $k$ 个专家索引,则 MoE 层输出:

$$
y = \sum_{i\in\mathcal{T}} \frac{g_i(x)}{\sum_{j\in\mathcal{T}} g_j(x)}\;\text{FFN}_i(x)
$$

**为什么要硬选 top-k,而不是对所有专家做软加权($\sum_{i=1}^N g_i\,\text{FFN}_i$)?** 软加权要算全部 $N$ 个 FFN,失去稀疏性、算力爆炸。top-k 只算 $k$ 个,这是 MoE 省算力的根本。代价是 $\text{TopK}$ 不可导(选择是离散的)——但门控分 $g_i$ 本身可导,梯度通过"被选中专家的加权系数"回流到 $W_g$,所以 router 仍能端到端训练。

**这带来一个隐患**:门控倾向于反复选同几个专家(强者愈强),导致**专家坍缩 / 负载失衡**。所以 top-k 路由几乎总要配一个 [[044 专家容量、丢弃与负载均衡损失|辅助负载均衡损失]],逼 token 在专家间均摊。

**抖动(jitter)**:训练时常给门控输入乘一点均匀噪声(如 Switch 的 input jitter),或给 logits 加噪,打破"门控分极接近时总选同一个"的僵局,鼓励探索。推理时关掉。

**Shazeer 2017 的原始「noisy top-k gating」公式**(很多面试官会顺手考它的来历)。最早把可学习门控 + top-k 写进论文的形式是:先在门控 logits 上加一项**可学习幅度的高斯噪声**,再取 top-k,再 softmax:

$$
H(x)_i = (x\,W_g)_i + \epsilon\cdot \text{softplus}\big((x\,W_{\text{noise}})_i\big),\quad \epsilon\sim\mathcal{N}(0,1)
$$

$$
g(x) = \text{softmax}\big(\text{KeepTopK}(H(x),\,k)\big)
$$

噪声的作用:① **鼓励探索**(打破强者愈强);② 噪声幅度本身可学习,模型自己决定每个专家「该多随机」。后来 Switch 把它简化成无噪声的 input jitter,Mixtral 干脆 top-k + softmax 不加噪——但「加噪做探索」的思路一直在。

**token-choice vs expert-choice(两种路由方向)**。上面讲的都是 **token 选专家**(token-choice):每个 token 挑它的 top-k 专家,问题是可能挤爆某些专家(需均衡损失)。还有反向的 **expert-choice**:每个专家挑它最想要的 top-C 个 token,**天然负载均衡**(每个专家恰好收 C 个,不需辅助损失),代价是某些 token 可能没被任何专家选中(被丢)。两种是对偶关系,面试偶尔深挖。

![[moe-token-vs-expert-choice.png]]

![[moe-Switch路由.png]]

## ④ 代码:top-k 门控前向(❌软混合 vs ✅稀疏 top-k)

```python
import torch, torch.nn.functional as F

class TopKGate(torch.nn.Module):
    def __init__(self, d, n_experts, k=2):
        super().__init__()
        self.k = k
        self.w_g = torch.nn.Linear(d, n_experts, bias=False)  # 门控:就一个线性层
        self.experts = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(d, 4*d), torch.nn.GELU(),
                                torch.nn.Linear(4*d, d)) for _ in range(n_experts))

    # ❌ 错:对所有专家软加权 → 算了全部 N 个 FFN,毫无稀疏可言
    def forward_dense_wrong(self, x):                 # x:(tokens, d)
        g = F.softmax(self.w_g(x), dim=-1)            # (tokens, N)
        outs = torch.stack([e(x) for e in self.experts], 1)  # 全算!N 倍算力
        return (g.unsqueeze(-1) * outs).sum(1)

    # ✅ 对:只算被选中的 k 个专家,门控分重归一化加权
    def forward(self, x):                             # x:(tokens, d)
        logits = self.w_g(x)                          # (tokens, N)
        g = F.softmax(logits, dim=-1)
        topv, topi = g.topk(self.k, dim=-1)           # (tokens, k) 值与索引
        topv = topv / topv.sum(-1, keepdim=True)      # 重归一化,系数和=1
        y = torch.zeros_like(x)
        for slot in range(self.k):                    # 对每个 top 槽位
            idx = topi[:, slot]                       # 该槽选中的专家 id
            for e_id in idx.unique():                 # 把送往同一专家的 token 批处理
                mask = idx == e_id
                y[mask] += topv[mask, slot:slot+1] * self.experts[e_id](x[mask])
        return y
```

(真实实现会用 scatter/分组 + capacity 截断做高效批处理并落到不同 GPU,见 [[049 专家并行 EP 与 MoE 部署|专家并行]]。)

```python
# 两种归一化口径的区别:先 softmax 后取 top-k  vs  先取 top-k 后 softmax
import torch, torch.nn.functional as F
logits = torch.tensor([-1.2, 2.3, -2.0, 0.1, 1.8, -1.5, -2.4, -1.0])

# 口径 A(Mixtral 等):先 TopK 把其余置 -inf,再 softmax → 选中专家权重自动和=1
vA, iA = logits.topk(2)
gA = F.softmax(vA, dim=-1)
print("先TopK后softmax:", dict(zip(iA.tolist(), gA.round(decimals=3).tolist())))

# 口径 B:先对全部 softmax,再取 top-2,再手动重归一化(数值上等价于 A)
gB_all = F.softmax(logits, dim=-1)
vB, iB = gB_all.topk(2)
gB = vB / vB.sum()
print("先softmax后重归一:", dict(zip(iB.tolist(), gB.round(decimals=3).tolist())))
# 两者结果相同;但 Switch(top-1)用「未归一化的 softmax 分」当缩放系数,不重归一
```

## 软 MoE 与「无丢弃」路线(扩展认知)

top-k 硬路由不是唯一选择,了解变体能答得更深:

- **Soft MoE**(2023):不做离散选择,而是让每个「slot」对所有 token 做加权组合(可微的软分派),**完全没有丢弃、无需均衡损失**,但失去了「单 token 只算 k 个专家」的稀疏性,更适合视觉等场景。
- **expert-choice**:专家选 token,天然均衡、无需辅助损失(见上)。
- **可学习偏置均衡(DeepSeek-V2/V3)**:仍是 token-choice top-k,但用每专家可学习偏置 $b_i$ 调路由打分实现无辅助损失均衡(见 [[047 DeepSeek MoE：细粒度与共享专家|DeepSeekMoE]])。
- **共享专家**:部分专家「人人必经」不走门控,承载通用知识,减轻路由专家负担(DeepSeek)。

一句话:**硬 top-k(主流,省算但要管均衡)↔ 软/专家选/偏置(各自换掉「均衡损失」这块麻烦)**。

## 面试高频

- **Q:门控网络是什么结构?** 通常就**一个线性层** $W_g\in\mathbb{R}^{N\times d}$ + softmax,极轻量;它对每个 token 独立打分,选 top-k 专家。
- **Q:top-k 不可导,router 怎么训练?** $\text{TopK}$ 这个"选择"不可导,但**被选中专家的门控权重 $g_i$ 可导**,梯度经加权系数回流到 $W_g$;没被选中的专家这一步拿不到该 token 的梯度。
- **Q:为什么不用对所有专家软加权?** 那要算全部 $N$ 个 FFN,失去 MoE 省算力的意义;top-k 才保证单 token 算力恒定。
- **Q:top-1 和 top-2 怎么选?** top-1([[045 Switch Transformer 与 GShard|Switch]])通信/实现最省但表达力略弱;top-2([[046 Mixtral 稀疏 MoE|Mixtral]]、GShard)质量更稳、几乎是事实标准。
- **Q:门控输出既选专家又当权重,会有什么问题?** 容易"强者愈强"导致 [[044 专家容量、丢弃与负载均衡损失|专家坍缩]],必须配辅助负载均衡损失;logits 还可能漂移变大,需 [[048 路由稳定性：router z-loss|router z-loss]] 稳住。
- **Q:Shazeer 2017 的原始门控为什么要加噪?** noisy top-k gating 在 logits 上加可学习幅度的高斯噪声再取 top-k,目的是鼓励探索、打破强者愈强;噪声幅度本身可学习。
- **Q:token-choice 和 expert-choice 区别?** token-choice(主流):token 选 top-k 专家,可能挤爆专家需均衡损失;expert-choice:专家选 top-C token,天然均衡无需辅助损失,但可能丢 token。两者对偶。
- **Q:「先 TopK 后 softmax」和「先 softmax 后重归一」一样吗?** 数值上等价(选中专家权重和都为 1);Mixtral 用前者更简洁,Switch top-1 则直接用未归一化 softmax 分当缩放系数。
- **Q:Soft MoE 怎么避免丢弃?** 用可微的软分派(每 slot 对所有 token 加权组合)代替离散 top-k,无丢弃、无需均衡损失,但牺牲了「单 token 只算 k 个」的稀疏性。

## 关键事实

- "可学习门控 + top-k 稀疏选择"的现代形态出自《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》(Shazeer et al., 2017,arXiv:1701.06538),其用 **noisy top-k gating**(加噪 + 取 top-k)。
- 门控通常是**单线性层 $W_g$ + softmax**;输出既是 top-k 索引、又是加权系数(选中后常重归一化)。
- top-1 路由由 Switch Transformer(Fedus et al., 2021,arXiv:2101.03961)推广;top-2 由 GShard(Lepikhin et al., 2020,arXiv:2006.16668)用于 Transformer FFN,后被 Mixtral(2024)沿用。
- 输入抖动(input jitter / multiplicative noise)出自 Switch、ST-MoE,用于鼓励路由探索、缓解僵化;推理时关闭。
- 原始 noisy top-k:$H(x)_i=(xW_g)_i+\epsilon\cdot\text{softplus}((xW_{\text{noise}})_i)$ 加可学习幅度噪声再取 top-k(Shazeer et al., 2017)。
- expert-choice 路由(Zhou et al., 2022,arXiv:2202.09368):专家选 top-C token,天然负载均衡;Soft MoE(Puigcerver et al., 2023,arXiv:2308.00951)用软分派去掉离散选择与丢弃。
- 两种归一化口径(先 TopK 后 softmax / 先 softmax 后重归一)数值等价;DeepSeek-V2/V3 用可学习专家偏置做无辅助损失均衡(见 [[047 DeepSeek MoE：细粒度与共享专家|DeepSeekMoE]])。
