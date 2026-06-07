[[045 Switch Transformer 与 GShard]]:两篇把 MoE 真正塞进 Transformer 并跑到超大规模的奠基工作——GShard(2020)用 **top-2** 路由把 MoE 接到 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]] 层、给出容量与均衡损失;Switch(2021)进一步简化成 **top-1**(每 token 只走 1 个专家),证明 $k=1$ 也能保质,还把训练提速最高 7×。

## ① 直觉:把 FFN 换成"一堆 FFN + 一个分诊台",再想办法在上千张卡上跑

标准 Transformer 每层的 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]] 是稠密的:每个 token 都过同一套权重。GShard 与 Switch 的核心改动只有一处——**把这层 FFN 替换成 $N$ 个并列的专家 FFN + 一个 [[043 门控路由与 top-k 选择|门控]]**;注意力层原封不动。

两者的分歧只在"每 token 送几个专家":

- **GShard:top-2**。继承 Shazeer 2017 的思路,每 token 选 2 个专家(第 1 个是 argmax,第 2 个按门控概率随机/比例选),用门控分加权。
- **Switch:top-1**。Fedus 等大胆假设 $k=1$ 就够——每 token **只送 1 个专家**。这听起来太激进,但实验证明在算力相同下质量不降,而**通信量、路由实现都减半**。

为什么 top-1 这么香?MoE 的瓶颈往往不是算力而是 **all-to-all 通信**(token 要跨卡发给对应专家)。top-1 把"每 token 发 2 份"减成"发 1 份",通信直接对半砍。代价是单专家更容易失衡,所以 Switch 把 [[044 专家容量、丢弃与负载均衡损失|容量因子和辅助均衡损失]] 调得更讲究。

## ② 例子:同一 token 在 GShard 与 Switch 的不同命运

设 $N=4$ 专家,门控 softmax 后 $g=[0.1,\,0.5,\,0.35,\,0.05]$。

**GShard(top-2)**:选专家 2(0.5)和专家 3(0.35)。

$$
y = \frac{0.5}{0.5+0.35}\text{FFN}_2(x) + \frac{0.35}{0.5+0.35}\text{FFN}_3(x)
$$

每 token 跨卡发 2 份、算 2 个 FFN。

**Switch(top-1)**:只选专家 2。

$$
y = g_2\cdot\text{FFN}_2(x) = 0.5\cdot\text{FFN}_2(x)
$$

注意 Switch 直接用未归一化的门控分 $g_2=0.5$ 当缩放系数(让门控分参与梯度,作为路由置信度)。每 token 只发 1 份、算 1 个 FFN。

规模对照:Switch-C 把专家数推到 **2048**,做出 **1.6 万亿(1.6T)参数**的模型,但每 token 激活的算力仍约等于一个稠密 FFN——这就是 [[042 MoE 动机：稀疏激活与容量解耦|容量与算力解耦]]的极致。

**「7× 提速」到底是怎么算出来的(别误读)**。Switch 论文报告的 7× 是「**在相同算力(FLOPs)预算下,达到同等预训练 loss 所需的时间**」——即同样的钱(算力),Switch-Base 比稠密 T5-Base 快约 7 倍达标。注意这**不是**「同参数量下快 7 倍」(MoE 参数量远大于稠密),而是「同算力下、靠稀疏带来的更大有效容量,收敛更快」。面试若问「MoE 为什么快」,要答清是「同算力预算下收敛更快」,不是「单次前向更快」。

![[moe-Switch路由.png]]

## ③ 原理:Switch 层的前向与三个关键设计

Switch 层对每个 token $x$:

$$
\ell = x\,W_g^\top,\quad i^\star = \arg\max_j\, \text{softmax}(\ell)_j,\quad
y = \text{softmax}(\ell)_{i^\star}\cdot \text{FFN}_{i^\star}(x)
$$

三个让它训得稳的设计:

1. **容量因子与丢弃**(见 [[044 专家容量、丢弃与负载均衡损失|044]]):每专家容量 $C=\text{cf}\cdot\frac{T}{N}$,溢出 token 丢弃走残差。top-1 下失衡风险更高,$cf$ 要选好。
2. **辅助负载均衡损失** $L_{\text{aux}}=\alpha N\sum_i f_i P_i$(Switch 取 $\alpha=10^{-2}$),逼 token 均摊。
3. **训练精度技巧**:门控部分用 **fp32(selective precision)** 算 softmax,避免 bf16 下路由 logits 的数值不稳;并用更小的初始化。这是后来 [[048 路由稳定性：router z-loss|router z-loss]] 要系统解决的同一类问题。

GShard 的另一贡献是**工程**:它提出 `SPMD` + 自动分片注解,让 MoE 层在数千 TPU 上做高效 **all-to-all** 路由通信,是 [[049 专家并行 EP 与 MoE 部署|专家并行(EP)]] 的源头。

**Switch 的另外两招实用技巧**:

- **专家 dropout(expert dropout)**:微调小数据集时,专家容易过拟合。Switch 对专家层用更大的 dropout(如 0.4),对非专家层用小 dropout——分别对待,缓解 MoE 微调过拟合。
- **蒸馏回稠密**:1.6T 的 MoE 部署贵。Switch 证明可以把稀疏大模型**蒸馏**进一个小稠密模型,保留约 30% 的「稀疏增益」——这是把 MoE 训练的好处搬到便宜部署上的常见思路。

**GShard 的容量因子细节**:它把 token 按位置分到专家 buffer,溢出走残差;并提出 **expert capacity = $\frac{2N}{E}$**(top-2 下)这类经验设置。GShard 用这套在 2048 个 TPU 上训了 600B 参数的多语言翻译模型(覆盖 100 种语言),是首个真正「巨型 + 能跑」的 Transformer MoE。

![[moe-门控topk.png]]

## ④ 代码:top-1(Switch) vs top-2(GShard) 路由对比

```python
import torch, torch.nn.functional as F

def switch_top1(x, w_g, experts):                    # Switch:每 token 1 个专家
    g = F.softmax(x @ w_g.T, dim=-1)                  # (T, N)
    val, idx = g.max(-1)                              # top-1:值 val 当缩放系数
    y = torch.zeros_like(x)
    for e in idx.unique():
        m = idx == e
        y[m] = val[m, None] * experts[e](x[m])        # 直接用未归一化门控分
    return y

def gshard_top2(x, w_g, experts):                    # GShard:每 token 2 个专家
    g = F.softmax(x @ w_g.T, dim=-1)
    val, idx = g.topk(2, dim=-1)                      # (T, 2)
    val = val / val.sum(-1, keepdim=True)             # 两专家门控分重归一化
    y = torch.zeros_like(x)
    for slot in range(2):
        for e in idx[:, slot].unique():
            m = idx[:, slot] == e
            y[m] += val[m, slot, None] * experts[e](x[m])
    return y

# ❌ 误区:以为 top-1 一定比 top-2 差很多
#    → Switch 实验证明算力相同下 top-1 质量不降,且通信/实现减半(更快)
# ✅ 取舍:追极致吞吐/省通信 → top-1;要更稳的质量 → top-2(后来事实标准)
```

## 面试高频

- **Q:GShard 和 Switch 的核心区别?** 路由的 $k$:GShard top-2、Switch top-1。Switch 证明每 token 只走 1 个专家也能保质,且通信/实现减半、提速最高 7×。
- **Q:为什么 top-1 能省这么多?** MoE 瓶颈常是跨卡 all-to-all 通信;top-1 把每 token 发的份数从 2 减到 1,通信直接砍半。
- **Q:Switch 把专家放到了 Transformer 哪一层?** 替换 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]] 子层(注意力不动);通常隔层替换(每隔一个 block 才放 MoE FFN)。
- **Q:Switch 为什么门控用 fp32?** bf16 下路由 logits 的 softmax 数值易不稳,selective precision 只在门控处用 fp32 稳住路由,不增加多少成本(这一思路后来发展为 [[048 路由稳定性：router z-loss|router z-loss]])。
- **Q:Switch-C 多大?** 专家数 2048、总参 1.6T,但每 token 激活算力约等于一个稠密 FFN——[[042 MoE 动机：稀疏激活与容量解耦|容量/算力解耦]]的极端例子。
- **Q:GShard 的工程贡献?** SPMD 自动分片 + all-to-all 路由通信,是 [[049 专家并行 EP 与 MoE 部署|专家并行]] 的源头。
- **Q:Switch 的「7× 提速」是什么意思?** 同**算力预算**下达到同等 loss 快约 7 倍(靠稀疏带来的更大有效容量收敛更快),**不是**单次前向快 7 倍,也不是同参数量。
- **Q:MoE 微调容易过拟合怎么办?** Switch 用「专家层大 dropout、其余小 dropout」的差异化策略;也可把稀疏模型蒸馏回小稠密模型省部署。
- **Q:能把 MoE 变回稠密吗?** 能,蒸馏:用 MoE 当老师训一个稠密学生,保留部分稀疏增益,换取便宜部署。
- **Q:GShard 训了多大?** 在 2048 个 TPU 上训 600B 参数、100 语言的翻译模型,首个「巨型且能高效跑」的 Transformer MoE。

## 关键事实

- **GShard**:Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*,2020,arXiv:2006.16668。**top-2** 路由、容量因子、辅助负载均衡损失、SPMD 分片;用于 6000 亿参数多语言翻译模型。
- **Switch Transformer**:Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*,2021,arXiv:2101.03961。**top-1** 路由;基于 T5-Base/Large,预训练提速最高 **7×**;Switch-C 达 **1.6T 参数 / 2048 专家**。
- 两者都把 MoE 接在 Transformer 的 FFN 子层,注意力不变,常**隔层**放置 MoE 层。
- Switch 的 selective precision(门控用 fp32)与小初始化,是为路由数值稳定;后由 [[048 路由稳定性：router z-loss|router z-loss]](ST-MoE, Zoph 2022)系统化。
- Switch 的「7×」是同算力预算下的收敛提速(非单次前向快、非同参数量);并提出专家差异化 dropout 缓解微调过拟合、可蒸馏回稠密模型。
- GShard 在 2048 TPU 上训 600B、100 语言翻译模型;top-2 容量经验设置如 expert capacity ≈ $2N/E$。
- 后续 [[046 Mixtral 稀疏 MoE|Mixtral]](2024)回到 top-2,成为开源 MoE 的事实标准配置;GLaM(Du et al., 2021)也是 top-2 的大规模 MoE 代表。
