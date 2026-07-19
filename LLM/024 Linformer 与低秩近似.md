[[024 Linformer 与低秩近似|Linformer 与低秩近似]]:以“注意力可作低秩近似”为假设，用投影矩阵把 K、V 的长度维从 $n$ 压到 $k$，使固定 $k$ 下的注意力主项变为 $O(nk)$。这是近似方法；$k$、序列长度与是否因果都必须随任务验证，不能把它当作任意模型的无损线性替代。

## 直觉:注意力矩阵其实"信息不满"
标准注意力先算 $A=\text{softmax}(QK^\top)$,这是个 $n\times n$ 的大矩阵——瓶颈就在它(显存与算力都 $\propto n^2$,见 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²) 与瓶颈]])。

Linformer 的出发点是：在其假设和实验中，注意力矩阵可被低秩近似。于是可用学得的投影矩阵 $E$ 把 $n$ 个 key 压成 $k$ 个“摘要 key”，再做注意力。这里 $k$ 是**需验证的近似秩/工程超参**；固定它才得到相对 $n$ 的线性主项，并不意味着所有数据分布、长度和任务都可用同一个 $k$。

类比:开会有 $n=500$ 人发言,但观点其实就 $k=20$ 类。与其让每个人都和所有人一一对话($n^2$),不如先把 $500$ 条发言归并成 $20$ 条"代表观点",每个人只跟这 $20$ 条对齐——工作量从 $n^2$ 掉到 $nk$。

## 例子:小数字看压缩
设 $n=8192$、$d=64$、投影维 $k=256$。
- **标准**:$QK^\top$ 形状 $n\times n=8192\times8192\approx6.7\times10^7$ 个注意力值,算力 $\propto n^2 d\approx3.5\times10^{10}$。
- **Linformer**:先 $E K$ 把 $K$ 从 $8192\times64$ 压成 $256\times64$;$Q(EK)^\top$ 形状 $n\times k=8192\times256\approx2.1\times10^6$,算力 $\propto n k d\approx1.3\times10^9$。

注意力矩阵元素数少了 **32 倍**($n/k=32$),算力同量级下降,且**最大中间矩阵只有 $n\times k$ 不再是 $n\times n$**。$n$ 越长,省得越多——这正是线性复杂度的含义。

**“低秩”在 SVD 上的样子。** 把注意力矩阵 $A=\text{softmax}(QK^\top)\in\mathbb{R}^{n\times n}$ 做 SVD:$A=\sum_{r=1}^{n}\sigma_r u_r v_r^\top$。若目标数据和层的谱在前 $k$ 项后快速衰减，用 rank-$k$ 近似才可能保留主要能量；若谱不衰减，压缩就会损失关键信息。因而应在目标 checkpoint/任务上测谱、下游质量与长度外推，不能把“前 128 项已足够”写成普适事实。

## 原理:在长度维度上做低秩投影
标准多头注意力(略去 $\sqrt d$ 与多头下标):
$$\text{Attn}(Q,K,V)=\underbrace{\text{softmax}\!\big(QK^\top\big)}_{n\times n}\,V,\qquad Q,K,V\in\mathbb{R}^{n\times d}$$

Linformer 引入两个投影矩阵 $E,F\in\mathbb{R}^{k\times n}$,**只作用在序列长度维 $n$ 上**:
$$\bar K=EK\in\mathbb{R}^{k\times d},\qquad \bar V=FV\in\mathbb{R}^{k\times d}$$
于是注意力变成
$$\text{Attn}=\underbrace{\text{softmax}\!\big(Q\bar K^\top\big)}_{n\times k}\,\bar V\in\mathbb{R}^{n\times d}$$

关键:$Q\bar K^\top$ 形状是 $n\times k$(瘦长条),不再是 $n\times n$。两步主算力 $Q\bar K^\top$ 与 $P\bar V$ 都是 $O(nkd)$;$k$ 固定 → **整体 $O(n)$**。

**为什么能这么压?** 论文给出了低秩近似的理论与实验论据；工程上仍应把 $k$ 视作精度–成本旋钮。Johnson–Lindenstrauss 引理关于随机投影保距的结论不能直接推出“任意 softmax 注意力只要固定 $k=256/512$ 就无损”。这里最重要的可复现问题是：模型/层/任务/最大长度是什么，$k$ 多大，使用哪一个质量与延迟指标。

**参数共享变体(论文给的三档,省参数)。** $E,F$ 各是 $k\times n$,注意力头多、层多时这些投影矩阵参数不小。Linformer 提供共享方案:
- **headwise**:每个头独立的 $E,F$(最灵活、参数最多)。
- **key-value 共享**:同一层内 $E=F$(K、V 用同一投影),参数减半。
- **layerwise 共享**:所有层共用一套 $E,F$,参数最省。
共享会改变参数容量；其影响应以论文中相应设置或你的目标任务消融为准，不能概括为“影响很小”。

![[attn-Linformer低秩投影.png]]

**和 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]] 的区别**:线性注意力**去掉 softmax**、靠核函数 + 结合律换序;Linformer **保留 softmax**,只是在长度维上做**低秩投影**把 $n$ 换成 $k$。两条都把 $O(n^2)$ 变 $O(n)$,但机制不同——一个改"乘法顺序",一个改"矩阵尺寸"。

**代价与因果边界**:$E,F$ 是固定形状($k\times n$)，该直接参数化要求预先约定最大长度，长度外推需重新设计/训练。更关键的是，若把全序列 K/V 直接投影，早期 query 的摘要会混入未来 token，**不满足 decoder 的因果约束**。下面的线性复杂度代码因此明确只演示双向/编码器注意力；不要给它悄悄加一个普通 attention mask 就当成因果 Linformer。

## 代码:压的就是长度维
```python
import torch
import torch.nn.functional as F

# ❌ 标准:显式 n×n 注意力矩阵,O(n²)
def standard_attn(Q, K, V):                     # (B, n, d)
    A = (Q @ K.transpose(-1, -2)) / Q.size(-1) ** 0.5   # (B, n, n) ← 大矩阵
    return F.softmax(A, dim=-1) @ V

# ✅ 双向 Linformer 教学版：用 E、F 把长度 n 投影到 k，不能用于 causal decoder
class BidirectionalLinformerAttn(torch.nn.Module):
    def __init__(self, n, k, d):
        super().__init__()
        # 投影矩阵作用在"长度维 n"上,不是特征维 d —— 易错点!
        self.E = torch.nn.Parameter(torch.randn(k, n) / n ** 0.5)  # (k, n)
        self.F = torch.nn.Parameter(torch.randn(k, n) / n ** 0.5)  # (k, n)
        self.scale = d ** 0.5

    def forward(self, Q, K, V, *, is_causal=False):  # (B, n, d)
        if is_causal:
            raise ValueError("全序列共享 E/F 会把未来 token 混入摘要；不能直接用于因果解码")
        Kbar = self.E @ K                       # (B, k, d) ← K 长度 n→k
        Vbar = self.F @ V                       # (B, k, d) ← V 长度 n→k
        A = (Q @ Kbar.transpose(-1, -2)) / self.scale   # (B, n, k) 瘦条,不是 n×n!
        return F.softmax(A, dim=-1) @ Vbar      # (B, n, d)
# ❌ 易错:把投影加到特征维 d 上(KbarT=K@E)→ 没动到长度,白压。必须压 n。
# ❌ 易错:给 A 再加下三角 mask 也无法挽回泄漏，因为 Kbar/Vbar 在打分前已混入未来。

# 因果正确但不再保留上述线性预计算的示意：每个 query t 使用自己的前缀投影 E_t/F_t，
# 并强制 E_t[:, s]=F_t[:, s]=0 (s>t)。构造所有 t 的摘要本身需 O(n²kd)，
# 所以这是“先保证不泄漏”的说明，不是可宣称 O(n) 的 decoder 实现。
def causal_prefix_project(E_by_query, K):       # E_by_query:(n,k,n), K:(B,n,d)
    n = K.size(1)
    causal_E = E_by_query * torch.tril(torch.ones(n, n, device=K.device))[..., None, :]
    return torch.einsum("tks,bsd->btkd", causal_E, K)  # t 的摘要只看 s<=t
```

## 面试高频
- **Linformer 凭什么 $O(n)$?** 注意力矩阵近似**低秩**,故用投影矩阵 $E,F\in\mathbb{R}^{k\times n}$ 把 K、V 的长度从 $n$ 压到常数 $k$,注意力矩阵从 $n\times n$ 变 $n\times k$,算力 $O(nkd)=O(n)$。
- **它和 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]] 区别?** 线性注意力去 softmax + 结合律换序;Linformer 保留 softmax,在长度维做低秩投影。一个换乘法顺序,一个换矩阵尺寸。
- **最大缺点?** 投影矩阵形状含 $n$，长度外推不自然；低秩近似可能丢信息。用于 decoder 时，全序列共享投影还会发生未来泄漏，不能仅靠普通下三角 mask 修好。
- **"低秩"到底指什么低秩?** 指 $\text{softmax}(QK^\top)$ 这个注意力矩阵的有效秩远小于 $n$(奇异值快速衰减),不是 Q/K 本身低秩。
- **和 [[022 稀疏注意力(BigBird、块稀疏)|稀疏注意力]] 哪个更激进?** 稀疏是"只算一部分对、仍是 softmax";Linformer 是"把所有 key 压成少数摘要"。一个删边,一个降维。
- **Linformer 能做自回归解码吗?** 朴素的共享 $E/F$ 会把所有位置压进同一摘要，早期 query 因而看到未来；这份实现只能用于双向注意力。因果版本必须对每个 query 限制投影支持集到前缀或采用专门的可因果设计，并重新核算复杂度/缓存策略。
- **为什么常把 $k$ 固定?** 固定 $k$ 才有相对 $n$ 的线性主项，但这是精度–成本假设。应报告模型、最大长度、任务、$k$、质量指标与延迟，而不是以 JL 引理承诺固定 $k$ 对任意长度足够。
- **追问:如何测出未来泄漏?** 构造两条序列，令 $0\ldots t$ 完全相同、$t+1\ldots n$ 不同；真正因果层在 $t$ 的输出必须相同。全序列投影版会因摘要改变而失败该测试。
- **Linformer 的低秩 vs [[020 MLA 多头潜在注意力(DeepSeek)|MLA]] 的低秩区别?** Linformer 压**序列长度维 $n$**(把 $n$ 个 key 摘要成 $k$ 个),目的是省算力;MLA 压**特征/头维度**(把 K/V 联合压成 $d_c$ 维潜向量),目的是省 KV-Cache。同叫"低秩",压的维度不同、目的不同。

## 关键事实
- Sinong Wang、Belinda Z. Li、Madian Khabsa、Han Fang、Hao Ma,*Linformer: Self-Attention with Linear Complexity*,2020,arXiv:2006.04768。论文在其低秩近似设定下以投影降低自注意力时空主项；这不等于所有数据/长度的无损 $O(n)$ 替代。
- 核心式:$\bar K=EK,\ \bar V=FV$,$E,F\in\mathbb{R}^{k\times n}$,注意力 $\text{softmax}(Q\bar K^\top)\bar V$,形状 $n\times k$。
- 理论/实验使用边界:将论文的秩、投影、任务与最大长度作为一张实验卡复现；不要把 Johnson–Lindenstrauss 的一般随机投影结论简化成任意 softmax 注意力的固定-rank 保证。
- 因果边界:此处全序列共享投影公式适用于双向注意力；自回归系统必须先证明前缀独立性，再宣称正确性或复杂度收益。
- 定位:与 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 的另几条破解路线——[[023 线性注意力(Linear Transformer、Performer)|线性注意力]](核近似)、[[022 稀疏注意力(BigBird、块稀疏)|稀疏]](删边)、[[025 FlashAttention(IO 感知精确注意力)|FlashAttention]](精确 IO 优化)——并列;Linformer 走"低秩降维"一路。
