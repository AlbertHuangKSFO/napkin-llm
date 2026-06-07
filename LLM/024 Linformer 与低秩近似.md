[[024 Linformer 与低秩近似|Linformer 与低秩近似]]:发现注意力矩阵近似**低秩**,于是用两个投影矩阵把 K、V 的序列长度从 $n$ 压成一个小常数 $k$,把 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 降到 $O(nk)=O(n)$。

## 直觉:注意力矩阵其实"信息不满"
标准注意力先算 $A=\text{softmax}(QK^\top)$,这是个 $n\times n$ 的大矩阵——瓶颈就在它(显存与算力都 $\propto n^2$,见 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²) 与瓶颈]])。

Linformer 的观察:把这个 $n\times n$ 矩阵做奇异值分解,**能量集中在前几个奇异值**,后面的几乎为零——也就是它**近似低秩**。既然 $n$ 个 key 携带的"有效信息"只有大约 $k$ 维,那不如先用一个学好的投影矩阵 $E$ 把 $n$ 个 key**压成 $k$ 个"摘要 key"**,再做注意力。$k$ 是常数(论文里 256 这种),与 $n$ 无关。

类比:开会有 $n=500$ 人发言,但观点其实就 $k=20$ 类。与其让每个人都和所有人一一对话($n^2$),不如先把 $500$ 条发言归并成 $20$ 条"代表观点",每个人只跟这 $20$ 条对齐——工作量从 $n^2$ 掉到 $nk$。

## 例子:小数字看压缩
设 $n=8192$、$d=64$、投影维 $k=256$。
- **标准**:$QK^\top$ 形状 $n\times n=8192\times8192\approx6.7\times10^7$ 个注意力值,算力 $\propto n^2 d\approx3.5\times10^{10}$。
- **Linformer**:先 $E K$ 把 $K$ 从 $8192\times64$ 压成 $256\times64$;$Q(EK)^\top$ 形状 $n\times k=8192\times256\approx2.1\times10^6$,算力 $\propto n k d\approx1.3\times10^9$。

注意力矩阵元素数少了 **32 倍**($n/k=32$),算力同量级下降,且**最大中间矩阵只有 $n\times k$ 不再是 $n\times n$**。$n$ 越长,省得越多——这正是线性复杂度的含义。

**"低秩"在 SVD 上的样子(为什么压成 $k$ 不丢信息)。** 把注意力矩阵 $A=\text{softmax}(QK^\top)\in\mathbb{R}^{n\times n}$ 做 SVD:$A=\sum_{r=1}^{n}\sigma_r u_r v_r^\top$,奇异值 $\sigma_1\ge\sigma_2\ge\dots$。Linformer 论文实测:在预训练模型上,**前约 128 个奇异值已占了绝大部分能量(谱快速衰减)**,后面 $\sigma_r\approx0$——意味着 $A$ 的"有效秩"远小于 $n$。既然 $A$ 本质只有约 $k$ 维信息,用 $k\times n$ 的投影 $E$ 把 key 压成 $k$ 个,**丢掉的主要是 $\sigma$ 接近 0 的噪声方向**,信息损失很小。这就是它敢把 $n$ 换成常数 $k$ 的依据。

## 原理:在长度维度上做低秩投影
标准多头注意力(略去 $\sqrt d$ 与多头下标):
$$\text{Attn}(Q,K,V)=\underbrace{\text{softmax}\!\big(QK^\top\big)}_{n\times n}\,V,\qquad Q,K,V\in\mathbb{R}^{n\times d}$$

Linformer 引入两个投影矩阵 $E,F\in\mathbb{R}^{k\times n}$,**只作用在序列长度维 $n$ 上**:
$$\bar K=EK\in\mathbb{R}^{k\times d},\qquad \bar V=FV\in\mathbb{R}^{k\times d}$$
于是注意力变成
$$\text{Attn}=\underbrace{\text{softmax}\!\big(Q\bar K^\top\big)}_{n\times k}\,\bar V\in\mathbb{R}^{n\times d}$$

关键:$Q\bar K^\top$ 形状是 $n\times k$(瘦长条),不再是 $n\times n$。两步主算力 $Q\bar K^\top$ 与 $P\bar V$ 都是 $O(nkd)$;$k$ 固定 → **整体 $O(n)$**。

**为什么能这么压?** 论文用 Johnson–Lindenstrauss 引理证明:注意力矩阵的低秩结构使得用一个 $O(\log n / \varepsilon^2)$ 维的随机投影即可以 $\varepsilon$ 误差近似——实践中固定的小 $k$ 就够。直觉上,$\text{softmax}(QK^\top)$ 的有效秩远小于 $n$,投影"丢掉的"主要是噪声方向。

**JL 引理一句话直觉(为什么 $k$ 不随 $n$ 涨)。** Johnson–Lindenstrauss 说:$n$ 个高维点,可被一个**只依赖精度 $\varepsilon$、与 $n$ 仅对数相关**的低维($O(\log n/\varepsilon^2)$)随机投影保距嵌入。对数增长极慢——$n$ 从 $10^3$ 涨到 $10^6$,$\log n$ 才翻一倍。所以工程上直接把 $k$ 固定成 256/512 这种常数就够覆盖很长序列,投影维**几乎不随 $n$ 增长** → 复杂度退化成 $O(n)$。

**参数共享变体(论文给的三档,省参数)。** $E,F$ 各是 $k\times n$,注意力头多、层多时这些投影矩阵参数不小。Linformer 提供共享方案:
- **headwise**:每个头独立的 $E,F$(最灵活、参数最多)。
- **key-value 共享**:同一层内 $E=F$(K、V 用同一投影),参数减半。
- **layerwise 共享**:所有层共用一套 $E,F$,参数最省。
实测共享对质量影响很小,常用 key-value 共享或更激进的 layerwise 来压参数。

![[attn-Linformer低秩投影.png]]

**和 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]] 的区别**:线性注意力**去掉 softmax**、靠核函数 + 结合律换序;Linformer **保留 softmax**,只是在长度维上做**低秩投影**把 $n$ 换成 $k$。两条都把 $O(n^2)$ 变 $O(n)$,但机制不同——一个改"乘法顺序",一个改"矩阵尺寸"。

**代价**:$E,F$ 是固定形状($k\times n$),意味着**序列长度 $n$ 必须预先定死**。换长度就得换投影矩阵,无法像 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]] 那样任意外推;且对需要精确逐 token 检索的任务,低秩压缩会掉信息。

## 代码:压的就是长度维
```python
import torch
import torch.nn.functional as F

# ❌ 标准:显式 n×n 注意力矩阵,O(n²)
def standard_attn(Q, K, V):                     # (B, n, d)
    A = (Q @ K.transpose(-1, -2)) / Q.size(-1) ** 0.5   # (B, n, n) ← 大矩阵
    return F.softmax(A, dim=-1) @ V

# ✅ Linformer:用 E、F 把长度 n 投影到 k,O(n·k)
class LinformerAttn(torch.nn.Module):
    def __init__(self, n, k, d):
        super().__init__()
        # 投影矩阵作用在"长度维 n"上,不是特征维 d —— 易错点!
        self.E = torch.nn.Parameter(torch.randn(k, n) / n ** 0.5)  # (k, n)
        self.F = torch.nn.Parameter(torch.randn(k, n) / n ** 0.5)  # (k, n)
        self.scale = d ** 0.5

    def forward(self, Q, K, V):                 # (B, n, d)
        Kbar = self.E @ K                       # (B, k, d) ← K 长度 n→k
        Vbar = self.F @ V                       # (B, k, d) ← V 长度 n→k
        A = (Q @ Kbar.transpose(-1, -2)) / self.scale   # (B, n, k) 瘦条,不是 n×n!
        return F.softmax(A, dim=-1) @ Vbar      # (B, n, d)
# ❌ 易错:把投影加到特征维 d 上(KbarT=K@E)→ 没动到长度,白压。必须压 n。
# ❌ 易错:n 一旦在 __init__ 定死,推理时序列变长就用不了 → 需重训或截断。
```

## 面试高频
- **Linformer 凭什么 $O(n)$?** 注意力矩阵近似**低秩**,故用投影矩阵 $E,F\in\mathbb{R}^{k\times n}$ 把 K、V 的长度从 $n$ 压到常数 $k$,注意力矩阵从 $n\times n$ 变 $n\times k$,算力 $O(nkd)=O(n)$。
- **它和 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]] 区别?** 线性注意力去 softmax + 结合律换序;Linformer 保留 softmax,在长度维做低秩投影。一个换乘法顺序,一个换矩阵尺寸。
- **最大缺点?** 投影矩阵形状含 $n$,**序列长度必须预先固定**,无法外推;低秩压缩对精确长程检索类任务掉点。
- **"低秩"到底指什么低秩?** 指 $\text{softmax}(QK^\top)$ 这个注意力矩阵的有效秩远小于 $n$(奇异值快速衰减),不是 Q/K 本身低秩。
- **和 [[022 稀疏注意力(BigBird、块稀疏)|稀疏注意力]] 哪个更激进?** 稀疏是"只算一部分对、仍是 softmax";Linformer 是"把所有 key 压成少数摘要"。一个删边,一个降维。
- **Linformer 能做自回归解码吗?** 很别扭。投影 $E$ 把全部 $n$ 个 key 混成 $k$ 个摘要,**天然是双向的**(摘要里混进了未来 token),难以满足因果"只看过去";所以 Linformer 主要用在**编码器/双向**场景(如 RoBERTa 替换),而非 GPT 式 decoder。这是它没在生成式大模型里普及的关键原因之一。
- **为什么 $k$ 不随 $n$ 涨?** JL 引理:保距投影维只 $\propto\log n/\varepsilon^2$,对数增长极慢,实践固定 $k=256/512$ 即可覆盖很长序列 → $O(n)$。
- **Linformer 的低秩 vs [[020 MLA 多头潜在注意力(DeepSeek)|MLA]] 的低秩区别?** Linformer 压**序列长度维 $n$**(把 $n$ 个 key 摘要成 $k$ 个),目的是省算力;MLA 压**特征/头维度**(把 K/V 联合压成 $d_c$ 维潜向量),目的是省 KV-Cache。同叫"低秩",压的维度不同、目的不同。

## 关键事实
- Sinong Wang、Belinda Z. Li、Madian Khabsa、Han Fang、Hao Ma,*Linformer: Self-Attention with Linear Complexity*,2020,arXiv:2006.04768。证明自注意力矩阵可被低秩矩阵近似,用投影把时空复杂度从 $O(n^2)$ 降到 $O(n)$。
- 核心式:$\bar K=EK,\ \bar V=FV$,$E,F\in\mathbb{R}^{k\times n}$,注意力 $\text{softmax}(Q\bar K^\top)\bar V$,形状 $n\times k$。
- 理论依据:Johnson–Lindenstrauss 引理 → 低秩注意力可用 $O(\log n)$ 量级的投影维近似。
- 实验:在 RoBERTa 规模上,$k=256$ 即可与标准 Transformer 持平,长序列显著省时省显存。
- 定位:与 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 的另几条破解路线——[[023 线性注意力(Linear Transformer、Performer)|线性注意力]](核近似)、[[022 稀疏注意力(BigBird、块稀疏)|稀疏]](删边)、[[025 FlashAttention(IO 感知精确注意力)|FlashAttention]](精确 IO 优化)——并列;Linformer 走"低秩降维"一路。
