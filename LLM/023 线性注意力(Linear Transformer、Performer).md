[[023 线性注意力(Linear Transformer、Performer)]]:去掉 softmax、用**核函数**近似相似度,再靠矩阵乘**结合律**把 $(QK^\top)V$ 换成 $Q(K^\top V)$,让全注意力的 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 真正变成 $O(n)$。

## 直觉:换个乘法顺序就省一个量级
标准注意力先算 $A=QK^\top$,这是个 $n\times n$ 大矩阵(显存/算力 $\propto n^2$),再 softmax、再 $\times V$。瓶颈就在那个 $n\times n$。

如果**没有 softmax**,$\text{out}=(QK^\top)V$ 是三个矩阵连乘,可以用结合律换序:
$$(QK^\top)V \;=\; Q(K^\top V)$$
右边先算 $K^\top V$,得到一个 $d\times d$ 的**小矩阵**(与 $n$ 无关!),再左乘 $Q$。**全程不出现 $n\times n$**,两步都是 $O(n\cdot d^2)$ → 线性。

问题:softmax 是非线性归一,卡住了换序。线性注意力的招:**用核 $\varphi$ 近似** $\exp(q^\top k)\approx\varphi(q)^\top\varphi(k)$,把指数核拆成可分离的内积,于是又能换序了。

类比:$100$ 个人各自要把同一笔账(对所有 $j$ 求和)算一遍——与其每人重算,不如**先把账求和好(一次 $K^\top V$)**,每人直接取用。求和提到外面,这就是结合律省下的重复。

## 例子:小数字看省了多少
设 $n=10000$、$d=64$。
- **标准**:$QK^\top$ 要 $n^2 d = 10000^2\times64 \approx 6.4\times10^{9}$ 次乘,且存 $n\times n=10^8$ 个注意力值。
- **线性**:$K^\top V$ 要 $n\,d^2=10000\times64^2\approx4.1\times10^7$;再 $Q\times S$ 同量级。总量约 **$8\times10^7$**,比标准少**约 80 倍**;且最大中间矩阵只有 $d\times d=4096$ 个数(不再有 $10^8$)。

$n$ 越长,差距越大($\propto n/d^2$ 量级)——这正是为什么超长序列上线性注意力可比标准 Transformer 快上千倍。

**结合律换序的"小矩阵"手算(看清省在哪)。** 取极小例子 $n=3, d=2$,只看 $(QK^\top)V$ vs $Q(K^\top V)$ 两种算法的中间矩阵尺寸:
- 左序 $(QK^\top)V$:先 $QK^\top$ 得 $3\times3$ 矩阵(9 个数,正比 $n^2$),再 $\times V$。
- 右序 $Q(K^\top V)$:先 $K^\top V$ 得 $2\times2$ 矩阵(4 个数,正比 $d^2$,**与 $n$ 无关**),再 $Q\times$ 它。

具体数:设 $K^\top V=\begin{bmatrix}s_{11}&s_{12}\\ s_{21}&s_{22}\end{bmatrix}$,其中 $s_{ab}=\sum_{j=1}^{3}k_{j,a}v_{j,b}$——**对 $j$ 的求和(遍历所有 token)只做一次就压成 $2\times2$**,之后每个 query $q_i$ 直接乘这个固定小矩阵。把 $n=3$ 换成 $n=10^6$,$K^\top V$ 仍是 $2\times2$;左序却要 $10^6\times10^6$。这就是结合律"把对 $j$ 的求和提到外面、压成 $d\times d$ 常数矩阵"的全部威力。

## 原理:核近似 + 结合律 + RNN 形式
标准注意力第 $i$ 个输出(略去 $\sqrt d$):
$$o_i=\sum_{j=1}^{n}\frac{\exp(q_i^\top k_j)}{\sum_{j'}\exp(q_i^\top k_{j'})}\,v_j$$
softmax 让分子分母都含 $\exp(q_i^\top k_j)$,无法把对 $j$ 的求和提出去。

**第一步:核近似。** 找特征映射 $\varphi$ 使 $\exp(q^\top k)\approx\varphi(q)^\top\varphi(k)$。代入:
$$o_i=\frac{\sum_j \big(\varphi(q_i)^\top\varphi(k_j)\big)v_j}{\sum_j \varphi(q_i)^\top\varphi(k_j)}
=\frac{\varphi(q_i)^\top\Big(\sum_j \varphi(k_j)\,v_j^\top\Big)}{\varphi(q_i)^\top\Big(\sum_j \varphi(k_j)\Big)}$$

**第二步:结合律换序。** 关键在把对 $j$ 的求和**提到 $\varphi(q_i)$ 外面**。令
$$S=\sum_j \varphi(k_j)\,v_j^\top\in\mathbb{R}^{d\times d},\qquad z=\sum_j \varphi(k_j)\in\mathbb{R}^{d}$$
$S,z$ 与 $i$ 无关,**只算一次**。则
$$o_i=\frac{\varphi(q_i)^\top S}{\varphi(q_i)^\top z}$$
所有 $i$ 共用 $S,z$ → 总复杂度 $O(n\,d^2)$,线性。

**第三步:RNN 形式(因果版)。** 自回归时 $i$ 只能看 $j\le i$,把 $S,z$ 改成**前缀累加**:
$$S_t=S_{t-1}+\varphi(k_t)v_t^\top,\qquad z_t=z_{t-1}+\varphi(k_t)$$
$S_t$ 就是一个 $d\times d$ 的"隐状态",每步 $O(d^2)$ 更新、$O(1)$ 不依赖序列长 → **解码像 RNN**:常数显存、常数时间/步。这就是论文标题"Transformers are RNNs"的含义。

![[attn-线性注意力重排.svg]]

**两种 $\varphi$ 路线:**
- **Linear Transformer**(Katharopoulos 2020):$\varphi(x)=\text{elu}(x)+1$,简单、恒正(保证分母正)。不追求逼近真 softmax,只要可分离即可。
- **Performer**(Choromanski 2020,FAVOR+):用**随机正交特征**构造 $\varphi$,**无偏估计真正的 softmax 核**,有精度保证;随机特征数 $m$ 越大越准。可直接近似现成 Transformer。

**为什么 $\varphi$ 必须恒正?** 分母 $\sum_j\varphi(q_i)^\top\varphi(k_j)$ 要当归一化用,必须为正(否则除以负数/0 数值爆炸);$\text{elu}+1\ge0$、Performer 的 $\exp$ 特征也恒正,正是为此。若直接用线性核 $\varphi(x)=x$,内积可负 → 注意力权重出现负数、归一化失效,这是初学者最常踩的坑。

**现代演进:门控与增删(为什么线性注意力又回来了)。** 朴素线性注意力的状态 $S_t$ 只增不减(一直累加),**记忆永不遗忘 → 容量很快被旧信息塞满、长程退化**。新一代在递推里加"门控/删除":
- **门控线性注意力 GLA**(Yang et al. 2023):递推改成 $S_t=G_t\odot S_{t-1}+\varphi(k_t)v_t^\top$,$G_t$ 是数据相关的**遗忘门**,让状态能按内容衰减旧信息(与 [[027 状态空间模型与 Mamba|Mamba]] 的选择性同思想)。
- **DeltaNet / Gated DeltaNet**(Yang et al. 2024):用 delta rule(类似在线最小二乘/快速权重)做**精确的记忆增删**,能"覆盖写"而非只叠加,长程检索更强。
- **RetNet**(Sun et al. 2023,arXiv:2307.08621):retention 机制 = 带固定指数衰减的线性注意力,同时具备并行/递推/分块三种等价形式。
- **RWKV**(Peng et al.,持续迭代到 RWKV-6/7):线性注意力 + 时间衰减的 RNN 化,纯 token-shift + WKV 递推。

这些都属"门控线性注意力(gated linear attention)"大家庭,统一视角:**可并行训练 + 类 RNN 推理 + 数据相关门控**——把表达力做强,缩小与 softmax 注意力的差距。

这条路降的是 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²) 瓶颈]]里的**算力与显存**两维,代价是放弃了精确 softmax。

## 代码:换序就是全部秘密
```python
import torch
import torch.nn.functional as F

# ❌ 标准:显式 n×n 注意力矩阵,O(n²)
def softmax_attn(q, k, v):                     # (B,h,n,d)
    A = (q @ k.transpose(-1, -2)) / q.size(-1) ** 0.5   # (B,h,n,n) ← 大矩阵
    return F.softmax(A, dim=-1) @ v

# ✅ 线性(非因果):先算 KᵀV (d×d 小矩阵),再左乘 Q → O(n·d²)
def linear_attn(q, k, v, eps=1e-6):
    phi = lambda x: F.elu(x) + 1               # 恒正特征映射
    qf, kf = phi(q), phi(k)                    # (B,h,n,d)
    S = kf.transpose(-1, -2) @ v               # (B,h,d,d) ← 与 n 无关!关键一步
    z = kf.sum(dim=-2, keepdim=True)           # (B,h,1,d) 归一化项
    num = qf @ S                               # (B,h,n,d)
    den = (qf @ z.transpose(-1, -2)) + eps     # (B,h,n,1)
    return num / den

# ✅ 因果(自回归)版:把 S 当 RNN 隐状态逐步累加,解码 O(1)/步、常数显存
def linear_attn_causal_step(q_t, k_t, v_t, state, eps=1e-6):
    phi = lambda x: F.elu(x) + 1
    qf, kf = phi(q_t), phi(k_t)                # (B,h,d)
    S, z = state                              # S:(B,h,d,d)  z:(B,h,d)
    S = S + kf.unsqueeze(-1) * v_t.unsqueeze(-2)   # Sₜ = Sₜ₋₁ + φ(kₜ)vₜᵀ
    z = z + kf                                 # zₜ = zₜ₋₁ + φ(kₜ)
    out = (qf.unsqueeze(-2) @ S).squeeze(-2) / ((qf * z).sum(-1, keepdim=True) + eps)
    return out, (S, z)
# ❌ 易错:因果版不能直接用非因果的全局 S,会让 token 看到未来 → 必须前缀累加
```

## 面试高频
- **线性注意力凭什么 $O(n)$?** 去掉 softmax 后用结合律:$(QK^\top)V\to Q(K^\top V)$,先算 $d\times d$ 的 $K^\top V$(与 $n$ 无关),避开 $n\times n$。**softmax 是唯一拦路虎**,故用核 $\varphi$ 近似。
- **"Transformers are RNNs" 啥意思?** 因果线性注意力的 $S_t=S_{t-1}+\varphi(k_t)v_t^\top$ 就是 RNN 的隐状态递推 → 解码常数显存/时间,等价一个线性 RNN。
- **Linear Transformer vs Performer?** 前者 $\varphi=\text{elu}+1$ 简单、不保证逼近 softmax;后者 FAVOR+ 用随机特征**无偏估计 softmax**、有精度保证,可改造现成模型。
- **为什么实践中没取代标准注意力?** ① 核近似掉点,长程精确检索类任务弱;② $d\times d$ 状态记忆容量有限;③ [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 让**精确**全注意力也够快,多数场景不必近似。
- **它和 [[021 局部与滑窗注意力(Longformer、Mistral SWA)|稀疏注意力]]区别?** 稀疏是"少算一些对"(仍是 softmax);线性是"换核 + 换序"(放弃 softmax)。两条不同路线。
- **遗产在哪?** "注意力即线性 RNN"的思想延续到 [[027 状态空间模型与 Mamba|Mamba]]、RWKV、RetNet —— 都在用"可并行训练 + 类 RNN 推理"的状态递推。
- **$\varphi$ 为什么必须恒正?** 分母是归一化项,必须为正;$\text{elu}+1$、Performer 的 $\exp$ 特征都恒正。用普通线性核会出现负注意力权重、归一化爆炸。
- **朴素线性注意力的最大短板?** 状态 $S_t$ 只增不减、无遗忘 → 固定 $d\times d$ 容量被旧信息塞满,长程精确检索(大海捞针)弱。GLA/DeltaNet 加门控/删除来缓解。
- **GLA、RetNet、RWKV、Mamba 是一类吗?** 是同一"现代线性 RNN / 门控线性注意力"大家庭:可并行训练 + 递推推理 + 数据相关门控;区别在门控/状态更新规则与是否走 SSM 离散化(Mamba)。
- **线性注意力的状态显存为何"常数"?** 因果版只存 $S_t\in\mathbb{R}^{d\times d}$ 和 $z_t\in\mathbb{R}^d$,与序列长 $n$ 无关 → decode 显存恒定,而 softmax 注意力的 KV-Cache $\propto n$。

## 关键事实
- Linear Transformer:Katharopoulos、Vyas、Pappas、Fleuret,*Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*,2020,arXiv:2006.16236(ICML 2020)。用核特征 + 结合律把 $O(n^2)$ 降到 $O(n)$,自回归预测最高快约 **4000×**。
- Performer:Choromanski et al.,*Rethinking Attention with Performers*,2020,arXiv:2009.14794(ICLR 2021)。**FAVOR+**(Fast Attention Via positive Orthogonal Random features)用随机正交特征**无偏/近无偏估计 softmax 核**,线性时空、有理论保证。
- 核心恒等式:$(QK^\top)V=Q(K^\top V)$;近似前提 $\exp(q^\top k)\approx\varphi(q)^\top\varphi(k)$。
- 因果递推:$S_t=S_{t-1}+\varphi(k_t)v_t^\top$,$d\times d$ 隐状态,解码 $O(1)$/步。
- 现代家族(年份易错,核对):RetNet(Sun et al. 2023,arXiv:2307.08621);GLA(Yang et al. 2023,arXiv:2312.06635);DeltaNet / Gated DeltaNet(Yang et al. 2024);RWKV(Peng et al.,RWKV-4 2023 → RWKV-5/6 2024 → RWKV-7 2024/2025)。均属门控线性注意力。
- 与邻接概念:同为破 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 的路线之一,与[[021 局部与滑窗注意力(Longformer、Mistral SWA)|滑窗]]/[[022 稀疏注意力(BigBird、块稀疏)|稀疏]](保留 softmax)互为对照;精确 IO 优化路线是 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]];状态递推思想的现代延续是 [[027 状态空间模型与 Mamba|状态空间模型与 Mamba]]。
