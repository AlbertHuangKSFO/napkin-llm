[[022 稀疏注意力(BigBird、块稀疏)]]:用**窗口 + 全局 + 随机**三种稀疏模式叠出近似全注意力的连通性,把 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 降到 $O(n)$;为了在 GPU 上真加速,稀疏按"块(block)"对齐 → 块稀疏。

## 直觉:稀疏也要"连得通"
[[021 局部与滑窗注意力(Longformer、Mistral SWA)|滑窗]]只看邻居,远处信息要逐层中转,慢且易丢。BigBird 的洞察:**再补两类边,让任意两点几跳就能互达**——

1. **窗口(局部)**:看左右邻居,管局部语法/连贯。
2. **全局**:少数 token 行/列填满(看所有、被所有看),做长程"枢纽"。
3. **随机**:每行再随机连几个远处 token,像"小世界图"的随机捷径,把任意两点的**图距离**压到很小。

三者叠加后,注意力图既稀疏(每行非零数 $O(1)$,总量 $O(n)$),又连通性强(信息少跳即达),逼近全注意力的表达力。

类比:城市交通 = 小区内步行(窗口)+ 几个枢纽机场(全局)+ 随机直飞航线(随机)。光靠步行到处都远,加上枢纽和直飞,全城任意两点都几跳可达。

## 例子:连通性的威力
$n=4096$,全注意力每行 4096 个连接;BigBird 取窗口=3、全局=2、随机=3,每行约 $3+2+3=8$ 个非零 → **每行从 4096 降到约 8**,总连接量 $O(n)$。

随机边的作用(小世界直觉):纯窗口图里两个相距 $n$ 的点要走 $\sim n/W$ 跳;加少量随机边后,期望图距离骤降到 $\sim\log n$ 量级 → 任意两点几乎"两三跳可达",信息几乎不损失就能跨长程传播。这是 BigBird 能逼近全注意力的理论根基。

**图距离手算对比($n=4096$)。** 纯窗口 $w=1$(只看左右各 1):两端相距 4095,要 $\sim 4095/1\approx4095$ 跳才能互达——信息跨整段要经 4000 多次中转,几乎传不过去。加全局枢纽后,任意两点经枢纽 **2 跳**($i\to$ 全局 $\to j$)可达;再加随机边,$\log_2 4096=12$ 量级即覆盖。**从 4095 跳 → 2~12 跳**,这就是"窗口管局部、全局+随机管长程"为何缺一不可:少了枢纽/随机,长程信息要 4000 跳逐层接力,深度根本不够。

## 原理:三模式叠加 + 块稀疏
位置 $i$ 的可见集合是三者并集:
$$\mathcal{N}(i)=\underbrace{\{j:|i-j|\le w\}}_{\text{窗口}}\ \cup\ \underbrace{G}_{\text{全局集}}\ \cup\ \underbrace{R_i}_{\text{随机 }r\text{ 个}}$$
注意力只在 $\mathcal{N}(i)$ 上做 softmax:
$$\text{Attn}(i)=\sum_{j\in\mathcal{N}(i)}\frac{\exp(q_i^\top k_j/\sqrt d)}{\sum_{j'\in\mathcal{N}(i)}\exp(q_i^\top k_{j'}/\sqrt d)}\,v_j$$
每行 $|\mathcal{N}(i)|=2w+|G|+r=O(1)$,总复杂度:
$$O\big(n\cdot(2w+|G|+r)\big)=O(n)$$

**为什么必须"块"稀疏?** GPU 擅长**连续块**的稠密矩阵乘,讨厌零散索引(gather/scatter 慢)。若按单 token 稀疏,理论 FLOPs 省了但实际更慢。于是把序列切成块(如 64×64),三种模式都**对齐到块粒度**——整块算或整块跳,用块稀疏 kernel 批量处理 → 稀疏才换来真实墙钟加速。

**理论保证**:Zaheer 等证明 BigBird 的稀疏注意力是序列函数的**通用近似器**且**图灵完备**(保留了全注意力的这些性质),代价仅 $O(n)$。

**稀疏注意力家族谱(面试爱让你横向比)。** 按"谁决定稀疏图样"分两类:
- **固定/启发式稀疏**:稀疏图样预先写死,与内容无关。
  - **Sparse Transformer**(Child et al. 2019,OpenAI):跨步(strided)+ 局部固定图样,$O(n\sqrt n)$,首个把稀疏做进生成式 Transformer 的工作(图像/音频)。
  - **Longformer**(2020):滑窗 + 空洞 + 任务驱动全局,无随机边,$O(n)$。
  - **BigBird**(2020):窗口 + 全局 + **随机**,$O(n)$,随机边补连通性。
- **内容自适应稀疏**:按内容动态决定谁连谁。
  - **Reformer**(Kitaev et al. 2020):用 **LSH(局部敏感哈希)** 把相近的 Q/K 分到同一桶,只在桶内做注意力 → $O(n\log n)$;相似项大概率同桶,近似全注意力。
  - **Routing Transformer**(Roy et al. 2020):用**在线 k-means** 把 token 聚类,只在同簇内注意 → $O(n^{1.5})$。

固定稀疏实现简单、可预先排块(GPU 友好);自适应稀疏理论上更"对症下药",但哈希/聚类的动态索引对 GPU 不友好、易碎片化。这也是为何最后**精确的 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 反而胜出**——稠密但 IO 优化,省心又不掉点。

![[attn-自适应稀疏.png]]

这条路线降的是 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²) 瓶颈]]的**算力**维度,且比纯[[021 局部与滑窗注意力(Longformer、Mistral SWA)|滑窗]]连通性更强。

![[attn-块稀疏图样.png]]

## 代码:三模式掩码(示意) + 块稀疏要点
```python
import torch, torch.nn.functional as F

# ✅ 构造 BigBird 风格的稀疏掩码(窗口 + 全局 + 随机)
def bigbird_mask(n, w=1, n_global=2, n_random=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(n)[:, None]
    j = torch.arange(n)[None, :]
    window = (i - j).abs() <= w                       # ① 窗口
    glob = torch.zeros(n, n, dtype=torch.bool)
    glob[:n_global, :] = True; glob[:, :n_global] = True   # ② 全局(前 n_global 个)
    rand = torch.zeros(n, n, dtype=torch.bool)
    for r in range(n):                                # ③ 每行随机 n_random 个
        cols = torch.randint(0, n, (n_random,), generator=g)
        rand[r, cols] = True
    mask = window | glob | rand
    return torch.where(mask, 0.0, float("-inf"))

# ❌ 仅有掩码 ≠ 真加速:dense kernel 仍算了全部 n×n,只是把窗外置 -inf
def fake_sparse(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)   # O(n²) 没省!

# ✅ 真加速:把三模式对齐到块,用块稀疏 kernel 只算"非全零的块"
#    - 序列切成 block_size=64 的块
#    - 维护"哪些 (块i,块j) 需要计算"的块级稀疏图
#    - 调 block-sparse attention(如 BigBird 官方实现 / FlexAttention 的 block_mask)
#      → 实际跳过空块,FLOPs 与墙钟都降到 ~O(n)
```

## 面试高频
- **BigBird 三种模式各管什么?** 窗口=局部语法;全局=长程枢纽(看所有/被所有看);随机=小世界捷径(缩短任意两点图距离)。三者缺一,连通性或局部性就有短板。
- **为什么强调"块"稀疏?** GPU 对连续块矩阵乘快、对零散索引慢;按块对齐 + 块稀疏 kernel,稀疏才有真实加速。**只加掩码不换 kernel = $O(n^2)$ 没省**,是高频陷阱题。
- **随机边为什么关键?** 它把图变成"小世界",任意两点期望距离 $\sim\log n$,信息几跳即达 → 这是逼近全注意力的连通性来源,也是 BigBird 强于纯滑窗之处。
- **BigBird 的理论卖点?** 稀疏注意力仍是序列函数的**通用近似器 + 图灵完备**,复杂度 $O(n)$——稀疏没牺牲表达上界。
- **稀疏注意力为何没成主流?** 实现复杂、随机性影响稳定性;而 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 让**精确**全注意力也够快、不掉点,多数场景更省心。稀疏更多用于超长文档专用模型。
- **它和 [[021 局部与滑窗注意力(Longformer、Mistral SWA)|滑窗]]、[[019 GQA 分组查询注意力|GQA]] 关系?** 都降 [[014 注意力复杂度 O(n²) 与瓶颈|O(n²)]] 但维度不同:稀疏/滑窗降序列维算力,GQA 降 KV 头数带宽,可叠加。
- **固定稀疏 vs 内容自适应稀疏区别?** 固定(Sparse Transformer/Longformer/BigBird)图样预先写死、GPU 友好、可块稀疏;自适应(Reformer LSH $O(n\log n)$、Routing Transformer k-means $O(n^{1.5})$)按内容分桶/聚类、更对症但动态索引对 GPU 不友好。
- **复杂度数字背一下?** Sparse Transformer $O(n\sqrt n)$;Longformer/BigBird $O(n)$;Reformer $O(n\log n)$;Routing $O(n^{1.5})$;全注意力 $O(n^2)$。
- **2024 年稀疏注意力又火了?** 是。DeepSeek-V3.2 用细粒度稀疏注意力(NSA / lightning indexer 思路)做超长上下文,结合现代 kernel 让块稀疏在工业级模型上重新划算——稀疏并未消失,而是和 FlashAttention 式 kernel 融合后回归。

## 关键事实
- 出处:Zaheer et al.,*Big Bird: Transformers for Longer Sequences*,2020,arXiv:2007.14062(NeurIPS 2020)。
- 机制:**窗口 + 全局 + 随机**三种稀疏模式叠加;复杂度从 $O(n^2)$ 降到 $O(n)$。
- 理论:BigBird 的稀疏注意力是序列函数的**通用近似器**且**图灵完备**,保留全注意力性质。
- 工程:必须**块稀疏(block-sparse)**——按块对齐三模式、用块稀疏 kernel 才有真实加速(对应稠密 GPU 友好)。
- 谱系:Sparse Transformer(Child et al. 2019,arXiv:1904.10509,$O(n\sqrt n)$)是更早的固定稀疏(跨步 + 局部);BigBird 加了随机边。与[[021 局部与滑窗注意力(Longformer、Mistral SWA)|Longformer]]同属稀疏家族(Longformer 无随机边)。内容自适应一支:Reformer(Kitaev et al. 2020,arXiv:2001.04451,LSH,$O(n\log n)$)、Routing Transformer(Roy et al. 2020,k-means,$O(n^{1.5})$)。
- 与邻接概念:精确替代见 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]];彻底线性化见 [[023 线性注意力(Linear Transformer、Performer)|线性注意力]]与 [[027 状态空间模型与 Mamba|Mamba]]。
