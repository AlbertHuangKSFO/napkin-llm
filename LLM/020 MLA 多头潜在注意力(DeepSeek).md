[[020 MLA 多头潜在注意力(DeepSeek)]]:DeepSeek-V2 提出的注意力——把 K/V **联合压成一个低秩潜向量 c**,[[102 KV-Cache|KV-Cache]] 只缓存这个小向量,用时再上投影"解压"回各头,显存省得像 [[018 MQA 多查询注意力|MQA]],质量却接近甚至超过 [[005 多头注意力 Multi-Head|MHA]]。

## 直觉:别存完整 K/V,存它们的"压缩包"
[[019 GQA 分组查询注意力|GQA]] 通过减少 K/V 头数省缓存,但本质还是"存若干份完整 K/V"。MLA 换个思路:**K 和 V 高度相关、且有冗余,可以一起压到一个低维潜空间**。

每个 token 不再缓存 h 份(或 g 份)$d_{head}$ 维的 K/V,而是缓存**一个** $d_c$ 维潜向量 $c$($d_c\ll h\cdot d_{head}$)。需要时,用各头独立的上投影矩阵把 $c$ "解压"成该头的 K、V。

关键区别于 MQA/GQA:**MQA 是真共享一份 K/V(表达力受限);MLA 是共享一个压缩潜向量,但每个头有独立解压器** → 既省缓存,又保住多头的表达力。

类比:不给每人发一整套百科全书(MHA),也不让大家抢一本(MQA),而是发一张高度压缩的"索引卡"(潜向量 c),每人用自己的解码方式(上投影)还原出需要的内容。

## 例子:压缩到一个潜向量
设 MHA 配置 h=128 头、d_head=128,则每 token 缓存 K+V 共 $2\times128\times128=32768$ 维。

MLA 用 $d_c=512$ 的潜向量(DeepSeek-V2 量级):每 token 只缓存约 $d_c + d_{RoPE}$(位置那一小支,见下)维,数量级从 32768 降到几百 → **约 1/50 量级**。

DeepSeek-V2 官方:KV-Cache 相比同规模 MHA **减少约 93.3%**,同时生成吞吐显著提升,且基准质量不降反升。

> 直觉小账:省缓存的核心是把"缓存维度"从 $2hd_{head}$ 换成 $d_c$,只要 $d_c\ll 2hd_{head}$ 就赢。

**横向对比每 token 缓存维度(同一规模)。** 设 h=128、d_head=128。

| 方案 | 每 token 缓存的"等效维度" | 相对 MHA |
|---|---|---|
| MHA | $2hd_{head}=32768$ | 1 |
| GQA(g=8) | $2g\,d_{head}=2048$ | 1/16 |
| MQA | $2d_{head}=256$ | 1/128 |
| MLA | $d_c+d_{RoPE}\approx 512+64=576$ | 约 1/57 |

关键反差:**MQA 缓存维度比 MLA 还小,但 MLA 质量远好于 MQA**——因为 MQA 是"真共享一份 K/V(表达受限)",MLA 是"共享一个压缩潜向量但每头独立解压(表达力保住)"。用业界常引的字节数:DeepSeek-V3 每 token KV 约 **70 KB**,而同档 LLaMA-3.1 405B(GQA)约 **516 KB**——约 93% 的削减,且 MLA 的基准质量优于其 MHA 基线。这是"缓存小 + 质量好"罕见地同时成立的方案。

## 原理:低秩联合压缩 + 解耦 RoPE
**第一步:下投影压缩(只缓存这个)。** 对输入 $h_t$ 投影出潜向量:
$$c^{KV}_t = W^{DKV}\,h_t,\qquad c^{KV}_t\in\mathbb{R}^{d_c},\ d_c\ll d$$
KV-Cache **只存** $c^{KV}_t$。

**第二步:上投影解压。** 各头的 K、V 由 $c$ 还原:
$$k^{C}_{t,i} = W^{UK}_i\,c^{KV}_t,\qquad v_{t,i} = W^{UV}_i\,c^{KV}_t$$
每个头 $i$ 有独立的 $W^{UK}_i,W^{UV}_i$ → 保留多头表达力(这是 MLA ≠ MQA 的关键)。

**第三步:解耦 RoPE(易踩的坑)。** [[031 RoPE 旋转位置编码(推导与实现)|RoPE]] 是对 Q/K 施加位置相关的旋转,**不能压进与位置无关的 $c$**(否则推理时的"吸收"技巧失效)。MLA 因此把 K 拆成两段:
$$k_{t,i} = \big[\,\underbrace{k^{C}_{t,i}}_{\text{从 }c\text{ 解压,无位置}}\ \big\Vert\ \underbrace{k^{R}_{t}}_{\text{带 RoPE,所有头共享}}\,\big]$$
位置那一支 $k^R_t$ 单独走、维度小($d_{RoPE}$),也缓存它。Q 同理拆两段。这就是"**解耦 RoPE**(decoupled RoPE)"。

**第四步:吸收(推理加速的精髓)。** 注意 $q^\top k^C = q^\top W^{UK}_i c$,可把 $W^{UK}_i$ **吸收进** $W_Q$;同理 $W^{UV}_i$ 吸收进输出投影 $W_O$。于是推理时**根本不必真的解压出完整 K/V**,直接在潜空间算 → 省算又省显存,KV-Cache 体量等于潜向量本身。

**吸收的代数推导(逐步看为什么省算)。** 头 $i$ 的非 RoPE 部分打分:
$$q_{t,i}^\top k^{C}_{s,i}=q_{t,i}^\top\big(W^{UK}_i\,c^{KV}_s\big)=\underbrace{\big((W^{UK}_i)^\top q_{t,i}\big)}_{\tilde q_{t,i}}{}^\top\,c^{KV}_s$$
把 $W^{UK}_i$ 提前乘进 query 侧得到 $\tilde q_{t,i}=(W^{UK}_i)^\top q_{t,i}$,于是打分直接是 $\tilde q_{t,i}^\top c^{KV}_s$——**KV 侧只需缓存并读取 $c^{KV}_s\in\mathbb{R}^{d_c}$,从不物化出 $h$ 份 $d_{head}$ 维的 $k^C$**。输出侧同理:$\sum_s\alpha_s v_{s,i}=\sum_s\alpha_s W^{UV}_i c^{KV}_s=W^{UV}_i\big(\sum_s\alpha_s c^{KV}_s\big)$,$W^{UV}_i$ 可并入 $W_O$。**两个上投影都"消失"进相邻权重**,推理时只在 $d_c$ 维潜空间算 → 既不解压、读取量又只 $\propto d_c$。这是 MLA "缓存小到 MQA 级、表达力却是 MHA 级"得以两全的工程机关。

整体上,MLA 在 [[014 注意力复杂度 O(n²) 与瓶颈|瓶颈]]的"显存/带宽"维度做到 MQA 级别,而表达力维度保持 MHA 级别。

![[attn-MLA低秩KV.png]]

## 代码:MLA 前向骨架
```python
import torch
import torch.nn.functional as F

# ✅ MLA 简化版:只缓存潜向量 c(+ 共享的 RoPE 那一小支)
class MLA(torch.nn.Module):
    def __init__(self, d, h, dh, d_c, d_rope):
        super().__init__()
        self.h, self.dh, self.d_rope = h, dh, d_rope
        self.W_dkv = torch.nn.Linear(d, d_c, bias=False)        # 下投影:压成潜向量
        self.W_uk  = torch.nn.Linear(d_c, h * dh, bias=False)   # 上投影:解压 K(每头独立)
        self.W_uv  = torch.nn.Linear(d_c, h * dh, bias=False)   # 上投影:解压 V
        self.W_kr  = torch.nn.Linear(d, d_rope, bias=False)     # 解耦 RoPE 的 K 支(共享)
        self.W_q   = torch.nn.Linear(d, h * (dh + d_rope), bias=False)
        self.W_o   = torch.nn.Linear(h * dh, d, bias=False)

    def forward(self, x, rope):                                  # x:(B,n,d)
        B, n, _ = x.shape
        c_kv = self.W_dkv(x)                                     # (B,n,d_c) ← 只缓存它
        k_rope = rope(self.W_kr(x))                              # (B,n,d_rope) ← 也缓存(小)
        k_c = self.W_uk(c_kv).view(B, n, self.h, self.dh)        # 解压 K
        v   = self.W_uv(c_kv).view(B, n, self.h, self.dh).transpose(1, 2)
        # K = [解压部分 ‖ RoPE 部分],RoPE 段所有头共享
        k = torch.cat([k_c, k_rope.unsqueeze(2).expand(-1, -1, self.h, -1)], dim=-1)
        k = k.transpose(1, 2)                                    # (B,h,n,dh+d_rope)
        q = rope_split(self.W_q(x), self.h, self.dh, self.d_rope, rope)  # Q 同样拆两段
        o = F.scaled_dot_product_attention(q, k, v.repeat(1,1,1,1))  # 维度对齐略
        return self.W_o(o.transpose(1, 2).reshape(B, n, -1))

# ❌ 常见错误:把 RoPE 直接作用在解压后的 k_c 上
#    k_c = rope(self.W_uk(c_kv))   # 错!位置必须解耦走单独一支,否则无法做权重吸收、缓存又变大
```

## 面试高频
- **MLA 和 MQA/GQA 的本质区别?** MQA/GQA 缓存的是**完整 K/V**(只是头数少);MLA 缓存的是**低秩潜向量 c**,各头有独立上投影解压 → 显存像 MQA、表达力像 MHA。
- **为什么 RoPE 要"解耦"?** [[031 RoPE 旋转位置编码(推导与实现)|RoPE]] 让 K 带位置相关旋转,与位置无关的潜向量 $c$ 无法吸收它;若硬压进 $c$,就不能把上投影矩阵吸收进 $W_Q/W_O$、缓存也回到 K/V 级别。故把位置信息单拎一支(共享、维度小)。
- **"吸收(absorb)"是什么、为何重要?** 把上投影 $W^{UK}/W^{UV}$ 合并进 $W_Q/W_O$,推理时直接在潜空间计算,**不必解压出完整 K/V**——这才让 MLA 既省显存又不增加 decode 算力。
- **MLA 能用 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 吗?** 需适配(解耦 RoPE 的拼接维度、吸收后的等价形式),DeepSeek 提供了专门 kernel。
- **MLA vs GQA 取舍?** MLA 省得更狠且质量更好,但实现复杂(下/上投影、解耦、吸收);GQA 实现简单、生态成熟。新模型若追求极致长上下文/吞吐选 MLA。
- **MLA 为什么质量能不降反升?** 低秩压缩相当于在 K/V 上加了一个**结构先验/正则**(强制 K、V 落在共享低秩子空间),既去冗余又略增稳健;加上各头独立上投影保留多头表达,综合下来基准常优于同规模 MHA 基线。
- **Q 也压缩吗?** 是。DeepSeek-V2 对 Q 也做低秩(下投影到 $d_c'$ 再上投影),但 Q 不进 KV-Cache、压它只为省**训练激活显存**,与省 KV-Cache 是两件事——别混。
- **解耦 RoPE 那一支为什么所有头共享、维度还小?** 位置那支 $k^R_t$ 只携带位置信息、与内容无关,共享一份即可;它单独缓存,维度 $d_{RoPE}$(如 64)远小于 $d_{head}$,对总缓存影响很小,却让"吸收"得以成立。
- **MLA 的缓存里到底存了哪两样?** ① 低秩潜向量 $c^{KV}_t$($d_c$ 维,内容);② 解耦 RoPE 的 K 支 $k^R_t$($d_{RoPE}$ 维,位置)。合计约 $d_c+d_{RoPE}$,这就是它每 token 缓存的全部。
- **MLA 算 [[018 MQA 多查询注意力|MQA]]/[[019 GQA 分组查询注意力|GQA]] 那一类吗?** 不是同机制。MQA/GQA 是"减少 KV 头数"(仍存完整 K/V);MLA 是"低秩联合压缩 + 解压 + 吸收"。它是 KV 优化的**另一条主线**。

## 关键事实
- 出处:DeepSeek-AI,*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*,2024,arXiv:2405.04434。MLA 是其核心注意力创新。
- 核心机制:**low-rank key-value joint compression**——把 K/V 联合压成潜向量 $c$,缓存它;配 **decoupled RoPE** 单独处理位置。
- 实测:KV-Cache 较 MHA **减少约 93.3%**,生成吞吐显著提升,质量优于 MHA。
- 推理技巧:**矩阵吸收(absorb)**——上投影并入 $W_Q/W_O$,潜空间直接计算,免解压。
- 量化数据:DeepSeek-V3 每 token KV 约 70 KB,对比同档 GQA 模型(LLaMA-3.1 405B)约 516 KB,约 93% 削减;后续 DeepSeek-V3.2 进一步把潜向量量化(如 FP8)再压。
- 谱系:DeepSeek-V2(2024)首提 MLA → V3 沿用并规模化 → V3.2 叠加稀疏注意力。MLA 已被视为 GQA 之外另一条主流 KV 优化路线。
- 与邻接概念:是 [[018 MQA 多查询注意力|MQA]] / [[019 GQA 分组查询注意力|GQA]] 之后更激进的 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]];DeepSeek-V2 还用了 [[047 DeepSeek MoE：细粒度与共享专家|细粒度 MoE]];位置编码细节见 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]]。
