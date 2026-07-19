[[020 MLA 多头潜在注意力(DeepSeek)]]:DeepSeek-V2 提出的注意力——把 K/V **联合压成一个低秩潜向量 $c$**,[[102 KV-Cache|KV-Cache]] 缓存该潜向量及小的解耦位置分量；数学上可把部分上投影吸收进相邻权重，避免 decode 时重建完整 K/V。它是与 [[018 MQA 多查询注意力|MQA]]、[[019 GQA 分组查询注意力|GQA]] 不同的 KV-cache 优化路线，实际质量/吞吐必须按具体模型、kernel 和服务配置评测。

## 直觉:别存完整 K/V,存它们的"压缩包"
[[019 GQA 分组查询注意力|GQA]] 通过减少 K/V 头数省缓存,但本质还是"存若干份完整 K/V"。MLA 换个思路:**K 和 V 高度相关、且有冗余,可以一起压到一个低维潜空间**。

每个 token 的**内容支**不再缓存 h 份(或 g 份)$d_{head}$ 维的 K/V,而是缓存一个 $d_c$ 维潜向量 $c^{KV}$($d_c\ll h\cdot d_{head}$)；**完整 MLA cache payload = $c^{KV}+k^R$**，其中 $k^R$ 是小的解耦位置支。需要时,用各头独立的上投影矩阵把 $c^{KV}$ "解压"成该头的内容 K、V。

关键区别于 MQA/GQA:**MQA 是真共享一份 K/V(表达力受限);MLA 是共享一个压缩潜向量,但每个头有独立解压器** → 既省缓存,又保住多头的表达力。

类比:不给每人发一整套百科全书(MHA),也不让大家抢一本(MQA),而是发一张高度压缩的"索引卡"(潜向量 c),每人用自己的解码方式(上投影)还原出需要的内容。

## 例子:压缩到一个潜向量
设 MHA 配置 h=128 头、d_head=128,则每 token 缓存 K+V 共 $2\times128\times128=32768$ 维。

若用一个**示例配置** $d_c=512,d_{RoPE}=64$，每 token 只缓存约 $d_c+d_{RoPE}=576$ 个标量，数量级从 32768 降到几百，约为 1/57。这个算例用于解释形状，**不是** MLA 或 DeepSeek-V2 的固定签名。

DeepSeek-V2 论文(2024)报告：在其与 DeepSeek 67B 的比较配置中，KV cache 减少 93.3%、最大生成吞吐提升 5.76 倍。它们是论文系统实验的结果，不能直接外推到任意 MLA 实现。

> 直觉小账:省缓存的核心是把**内容支**的缓存维度从 $2hd_{head}$ 换成 $d_c$；完整 payload 还要加一条小的解耦 RoPE K 支 $k^R$。只要 $d_c+d_{RoPE}\ll 2hd_{head}$，总账本仍显著变小。

**横向对比每 token 缓存维度(同一规模)。** 设 h=128、d_head=128。

| 方案 | 每 token 缓存的"等效维度" | 相对 MHA |
|---|---|---|
| MHA | $2hd_{head}=32768$ | 1 |
| GQA(g=8) | $2g\,d_{head}=2048$ | 1/16 |
| MQA | $2d_{head}=256$ | 1/128 |
| MLA(示例 $d_c=512,d_{RoPE}=64$) | $d_c+d_{RoPE}=576$ | 约 1/57 |

关键差别是机制：MQA 让所有 query 头读取同一份 K/V；MLA 让每个头经自己的上投影从共同潜向量取信息。谁的质量更高并非由缓存维度单独决定，需在同一训练预算、数据、上下文长度和 serving kernel 下比较。避免拿不同模型规模、量化和系统配置的“每 token KB”做结论。

## 原理:低秩联合压缩 + 解耦 RoPE
**第一步:下投影压缩(缓存内容支)。** 对输入 $h_t$ 投影出潜向量:
$$c^{KV}_t = W^{DKV}\,h_t,\qquad c^{KV}_t\in\mathbb{R}^{d_c},\ d_c\ll d$$
内容支 cache **只存** $c^{KV}_t$；**完整 MLA cache payload = $c^{KV}_t + k^R_t$**，不能把“内容支”误说成完整 payload。

**第二步:上投影解压。** 各头的 K、V 由 $c$ 还原:
$$k^{C}_{t,i} = W^{UK}_i\,c^{KV}_t,\qquad v_{t,i} = W^{UV}_i\,c^{KV}_t$$
每个头 $i$ 有独立的 $W^{UK}_i,W^{UV}_i$ → 保留多头表达力(这是 MLA ≠ MQA 的关键)。

**第三步:解耦 RoPE(易踩的坑)。** [[031 RoPE 旋转位置编码(推导与实现)|RoPE]] 对 Q/K 施加位置相关旋转。为了让内容潜向量保持可被矩阵吸收的、与位置分开的形式，MLA 不把常规 RoPE 旋转后的 K 直接混入 $c$，而把 K 拆成两段:
$$k_{t,i} = \big[\,\underbrace{k^{C}_{t,i}}_{\text{从 }c\text{ 解压,无位置}}\ \big\Vert\ \underbrace{k^{R}_{t}}_{\text{带 RoPE,所有头共享}}\,\big]$$
位置那一支 $k^R_t$ 单独走、维度小($d_{RoPE}$),也缓存它。Q 同理拆两段。这就是"**解耦 RoPE**(decoupled RoPE)"。

![[attn-MLA解耦RoPE.png]]

**第四步:吸收(推理加速的精髓)。** 注意 $q^\top k^C = q^\top W^{UK}_i c$,可把 $W^{UK}_i$ **吸收进** query 侧的等价权重；同理 $W^{UV}_i$ 可与输出侧组合。于是采用等价实现的 decode kernel 不必真的解压完整 K/V，而可直接在潜空间计算。是否实际实现为“吸收”以及收益大小取决于 kernel 与并行布局。

**吸收的代数推导(逐步看为什么省算)。** 头 $i$ 的非 RoPE 部分打分:
$$q_{t,i}^\top k^{C}_{s,i}=q_{t,i}^\top\big(W^{UK}_i\,c^{KV}_s\big)=\underbrace{\big((W^{UK}_i)^\top q_{t,i}\big)}_{\tilde q_{t,i}}{}^\top\,c^{KV}_s$$
把 $W^{UK}_i$ 提前乘进 query 侧得到 $\tilde q_{t,i}=(W^{UK}_i)^\top q_{t,i}$,于是内容支打分直接是 $\tilde q_{t,i}^\top c^{KV}_s$——**内容支的 KV 侧只需缓存并读取 $c^{KV}_s\in\mathbb{R}^{d_c}$,从不物化出 $h$ 份 $d_{head}$ 维的 $k^C$**。输出侧同理:$\sum_s\alpha_s v_{s,i}=\sum_s\alpha_s W^{UV}_i c^{KV}_s=W^{UV}_i\big(\sum_s\alpha_s c^{KV}_s\big)$,$W^{UV}_i$ 可并入 $W_O$。完整 payload 仍要读取独立缓存的 $k^R$ 计算位置支；两个内容上投影则"消失"进相邻权重。这样内容计算只在 $d_c$ 维潜空间进行，且不解压 $h$ 份完整 K/V。这是 MLA "缓存小到 MQA 级、表达力却是 MHA 级"得以两全的工程机关。

![[attn-MLA吸收.png]]

整体上,MLA 以较小的 KV cache 表示换取更复杂的投影与 kernel；它不是“MHA 表达力不变、缓存必然等于 MQA”的数学保证。

![[attn-MLA低秩KV.png]]

## 代码:MLA 前向骨架
```python
import torch
import torch.nn.functional as F

def apply_rope(x, base=10_000):
    """教学版 RoPE，x:(B,h,T,d)，要求 d 为偶数。"""
    _, _, T, d = x.shape
    assert d % 2 == 0
    inv_freq = base ** (-torch.arange(0, d, 2, device=x.device,
                                      dtype=torch.float32) / d)
    phase = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv_freq
    cos, sin = phase.cos().to(x.dtype)[None, None], phase.sin().to(x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

# ✅ 可运行的“显式解压”教学前向：服务端可进一步把上投影吸收进相邻权重。
class ToyMLA(torch.nn.Module):
    def __init__(self, d, h, dh, d_c, d_rope):
        super().__init__()
        assert dh % 2 == 0 and d_rope % 2 == 0
        self.h, self.dh, self.d_rope = h, dh, d_rope
        self.W_dkv = torch.nn.Linear(d, d_c, bias=False)         # cache 内容支 c^KV；完整 payload 还需共享 k^R
        self.W_uk = torch.nn.Linear(d_c, h * dh, bias=False)      # 每头内容 K 上投影
        self.W_uv = torch.nn.Linear(d_c, h * dh, bias=False)      # 每头 V 上投影
        self.W_kr = torch.nn.Linear(d, d_rope, bias=False)        # 共享 K 的 RoPE 支；也进入 cache
        self.W_qc = torch.nn.Linear(d, h * dh, bias=False)
        self.W_qr = torch.nn.Linear(d, h * d_rope, bias=False)
        self.W_o = torch.nn.Linear(h * dh, d, bias=False)

    def forward(self, x):                                         # x:(B,T,d)
        B, T, _ = x.shape
        c_kv = self.W_dkv(x)                                     # (B,T,d_c)  decode cache 的内容支
        k_c = self.W_uk(c_kv).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.W_uv(c_kv).view(B, T, self.h, self.dh).transpose(1, 2)
        k_r = apply_rope(self.W_kr(x).unsqueeze(1))              # (B,1,T,d_rope),共享给各 K 头
        q_c = self.W_qc(x).view(B, T, self.h, self.dh).transpose(1, 2)
        q_r = apply_rope(self.W_qr(x).view(B, T, self.h, self.d_rope).transpose(1, 2))
        q = torch.cat((q_c, q_r), dim=-1)
        k = torch.cat((k_c, k_r.expand(-1, self.h, -1, -1)), dim=-1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.W_o(out.transpose(1, 2).reshape(B, T, -1))

toy = ToyMLA(d=32, h=4, dh=8, d_c=12, d_rope=4)
assert toy(torch.randn(2, 5, 32)).shape == (2, 5, 32)

# ❌ 不能调用未定义的 rope_split，也不应以 v.repeat(1,1,1,1) 伪装维度正确。
# 教学前向显式解压 K/V；生产 MLA 用吸收后的等价 kernel，完整 cache payload = c^KV（内容） + k^R（位置）。
```

## 面试高频
- **MLA 和 MQA/GQA 的本质区别?** MQA/GQA 缓存的是**完整 K/V**(只是头数少)；MLA 的内容支缓存低秩潜向量 $c^{KV}$，完整 payload 再加小的共享解耦位置支 $k^R$，各头有独立上投影解压 → 缓存账本小、但不等同于一份完全共享 K/V。
- **为什么 RoPE 要"解耦"?** [[031 RoPE 旋转位置编码(推导与实现)|RoPE]] 让 K 带位置相关旋转,与位置无关的潜向量 $c$ 无法吸收它;若硬压进 $c$,就不能把上投影矩阵吸收进 $W_Q/W_O$、缓存也回到 K/V 级别。故把位置信息单拎一支(共享、维度小)。
- **"吸收(absorb)"是什么、为何重要?** 把上投影 $W^{UK}/W^{UV}$ 合并进 $W_Q/W_O$,推理时直接在潜空间计算,**不必解压出完整 K/V**——这才让 MLA 既省显存又不增加 decode 算力。
- **MLA 能用 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 吗?** 需适配(解耦 RoPE 的拼接维度、吸收后的等价形式),DeepSeek 提供了专门 kernel([[LLM Infra/026 FlashMLA：DeepSeek 的 MLA 推理内核|FlashMLA]])。
- **MLA vs GQA 取舍?** GQA 减少 KV 头数，结构简单、生态成熟；MLA 改成潜向量、解耦 RoPE 与吸收，对实现和 kernel 要求更高。选择应比较目标模型在相同上下文、batch、精度与服务引擎上的端到端吞吐、TTFT/TPOT、显存和质量。
- **MLA 为什么可能保持质量?** 每头可对共同潜向量使用独立上投影，容量不等同于“一份完全共享 K/V”。但低秩瓶颈也可能损失信息；只能以受控 ablation 和 held-out 指标说明某个配置的效果，不能从结构推出“必优于 MHA/MQA”。
- **Q 也压缩吗?** 是。DeepSeek-V2 对 Q 也做低秩(下投影到 $d_c'$ 再上投影),但 Q 不进 KV-Cache、压它只为省**训练激活显存**,与省 KV-Cache 是两件事——别混。
- **解耦 RoPE 那一支为什么可共享、维度为何小?** DeepSeek-V2 的设计将 K 的位置分量单列并共享；其维度是架构超参，应随 checkpoint 配置核对，而不是把“64”当作 MLA 通用常数。该拆分服务于位置建模与可吸收的内容分量。
- **MLA 的缓存里到底存了哪两样?** ① 低秩潜向量 $c^{KV}_t$($d_c$ 维,内容);② 解耦 RoPE 的 K 支 $k^R_t$($d_{RoPE}$ 维,位置)。合计约 $d_c+d_{RoPE}$,这就是它每 token 缓存的全部。
- **MLA 算 [[018 MQA 多查询注意力|MQA]]/[[019 GQA 分组查询注意力|GQA]] 那一类吗?** 不是同机制。MQA/GQA 是"减少 KV 头数"(仍存完整 K/V);MLA 是"低秩联合压缩 + 解压 + 吸收"。它是 KV 优化的**另一条主线**。

## 关键事实
- 出处:DeepSeek-AI,*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*,2024,arXiv:2405.04434。MLA 是其核心注意力创新；该文报告的 93.3% KV cache 降低与 5.76× 最大生成吞吐均针对文中 DeepSeek-V2 与 DeepSeek 67B 的比较设置。
- 核心机制:**low-rank key-value joint compression**——把 K/V 内容联合压成潜向量 $c^{KV}$ 并缓存；配 **decoupled RoPE**，把共享位置支 $k^R$ 作为完整 payload 的另一小段单独缓存。
- 实测卡:论文在其训练/服务设置报告 KV cache 减少 93.3%、最大生成吞吐 5.76×；使用其他模型、精度、context length 或 kernel 前应重新测量。
- 推理技巧:**矩阵吸收(absorb)**——上投影并入 $W_Q/W_O$,潜空间直接计算,免解压。
- 版本边界:本文以 DeepSeek-V2 论文(2024)的 MLA 机制与其已注明的实验卡为准；后续模型的每 token 字节、量化与稀疏注意力配置应查对应模型卡/论文，不能混进 V2 的结论。
- 与邻接概念:是 [[018 MQA 多查询注意力|MQA]] / [[019 GQA 分组查询注意力|GQA]] 之后更激进的 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]];DeepSeek-V2 还用了 [[047 DeepSeek MoE：细粒度与共享专家|细粒度 MoE]];位置编码细节见 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]]。
