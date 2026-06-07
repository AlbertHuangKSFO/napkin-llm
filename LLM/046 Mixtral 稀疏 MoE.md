[[046 Mixtral 稀疏 MoE]]:Mistral AI 2024 年开源的稀疏 MoE 模型,每层 **8 个专家**、每 token **top-2** 路由——总参约 **47B**,但每 token 只激活约 **13B**;以远低于稠密 70B 的推理算力,匹配甚至超过 Llama 2 70B 与 GPT-3.5,是开源 MoE 的标杆配置。

## ① 直觉:8 个 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]] 排排站,每 token 只用 2 个

Mixtral 的骨架就是 Mistral 7B(用 [[019 GQA 分组查询注意力|GQA]]、[[021 局部与滑窗注意力(Longformer、Mistral SWA)|滑窗注意力]]),改动只在 FFN:把每层那一个稠密 FFN,换成**8 个并列的专家 FFN + 一个 [[043 门控路由与 top-k 选择|门控]]**。每个 token 进来,门控选出 **top-2** 个专家,加权汇总。

"8×7B" 的名字有点误导:它**不是** 8 个独立 7B 模型,也**不是** $8\times7=56$B。注意力、embedding 等是**共享**的,只有 FFN 被复制成 8 份。所以:

- **总参 ≈ 47B**(共享部分 + 8 份专家 FFN),不是 56B。
- **激活参数 ≈ 13B**(共享部分 + top-2 选中的 2 份专家 FFN)。

直觉账:推理成本(算力、激活内存带宽)只跟**激活的 13B** 走 → 快得像 13B 模型;而模型容量(能记多少知识)跟**总参 47B** 走 → 强得接近大模型。这正是 [[042 MoE 动机：稀疏激活与容量解耦|稀疏激活]]的卖点。

⚠️ **但显存得装下全部 47B**:8 个专家都要常驻显存(不知道下一个 token 路由到谁),所以 Mixtral 省的是**算力/带宽**,不省**显存容量**。这是 MoE 部署的核心矛盾,见 [[049 专家并行 EP 与 MoE 部署|专家并行]]。

## ② 例子:一个 token 的路由,以及参数账

设某 token 在某层,门控 logits 经 softmax 得 $g=[0.05,0.41,0.03,0.08,0.32,0.04,0.02,0.05]$(8 维)。top-2 = 专家 2(0.41)、专家 5(0.32):

$$
y = \frac{0.41}{0.41+0.32}\text{FFN}_2(x) + \frac{0.32}{0.41+0.32}\text{FFN}_5(x)
$$

(选中专家的门控分**重归一化**到和为 1。)其余 6 个专家这一层对该 token **零计算**。

**参数账**(粗略):Mixtral 8×7B 有 32 层,每层 8 个专家 FFN。单个 FFN(SwiGLU,hidden≈14336)约 1.3 亿×… 累加后:

- 全部 8 专家 × 32 层 + 共享(注意力/embedding/norm) ≈ **46.7B 总参**。
- 每 token 每层只激活 2/8 专家 → 激活 ≈ **12.9B**。

所以同样吐一个 token,Mixtral 的前向 FLOPs 约等于一个 13B 稠密模型,却拥有 47B 的"知识库"。Mixtral 训练上下文 32k,在数学、代码、多语言上明显强于 Llama 2 70B。

**Mixtral 8×7B 完整配置(精确数字)**:32 层、$d_{model}=4096$、32 个注意力头(GQA,8 个 KV 头)、每个专家 SwiGLU FFN 中间维 **14336**、每层 **8 专家 top-2**、词表 **32000**(同 Mistral 7B 的 SentencePiece-BPE)、上下文 **32768**、配 [[021 局部与滑窗注意力(Longformer、Mistral SWA)|SWA]]。8 个专家全在每一层(共 $32\times8=256$ 个专家 FFN)。

**「8×7B 为什么是 47B 不是 56B」的精算**。Mistral 7B 里 FFN 占约 6B/7B,其余约 1B 是共享的(注意力 + 嵌入 + norm)。复制 8 份的只有 FFN:共享 1B + 8×(每专家约 5.75B 的 FFN 相关)≈ 1 + 46 ≈ **47B**。激活时只算共享 + 2 份专家:1 + 2×5.75 ≈ **12.9B ≈ 13B**。关键:**注意力和嵌入不复制,所以不是 8×7=56**。

![[moe-Mixtral路由.png]]

## ③ 原理:Mixtral 的 MoE 层公式

每层用 8 个专家、top-2。门控 $W_g\in\mathbb{R}^{8\times d}$:

$$
G(x) = \text{Softmax}\big(\text{TopK}(x\,W_g^\top,\,2)\big)
$$

这里 $\text{TopK}$ 先把非 top-2 的 logits 置 $-\infty$,**再** softmax,所以未选中专家权重恰为 0、选中的两个自动归一。输出:

$$
y = \sum_{i=0}^{7} G(x)_i\cdot \text{FFN}_i(x) = \sum_{i\in\text{top2}} G(x)_i\cdot \text{SwiGLU}_i(x)
$$

每个专家是标准 [[008 前馈网络 FFN(为何 4 倍、为何两层)|SwiGLU FFN]]。

几个值得注意的工程点:

- **每一层都是 MoE 层**(不像 Switch/GShard 隔层放),32 层全是 8 专家。
- **门控用 top-2 + 先 TopK 后 softmax**,等价于在选中专家上做局部 softmax,无需手动重归一化。
- Mixtral 论文报告**路由没有明显的"主题专精"**(专家不对应"代码/数学"这种语义类别),路由更多在**句法/token 级**有规律——常被面试问到。
- 推理实测:虽然显存要 47B,但因每 token 只算 13B,**吞吐接近 13B 模型**;配合专家并行/卸载可进一步省显存。

**论文的专家专精分析(常被深挖)**。Mixtral 论文专门看了「专家分工」:

- 按**数据领域**(代码 vs 数学 vs 维基)统计路由,发现**没有**「这个专家专管代码、那个专管数学」的语义分工——不同领域的路由分布很接近。
- 但在**句法/token 级**有规律:连续 token 常被路由到同一专家(时间局部性),缩进、标点等特定 token 倾向固定专家。
- 启示:MoE 的「专家」更像**学到的某种计算子程序**,不对应人类可解释的主题——这与「细粒度专家能否专精」的讨论相关(见 [[047 DeepSeek MoE：细粒度与共享专家|DeepSeekMoE]] 想用更细的粒度逼出更强专精)。

![[moe-门控topk.png]]

## ④ 代码:Mixtral 风格 top-2 MoE 层(❌ vs ✅)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(d,h,bias=False), nn.Linear(d,h,bias=False), nn.Linear(h,d,bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MixtralMoE(nn.Module):
    def __init__(self, d, h, n_experts=8, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d, h) for _ in range(n_experts))
        self.k = top_k

    # ❌ 错:对全部 8 个专家都算 FFN 再加权 → 失去稀疏,算力 8 倍
    def forward_dense_wrong(self, x):
        g = F.softmax(self.gate(x), -1)
        return sum(g[..., i:i+1] * e(x) for i, e in enumerate(self.experts))

    # ✅ 对:先 TopK 再 softmax,只算选中的 2 个专家
    def forward(self, x):                              # x:(tokens, d)
        logits = self.gate(x)                          # (T, 8)
        val, idx = logits.topk(self.k, dim=-1)         # 先取 top-2 的 logits
        w = F.softmax(val, dim=-1)                     # 仅在选中专家上 softmax → 自动归一
        y = torch.zeros_like(x)
        for slot in range(self.k):
            for e in idx[:, slot].unique():
                m = idx[:, slot] == e
                y[m] += w[m, slot, None] * self.experts[e](x[m])
        return y
```

## 面试高频

- **Q:Mixtral 8×7B 是 56B 参数吗?** 不是。注意力/embedding 共享,只有 FFN 复制成 8 份 → 总参 **≈47B**;每 token top-2 激活 **≈13B**。
- **Q:既然只激活 13B,显存是不是也只要 13B?** 不。8 个专家都得常驻显存(下个 token 可能路由到任意专家),显存要装满 **47B**;省的是算力/带宽,不是显存容量。
- **Q:Mixtral 用 top 几?跟 Switch 比?** top-2(回到 GShard 路线);[[045 Switch Transformer 与 GShard|Switch]] 是 top-1。top-2 质量更稳,是开源 MoE 事实标准。
- **Q:Mixtral 的专家会按主题分工吗(代码专家、数学专家)?** 论文发现**没有**明显语义/主题专精,路由更多体现在句法和 token 层面的规律性。
- **Q:Mixtral 每层都是 MoE 吗?** 是,32 层全是 8 专家 MoE(不同于 Switch/GShard 的隔层放置)。
- **Q:它怎么实现 top-2 的归一化?** "先 TopK 把其余置 $-\infty$、再 softmax",选中两专家权重自动和为 1,无需手动重归一化。
- **Q:Mixtral 专家 FFN 多大?** SwiGLU,中间维 14336($d=4096$);每层 8 个,共 32 层 × 8 = 256 个专家 FFN。
- **Q:为什么不是 8×7=56B?** 只有 FFN 复制 8 份,注意力/嵌入/norm 共享不复制;47B = 共享约 1B + 8 份专家约 46B。
- **Q:Mixtral 和 Mistral 7B 什么关系?** 同骨架(GQA + SWA + 32 层 + d=4096),唯一区别是把每层稠密 FFN 换成 8 专家 top-2 MoE。
- **Q:Mixtral 的上下文多长?** 32768(32k);继承 Mistral 的 SWA + rolling buffer 支持长上下文低成本。

## 关键事实

- 出处:Jiang et al.(Mistral AI),*Mixtral of Experts*,2024,arXiv:2401.04088。每层 **8 专家、top-2** 路由,基于 Mistral 7B(含 [[019 GQA 分组查询注意力|GQA]] 与 [[021 局部与滑窗注意力(Longformer、Mistral SWA)|SWA]])。
- **总参 ≈ 47B,每 token 激活 ≈ 13B**;训练上下文 32k。
- 配置:32 层、$d=4096$、32 注意力头(GQA,8 KV 头)、专家 SwiGLU 中间维 14336、每层 8 专家 top-2、词表 32000、上下文 32768。
- 专精分析:无明显**领域**专精(代码/数学路由分布接近),但有**句法/token 级**规律(连续 token 倾向同专家、特定 token 固定专家)。
- 性能:在多数基准上**匹配或超过 Llama 2 70B 与 GPT-3.5**,数学/代码/多语言尤其领先(同时推理快得多)。
- 路由门控:**先 TopK(置 −∞)再 softmax**;论文报告专家**无明显主题专精**,规律多在句法/token 级。
- 显存需装下全部专家(≈47B),推理省的是算力与激活带宽 → 部署需 [[049 专家并行 EP 与 MoE 部署|专家并行]] 与卸载;训练稳定性同样依赖 [[044 专家容量、丢弃与负载均衡损失|负载均衡]] 与 [[048 路由稳定性：router z-loss|z-loss]]。
