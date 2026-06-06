[[040 现代 decoder-only 配方汇总]]:把 2023—2024 年主流大模型([[038 LLaMA 架构解剖|LLaMA]]、[[039 Mistral、Qwen、DeepSeek 架构选择|Mistral、Qwen、DeepSeek]]、Gemma 等)的共同选型压成一张表——五个“旋钮”各对应一个目标,记住这张表就握住了现代 LLM 架构的 95%。

## 直觉:五个旋钮,各管一件事
现代 decoder-only 模型彼此 95% 相同。与其逐个背,不如把架构拆成**五个旋钮**,每个旋钮解决一个具体痛点:

| 旋钮 | 现代默认 | 解决什么 |
|---|---|---|
| 归一化 | Pre-RMSNorm | **训练稳定** + 更快 |
| 位置编码 | RoPE | **长度外推** / 相对位置 |
| FFN 激活 | SwiGLU | **效果质量** |
| 注意力 / KV | GQA(或 MLA),部分配 SWA | **省 KV-Cache / 推理成本** |
| FFN 结构 | 稠密 或 MoE | **解耦容量与算力**(大模型趋向 MoE) |

加两个小惯例:**去掉 bias**、**tied embedding**([[016 输出层、tied embedding 与 logits|输入输出权重共享]])。

一句话配方:**decoder-only · Pre-RMSNorm · RoPE · SwiGLU · GQA/MLA(±SWA) · 大模型加 MoE · 去 bias · tied embed。** 各家的差异**只落在最后两列**(注意力怎么省 KV、要不要 MoE),其余几乎一模一样。

## 例子:同一张表套到四个模型
- **LLaMA-2 70B**:Pre-RMSNorm / RoPE / SwiGLU / **GQA(g=8)** / 稠密。
- **Mistral 7B**:同上,但注意力 **GQA + SWA(W=4096)**;Mixtral 把稠密换成 **MoE(top-2 of 8)**。
- **DeepSeek-V3**:Pre-RMSNorm / RoPE / SwiGLU / **MLA** / **细粒度 MoE**(671B 总、激活 37B)。
- **Qwen3-MoE**:Pre-RMSNorm / RoPE / SwiGLU / **GQA** / **MoE(top-8,去共享专家)**。

看出来了吗?**前三列(norm / pos / act)完全相同**,只有“注意力”和“FFN 结构”两列在变。这正是 [[039 Mistral、Qwen、DeepSeek 架构选择|各家架构选择]]的本质。

**更多机型对号入座**(把表用熟):

| 模型 | 归一化 | 位置 | 激活 | 注意力 | FFN |
|---|---|---|---|---|---|
| Gemma-2 | Pre + Post RMSNorm | RoPE | GeGLU | GQA + 局部/全局交替 | 稠密 |
| Phi-3 | Pre-RMSNorm | RoPE | SwiGLU | GQA | 稠密 |
| Qwen2.5 | Pre-RMSNorm + QK-Norm | RoPE | SwiGLU | GQA | 稠密/MoE |
| Command-R | Pre-LayerNorm | RoPE | SwiGLU | GQA | 稠密 |

注意有些「微创新」:Gemma-2 同时在子层**前后各加一次** RMSNorm(sandwich norm)进一步稳训练,FFN 用 GeGLU(GELU 门控,与 SwiGLU 同类);Qwen2.5 加 QK-Norm。这些都属于「在五个旋钮里微调」,不动大格局。

![[hist-配方表.svg]]

## 原理:每个旋钮为什么是现在这个默认
**① Pre-RMSNorm**:Pre-Norm 把归一化放在子层输入端,残差主干是干净恒等路径,[[009 残差连接与梯度流|梯度]]畅通 → 深层好训(对比 [[010 层归一化：Pre-LN 与 Post-LN|Post-LN]] 深层易爆/难训)。[[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|RMSNorm]] 去掉减均值与平移:
$$\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot g$$
少一遍统计、更快,质量不降。部分新模型(如某些 Gemma/Qwen 变体)还加 **QK-Norm**(对 Q、K 单独归一化)进一步稳住注意力 logits。

**② RoPE**:旋转 Q/K,使注意力打分只依赖相对位置 $m-n$,长度外推友好;长上下文常配 **NTK-aware / YaRN** 等频率缩放扩展窗口。推导见 [[031 RoPE 旋转位置编码(推导与实现)|RoPE]]。

**③ SwiGLU**:门控前馈,质量优于 ReLU/GELU([[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)|激活函数对比]]):
$$\text{FFN}(x)=W_{\text{down}}\big(\text{Swish}(W_{\text{gate}}x)\odot(W_{\text{up}}x)\big),\quad \text{隐藏维}\approx\tfrac{8}{3}d$$
三矩阵,故缩隐藏维到 $\tfrac83 d$ 以保持参数量与 4d 的两矩阵 FFN 相当。

**④ 注意力 / KV**:[[014 注意力复杂度 O(n²) 与瓶颈|decode 瓶颈]]在 [[102 KV-Cache|KV-Cache]] 的显存带宽。[[019 GQA 分组查询注意力|GQA]] 把 K/V 头从 $h$ 砍到 $g$(KV-Cache $\propto g$,通常省 8×),是主流默认;[[020 MLA 多头潜在注意力(DeepSeek)|MLA]] 用低秩压缩省得更狠;[[021 局部与滑窗注意力(Longformer、Mistral SWA)|SWA]] 进一步把单层复杂度降到 $O(nW)$。

**⑤ FFN 结构(MoE)**:大模型趋向把稠密 FFN 换成 [[042 MoE 动机：稀疏激活与容量解耦|MoE]]——$E$ 个专家 + [[043 门控路由与 top-k 选择|路由器]],每 token 只激活 top-$k$:
$$y=\sum_{i\in\text{TopK}(x)} g_i(x)\,\text{FFN}_i(x)$$
**参数容量 $\propto E$,每 token 算力 $\propto k$**。这让模型在固定推理算力下塞进更多知识。是否上 MoE,是“小模型偏稠密、超大模型偏稀疏”的分水岭。

## 代码:配方字典 + 参数量粗算(可运行)
```python
# ✅ 把“现代配方”写成可切换的 spec;换模型只改最后两列
MODERN = dict(arch="decoder-only", norm="Pre-RMSNorm", pos="RoPE",
              act="SwiGLU", bias=False, tied_embed=True)

def model_spec(attn, ffn):                 # 只有这两个旋钮区分各家
    return {**MODERN, "attn": attn, "ffn": ffn}

zoo = {
    "LLaMA-2-70B": model_spec("GQA(g=8)",        "dense"),
    "Mistral-7B":  model_spec("GQA + SWA(4096)", "dense"),
    "Mixtral":     model_spec("GQA + SWA",        "MoE top-2/8"),
    "DeepSeek-V3": model_spec("MLA",              "MoE 细粒度+共享"),
}
for k, v in zoo.items():
    print(f"{k:13s} attn={v['attn']:18s} ffn={v['ffn']}")

# 稠密块参数量粗算:注意力 4·d² + SwiGLU FFN 3·d·hidden(hidden≈8/3·d → ≈8·d²)
def block_params(d, hidden):
    attn = 4 * d * d                       # Wq,Wk,Wv,Wo(忽略 GQA 对 K/V 的缩减)
    ffn  = 3 * d * hidden                  # gate,up,down
    return attn + ffn
print("一层 d=4096,hidden=11008 → 参数 ≈", block_params(4096, 11008) // 10**6, "M")
```

```python
# ❌ 误区:把每个新模型当“全新架构”从零学,背一堆名字
#    其实换的只有“注意力省KV方式”和“是否MoE”两列,前三列恒定

# ✅ 正确:记住一句话配方 + 五个旋钮的对应目标,新模型对号入座即可
```

## 旋钮之外:还有哪些「现代默认」常被忽略

五个旋钮抓住骨架,但完整配方还有一圈「小习惯」,面试追问时能答上来加分:

- **分词**:Byte-level BPE(GPT/LLaMA-3)或 SentencePiece(LLaMA-1/2、Qwen),词表 3.2 万→12.8 万→15 万一路涨(见 [[050 分词总览与子词动机|分词总览]]、[[051 BPE 与 Byte-level BPE|BPE]])。
- **注意力实现**:FlashAttention(IO 感知、省显存、不改数学)几乎是标配;长上下文配 NTK-aware/YaRN 频率缩放。
- **混合精度**:bf16 训练 + fp32 主权重(优化器状态);MoE 路由 logits 常用 fp32 防溢出(见 [[048 路由稳定性：router z-loss|z-loss]])。
- **优化器**:AdamW + cosine/warmup 学习率;权重衰减、梯度裁剪。
- **初始化与残差缩放**:深层模型常按 $1/\sqrt{2N}$ 缩放残差分支初始化,稳住深层训练。
- **MoE 的放置**:Mixtral 每层都是 MoE,Switch/GShard 隔层放——是个可调选项。

口诀升级版:**骨架五旋钮 + 外圈(BPE 分词、FlashAttention、bf16、AdamW、残差缩放)**。

## 面试高频
- **现代 decoder-only 的“标准配方”是什么?** Pre-RMSNorm + RoPE + SwiGLU + GQA(或 MLA,±SWA)+(大模型)MoE + 去 bias + tied embedding,全是 decoder-only。能背这一串基本就过架构关。
- **这五个选择各解决什么问题?** 归一化→训练稳;位置→长度外推;激活→质量;注意力→省 KV-Cache/推理成本;MoE→解耦容量与算力。**一个旋钮一个目标**,是高频追问。
- **各家模型差异主要在哪两列?** 注意力(GQA / MLA / SWA 组合)和 FFN 结构(稠密 / MoE);前三列几乎统一。
- **为什么大模型偏向 MoE、小模型偏向稠密?** MoE 在固定每 token 算力下塞更多参数(容量),但占显存大、路由难调;小模型容量瓶颈不突出,稠密更简单稳。见 [[042 MoE 动机：稀疏激活与容量解耦|容量解耦]]。
- **和原始 Transformer(2017)比改了哪些?** 五处(见表):Post-LN→Pre-RMSNorm、正弦绝对→RoPE、ReLU/GELU→SwiGLU、MHA→GQA/MLA、稠密→(可选)MoE;另去 bias、tied embedding。骨架(attn+FFN+残差堆叠)未变。
- **QK-Norm 是什么?** 对 Q、K 在注意力前各做一次归一化,稳住 logits 量级、缓解大模型训练发散,部分新模型采用,属配方的可选增强。
- **GeGLU 和 SwiGLU 区别?** 同为门控 FFN,只是门控激活不同:SwiGLU 用 Swish/SiLU,GeGLU 用 GELU(Gemma 系)。都比 ReLU/GELU 单激活强,隐藏维同样取约 $\tfrac83 d$。
- **sandwich norm(Gemma-2)是什么?** 在子层**前后各一次**归一化(不只 Pre-Norm),进一步稳住深层/大模型训练,代价是多一遍 norm。
- **FlashAttention 改了模型结构吗?** 没有。它是 IO 感知的精确注意力**实现**,省显存、提速、数学结果不变,属外圈工程而非旋钮。
- **为什么说「记住一张表就握住 95%」?** 因为各家前三列(归一化/位置/激活)几乎统一,只有注意力与 FFN 两列分化;再加外圈小习惯即可覆盖绝大多数现代模型。

## 关键事实
- 配方组件出处:Pre-Norm(Xiong et al. 2020,arXiv:2002.04745)、RMSNorm(Zhang & Sennrich 2019)、RoPE(Su et al. 2021,arXiv:2104.09864)、SwiGLU(Shazeer 2020,arXiv:2002.05202)、GQA(Ainslie et al. 2023,arXiv:2305.13245)、MLA(DeepSeek-V2 2024,arXiv:2405.04434)、MoE(Shazeer et al. 2017,arXiv:1701.06538)。
- 采用此配方的代表:LLaMA-2/3、Mistral/Mixtral、Qwen 系列、DeepSeek-V2/V3、Gemma 等;细节有别但骨架统一。
- SwiGLU 隐藏维约 $\tfrac{8}{3}d$(对齐参数量);GQA 典型 $g=8$;MoE 典型 top-2(Mixtral)到 top-8(Qwen3)。
- 长上下文扩展:RoPE + NTK-aware / YaRN 频率缩放是常见做法。
- 微创新举例:QK-Norm(Qwen2.5 等)、sandwich/Pre+Post norm(Gemma-2)、GeGLU(Gemma)、滑窗+全局注意力交替(Gemma-2);均属五旋钮内的可选增强。
- 外圈标配:Byte-level BPE/SentencePiece 分词、FlashAttention 实现、bf16 混合精度、AdamW + warmup/cosine、残差/初始化缩放。
- 与邻接概念:每列分别链回 [[010 层归一化：Pre-LN 与 Post-LN|Pre-LN]]、[[031 RoPE 旋转位置编码(推导与实现)|RoPE]]、[[35 激活函数(Sigmoid、Tanh、ReLU、GELU、SwiGLU)|SwiGLU]]、[[019 GQA 分组查询注意力|GQA]]/[[020 MLA 多头潜在注意力(DeepSeek)|MLA]]/[[021 局部与滑窗注意力(Longformer、Mistral SWA)|SWA]]、[[042 MoE 动机：稀疏激活与容量解耦|MoE]];具体机型见 [[038 LLaMA 架构解剖|LLaMA]] 与 [[039 Mistral、Qwen、DeepSeek 架构选择|各家选择]]。
