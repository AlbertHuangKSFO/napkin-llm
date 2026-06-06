[[039 Mistral、Qwen、DeepSeek 架构选择]]:三家开源大模型都站在 [[038 LLaMA 架构解剖|LLaMA 四件套]]这个底座上,各自再加一招——Mistral 押滑窗注意力(SWA)+ 稀疏 MoE,DeepSeek 押 [[020 MLA 多头潜在注意力(DeepSeek)|MLA]] + 细粒度 MoE,Qwen 押稳健工程 + 全尺寸覆盖。

## 直觉:同一个底座,各加一招
2023 年后,开源 LLM 几乎都从同一个模板出发:[[038 LLaMA 架构解剖|LLaMA]] 的 decoder-only + Pre-RMSNorm + RoPE + SwiGLU + GQA。真正区分各家的,是它们在这个底座上**额外押注的那一招**——通常瞄准两个痛点之一:**注意力怎么更省**(KV-Cache / 长上下文),**FFN 要不要换成 MoE**(参数多但每 token 只激活一部分)。

- **Mistral**:在注意力上做文章——用[[021 局部与滑窗注意力(Longformer、Mistral SWA)|滑窗注意力 SWA]]让每个 token 只看最近一个窗口,长文本成本低;旗舰 Mixtral 把 FFN 换成稀疏 MoE。
- **DeepSeek**:两手都押激进创新——注意力用 [[020 MLA 多头潜在注意力(DeepSeek)|MLA]] 把 K/V 压成低秩潜向量(KV-Cache 省到极致),FFN 用细粒度 [[047 DeepSeek MoE：细粒度与共享专家|DeepSeekMoE]]。
- **Qwen**:不押单一激进创新,而是把每个零件做扎实 + 覆盖从 0.5B 到几百 B 的**全尺寸**,工程稳、生态全、好部署。

一句话记:**Mistral 赌稀疏注意力,DeepSeek 赌 MLA+MoE,Qwen 赌工程与全家桶。**

## 例子:三家旗舰的关键数字
- **Mistral 7B**:GQA + SWA,滑窗 $W=4096$;靠多层叠加,深层 token 的有效感受野可达 $W\times\text{层数}$ ≈ 上百 k。**Mixtral 8×7B**:每层 8 个专家、每 token 选 **top-2**;总参数约 47B,但每 token 只激活约 13B(MoE 的“参数多、算力省”)。
- **DeepSeek-V3**:MoE 总参数 **671B**,每 token 只激活约 **37B**(约 5.5%);注意力用 MLA,KV-Cache 较同规模 MHA **减少约 93%**(MLA 首发于 DeepSeek-V2,arXiv 2405.04434)。
- **Qwen**:从 0.5B 稠密到几百 B 的 MoE 全覆盖;长上下文优化(RoPE 频率调整 / 外推)、大词表多语种。Qwen3 的 MoE 把专家数提到 8、并去掉了早期版本的共享专家。

**滑窗注意力的「滚动缓存」(具体怎么省)**。Mistral 的 SWA 配一个 **rolling buffer KV-Cache**:窗口 $W=4096$,缓存就只留**最近 4096 个** token 的 K/V,环形覆盖旧的——KV-Cache 显存被钉死在 $O(W)$ 而非随序列线性涨 $O(n)$。生成第 10000 个 token 时,缓存里仍只有最近 4096 个,远处信息靠层间逐跳传递补偿。这是「长上下文 + 固定显存」的工程妙手。

**DeepSeek-V3 的 MoE 配置(精确数字)**。每个 MoE 层 = **1 个共享专家 + 256 个路由专家**,每 token 选 **top-8** 路由专家(+ 必经共享专家),每个专家中间维仅 2048(细粒度小专家);总参 671B、激活 37B(≈5.5%);并限制每 token 最多发往 4 个节点以压跨机通信,用**无辅助损失**(可学习偏置)做负载均衡(见 [[047 DeepSeek MoE：细粒度与共享专家|DeepSeekMoE]]、[[044 专家容量、丢弃与负载均衡损失|负载均衡]])。

![[hist-三家架构.svg]]

## 原理:三个押注点各解决什么
**① SWA(Mistral)——用稀疏换长上下文成本。** 标准注意力是 [[014 注意力复杂度 O(n²) 与瓶颈|$O(n^2)$]];SWA 让每个 token 只 attend 最近 $W$ 个,单层复杂度降到 $O(n\cdot W)$。远处信息靠**层间逐跳传递**:第 $\ell$ 层一个 token 能间接“看到” $\approx \ell\times W$ 范围。代价是远距离依赖需要足够深才能传到,见 [[021 局部与滑窗注意力(Longformer、Mistral SWA)|SWA]]。

**② MLA(DeepSeek)——用低秩压缩换 KV-Cache。** 不缓存完整 K/V,而把它们联合压成一个 $d_c$ 维潜向量 $c=W^{DKV}h$,只缓存 $c$;用时各头独立上投影“解压”。配解耦 RoPE,把 KV-Cache 砍到 MHA 的零头,详见 [[020 MLA 多头潜在注意力(DeepSeek)|MLA]]。

**③ MoE(Mistral + DeepSeek)——用稀疏激活解耦“容量”与“算力”。** 把一层的单个 FFN 换成 $E$ 个专家 FFN + 一个[[043 门控路由与 top-k 选择|路由器]],每 token 只选 top-$k$ 个专家计算:
$$y = \sum_{i\in \text{TopK}(x)} g_i(x)\,\text{FFN}_i(x)$$
总参数 $\propto E$(容量大),但每 token 算力 $\propto k$(只激活几个)。这就是 [[042 MoE 动机：稀疏激活与容量解耦|MoE 动机]];DeepSeek 用“**很多小专家 + 共享专家**”的细粒度变体提高专精度与利用率,见 [[047 DeepSeek MoE：细粒度与共享专家|DeepSeekMoE]];Mixtral 是经典“8 选 2”,见 [[046 Mixtral 稀疏 MoE|Mixtral]]。

**底座共性**:三家都保留 LLaMA 四件套,所以差异**不在骨架,而在“注意力怎么省 KV”和“FFN 用不用 MoE”这两个旋钮**——面试常考的就是这两个轴。

**三种「省 KV」手段的强度排序**。同一个痛点(decode 时 KV-Cache 吃显存带宽),三家给出不同力度的解:

| 手段 | KV-Cache 相对 MHA | 思路 | 代价 |
|---|---|---|---|
| MHA(基准) | 1× | 每头独立 K/V | 最贵 |
| GQA($g=8$) | 1/8 | 多 Q 头共享一组 K/V | 略损质量 |
| SWA($W$) | 钉死在 $O(W)$ | 只缓存最近窗口 | 远距离需够深 |
| MLA(低秩) | ≈ 1/13(省 ~93%) | K/V 压成潜向量只缓存它 | 实现复杂、需解耦 RoPE |

DeepSeek 的 MLA 最激进(省得最狠),Mistral 的 SWA 把显存**钉成常数**(跟序列长无关),GQA 是「人人都用」的稳妥默认。

## 谁加了 MoE、配置如何(横向表)

| 模型 | 注意力 | FFN | 总参 / 激活 | 路由 |
|---|---|---|---|---|
| LLaMA-2-70B | GQA | 稠密 | 70B / 70B | — |
| Mistral-7B | GQA+SWA | 稠密 | 7B / 7B | — |
| Mixtral-8×7B | GQA+SWA | MoE | 47B / 13B | top-2 of 8 |
| DeepSeek-V3 | MLA | 细粒度 MoE | 671B / 37B | top-8 of 256 + 1 共享 |
| Qwen3-MoE | GQA | MoE | (各档) | top-8,无共享 |

读法:**稠密模型「总参=激活」,MoE 模型激活远小于总参**——前者占算力也占显存,后者省算力但显存仍按总参算。

## 代码:用配置字典对比三家(可运行)
```python
# ✅ 把架构差异压成几个开关:底座相同,只切“注意力策略 / 是否 MoE”
def arch_config(name):
    base = dict(norm="RMSNorm(pre)", pos="RoPE", ffn_act="SwiGLU",
                arch="decoder-only", bias=False, tied_embed=True)
    presets = {
        "LLaMA-2-70B":  dict(attn="GQA(g=8)",        moe=None),
        "Mistral-7B":   dict(attn="GQA + SWA(W=4096)", moe=None),
        "Mixtral-8x7B": dict(attn="GQA + SWA",        moe="top-2 of 8"),
        "DeepSeek-V3":  dict(attn="MLA(低秩KV)",       moe="细粒度+共享, 激活37B/671B"),
        "Qwen3-MoE":    dict(attn="GQA",              moe="top-8, 去共享专家"),
    }
    return {**base, **presets[name]}

for m in ["LLaMA-2-70B", "Mistral-7B", "Mixtral-8x7B", "DeepSeek-V3", "Qwen3-MoE"]:
    c = arch_config(m)
    print(f"{m:14s} attn={c['attn']:22s} moe={c['moe']}")
# 输出可见:底座列(norm/pos/ffn)完全一样,只有 attn 和 moe 两列在变
```

```python
# ❌ 误区:以为这几家是“完全不同的架构”,逐个从零学
#    其实 90% 相同(LLaMA 四件套),只需记住每家额外押注的那一招

# ✅ 正确心智:一个底座 + 两个旋钮(KV 省法 / 是否 MoE)
#    Mistral→SWA;DeepSeek→MLA+细粒度MoE;Qwen→稳工程+全尺寸
```

## 面试高频
- **Mistral、Qwen、DeepSeek 架构上最大区别?** 底座(LLaMA 四件套)基本一致;区别在两个轴:**注意力省 KV 的方式**(Mistral=SWA、DeepSeek=MLA、Qwen=GQA)和**FFN 是否 MoE**(Mixtral/DeepSeek/Qwen3 是 MoE,稠密版不是)。
- **Mistral 的 SWA 怎么处理超出窗口的远距离依赖?** 靠层间传递:深层 token 的有效感受野 ≈ 窗口 × 层数;配合 [[102 KV-Cache|KV-Cache]] 的滚动缓存(rolling buffer)实现长上下文低成本。
- **DeepSeek 为什么同时上 MLA 和 MoE?** 两者解决不同维度:MLA 砍**推理时 KV-Cache 显存**,MoE 解耦**参数容量与每 token 算力**;叠加起来 → 大容量、长上下文、低推理成本,且训练成本可控。
- **MoE 模型“47B / 671B”这种数字怎么理解?** 前者(或更大数)是**总参数**(决定容量、占显存),后者“激活 X B”是**每 token 实际参与计算的参数**(决定算力)。Mixtral ≈ 47B 总 / 13B 激活;DeepSeek-V3 = 671B 总 / 37B 激活。见 [[042 MoE 动机：稀疏激活与容量解耦|容量解耦]]。
- **为什么 Qwen 没有“招牌创新”却很受欢迎?** 全尺寸覆盖(0.5B→几百 B)、长上下文与多语种打磨扎实、生态/工具链成熟 → 选型和部署成本低,是“稳”的胜利。
- **它们都还用 GQA 吗?** Mistral/Qwen 大量用 GQA;DeepSeek 用 MLA 取代 GQA(更激进)。GQA 见 [[019 GQA 分组查询注意力|GQA]]。
- **Mistral 的 rolling buffer KV-Cache 是什么?** 配合 SWA 的环形缓存,只保留最近 $W$ 个 token 的 K/V,旧的被覆盖,KV-Cache 显存固定在 $O(W)$、不随序列增长。
- **DeepSeek-V3 的 MoE 精确配置?** 每层 1 共享 + 256 路由专家、每 token top-8、专家中间维 2048;671B 总 / 37B 激活;每 token 最多 4 节点;无辅助损失负载均衡。
- **三种省 KV 手段强度排序?** MLA(≈省 93%)> SWA(钉成常数 $O(W)$)≈ GQA(省 8×);各有代价,见上表。
- **为什么这几家不算「全新架构」?** 90% 是 LLaMA 四件套,只在「注意力省 KV 方式 + 是否 MoE」两轴上分化,记两轴即可对号入座。

## 关键事实
- Mistral 7B:Jiang et al.,2023,arXiv:2310.06825。GQA + **SWA(W=4096)**;滚动缓存实现长上下文。Mixtral 8×7B:arXiv:2401.04088,**top-2 of 8** 专家,≈47B 总 / 13B 激活。
- DeepSeek-V2:DeepSeek-AI,2024,arXiv:2405.04434,首发 **MLA**(KV-Cache 较 MHA 省 ~93.3%)+ **DeepSeekMoE**。DeepSeek-V3:2024,arXiv:2412.19437,**671B 总 / 37B 激活**;每层 1 共享 + 256 路由专家、top-8、专家中间维 2048、每 token ≤4 节点、无辅助损失负载均衡;14.8T token 预训练。
- Qwen:Qwen / Qwen2(arXiv:2407.10671)/ Qwen2.5 / Qwen3 系列(阿里);全尺寸稠密 + MoE,GQA + RoPE 长上下文,多语种大词表。Qwen3 MoE 提到 8 专家、移除共享专家。
- 共同底座:三家均为 decoder-only + Pre-RMSNorm + RoPE + SwiGLU,详见 [[040 现代 decoder-only 配方汇总|配方汇总]]。
- 与邻接概念:注意力差异链回 [[019 GQA 分组查询注意力|GQA]] / [[020 MLA 多头潜在注意力(DeepSeek)|MLA]] / [[021 局部与滑窗注意力(Longformer、Mistral SWA)|SWA]];MoE 链回 [[042 MoE 动机：稀疏激活与容量解耦|MoE 动机]] / [[046 Mixtral 稀疏 MoE|Mixtral]] / [[047 DeepSeek MoE：细粒度与共享专家|DeepSeekMoE]];底座是 [[038 LLaMA 架构解剖|LLaMA]]。
