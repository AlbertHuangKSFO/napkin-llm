[[026 FlashMLA：DeepSeek 的 MLA 推理内核]] 是 DeepSeek 在 2025 年开源、面向 [[LLM/020 MLA 多头潜在注意力(DeepSeek)|MLA]](多头潜在注意力)布局的注意力内核库。它为压缩 KV、变长序列和分页 cache 提供专门路径；当前官方仓库同时列出 SM90 / SM100 的不同 dense / sparse kernel。它不是任何 MLA 服务自动"打满带宽"的保证，而是在给定内核、GPU、dtype、cache 格式和形状下争取更高有效带宽。

## 直觉类比
普通的 attention 内核像一个为"标准货架"设计的取货机器人:每个 token 的 K、V 按头整整齐齐放好,机器人按固定路线取。MLA 把货物压缩成"打包箱"(一个潜向量代替所有头的 K、V),货架布局全变了——标准机器人取不动,效率暴跌。FlashMLA 就是**为这个新货架重新设计的取货机器人**:知道每个箱子怎么解包(上投影)、变长队列怎么排班(变长调度)、怎么走最短路把 HBM 带宽吃满。

**生活类比**:坐飞机托运行李,分两步看。**① 压缩行李(MLA 干的事)**:传统每位旅客一堆散箱子(每 token 存 K、V 共 32768 个数)。DeepSeek-V3 的 MLA cache 不只含 512 维潜向量 $c_{KV}$，还要保存 64 维 RoPE 分量；即 576 个数。解码时每生成 1 个 token 都要读历史"行李"，行李变轻才有带宽空间。**② 专用传送带(FlashMLA 干的事)**:行李布局变了，需要内核理解 cache 页表和变长队列；调度元数据帮助把工作分给 SM。目标是减少无效搬运、提高有效带宽，效果必须在目标配置测量。

![[flash-026类比压缩行李专用传送带.png]]

## 小数字例子
以 DeepSeek-V3 的 128 头、每头 K/V 维度 128 为例，且假设 cache 使用 BF16:

- **传统 MHA 的 KV cache**:每 token 存 $2\times128\times128=32768$ 个数，即 $32768\times2=65536$ B。
- **MLA 的 cache schema**:存 $d_{c_{KV}}=512$ 的潜向量及 $d_{\rm RoPE}=64$ 的位置分量，即 $512+64=576$ 个数，或 $576\times2=1152$ B。只算潜向量的 $512/32768=1.56\%$；把实际需保留的 RoPE 分量也计入，则 $576/32768=1.76\%$。

在 8192 token 上下文中，这个**单层、单序列、BF16、只计 cache payload**的账为：传统 MHA $65536\times8192=536{,}870{,}912$ B($512$ MiB)，MLA $1152\times8192=9{,}437{,}184$ B($9$ MiB)，约 $56.9\times$ 的 payload 压缩。它不等于端到端速度比：页表、scale、读取模式、batch、其他层和计算都会参与。

## 原理:为什么 MLA 解码要专用内核
MLA 把每个 token 的 KV 压成共享潜向量 $c_{KV}=W_{DKV}\,h_t$,推理时**吸收(absorb)** 上投影矩阵,等效地直接在压缩空间算注意力:

$$
c_{KV,t}=W_{DKV}\,h_t,\qquad
k_t = W_{UK}\,c_{KV,t},\qquad
v_t = W_{UV}\,c_{KV,t}
$$

$$
\text{Attn}(q_t)=\text{softmax}\!\Big(\frac{q_t K^\top}{\sqrt{d}}\Big)V,\quad
K=[k_1\dots k_t],\ V=[v_1\dots v_t]
$$

**「吸收」是什么:把 $q\!\cdot\!k$ 代数重写成 $q'\!\cdot\!c_{KV}$。** 朴素地算注意力分数要先用 $W_{UK}$ 把每个历史 $c_{KV,j}$ 上投影成 $k_j$、再点积,等于对全部历史 token 各做一次上投影,白搬白算。注意 $k_j=W_{UK}c_{KV,j}$,把它代进分数:

$$
q_t^\top k_j = q_t^\top (W_{UK}\,c_{KV,j}) = \underbrace{(W_{UK}^\top q_t)}_{q_t'}{}^{\!\top} c_{KV,j} = q_t'^\top c_{KV,j}
$$

矩阵乘的结合律允许把 $W_{UK}$ **从 $k$ 侧"吸收"到 $q$ 侧**:推理时离线把 $W_{UK}$ 折进查询投影,在线只算一次 $q_t'=W_{UK}^\top q_t$(单个 query,极廉价),之后**直接拿 $q_t'$ 点压缩潜向量 $c_{KV,j}$**,全程不必把历史 $c_{KV}$ 还原成 $k$。于是从 HBM 搬的、参与点积的都是压缩态的 $c_{KV}$(512 维)而非展开的 $k$(几千维),带宽和算力双省——这就是 MLA 解码省钱的代数根源,也是 FlashMLA 内核要适配的核心。

关键在内存:**同一份 $c_{KV}$ 被所有头复用**，另有共享的 RoPE 分量。cache 可按页管理变长序列；但页大小是运行时和实现的配置，不能把 64 写成 FlashMLA 的普遍常量。不同通用 attention 路径对这种"压缩 + 共享 + 分页"布局的支持不同，不能断言一定"跑不了"；应核对内核契约、cache 格式和 benchmark。FlashMLA 的公开接口显式接收页表、各序列长度和调度元数据，以支持该类负载。

![[flash-FlashMLA数据流.png]]

![[flash-026MLA压缩KV手算.png]]

## 调用方式
```python
# FlashMLA README 所示的调用形状(版本/参数以安装包 README 为准)
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

# 变长序列:按每条序列的 KV 长度做调度元数据(负载均衡到各 SM)
is_fp8_kvcache = False
topk = None                    # dense decode 不提供 sparse indices
tile_metadata, num_splits = get_mla_metadata(
    cache_seqlens, s_q * h_q // h_kv, h_kv, h_q, is_fp8_kvcache, topk
)

out, _ = flash_mla_with_kvcache(
    q,                      # MTP 未启用时 s_q=1
    kv_cache,               # cache 格式必须和 is_fp8_kvcache 一致
    block_table,            # 每序列的页表
    cache_seqlens,
    dv,
    tile_metadata, num_splits,
    is_causal=True,
    is_fp8_kvcache=is_fp8_kvcache,
    indices=topk,
)
```

```text
❌ 未核对 cache schema、页表和 kernel 契约就把 MLA cache 喂给某个通用路径
✅ 用与模型和 cache 格式匹配的 FlashMLA 路径，并在目标 GPU、CUDA、dtype、batch、上下文长度下复跑仓库 benchmark
```

## 面试高频
- **FlashMLA 解决什么问题?** 解码常需读取历史 KV，而 MLA 的压缩、共享与分页 layout 需要匹配的 kernel / cache 契约。FlashMLA 为该类路径提供实现；是否受限于 HBM、以及带宽利用率，要看目标负载实测。
- **它和 FlashAttention-3 的关系?** 受 FA2/FA3 + CUTLASS 启发,但场景不同:FA3 主攻 compute-bound 的 prefill/训练(吃满 Tensor Core),FlashMLA 主攻 memory-bound 的 MLA 解码(吃满带宽)。
- **为什么 MLA 解码这么省?** 在上述 DeepSeek-V3 schema 中，512 维潜向量加 64 维 RoPE 分量共 576 个数，对传统 32768 个 K/V 数为 1.76%；这是 cache payload 账，不是端到端性能保证。
- **追问：为什么不能只说 512/32768=1.56%?** A:那只数了 $c_{KV}$，没有数 cache 中还需保存的 64 维 RoPE 分量。答题应先写 schema，再写数值格式和上下文长度。
- **分页 KV 的页大小一定是 64 吗?** A:不是普遍事实。页表的存在支持变长 cache 管理，但 block size 是实现/配置问题；面试中应查当前 backend 的接口而非把某个示例常量泛化。
- **为什么需要 get_mla_metadata?** 变长 batch 下各序列 KV 长度不同,要预先算调度元数据把工作均衡切到各 SM,避免长尾序列拖垮整批。

## 关键事实
- DeepSeek-V3 的公开实现配置包含 `kv_lora_rank=512` 与 `qk_rope_head_dim=64`；MLA cache 应将两部分共同计入账本。来源:DeepSeek-V3 Technical Report(2025 修订版)；`deepseek-ai/DeepSeek-V3` 官方实现(核验于 2026-07-18)。
- 截至 2026-07-18,`deepseek-ai/FlashMLA` README 列出 dense MLA decode 在 **H800 SXM5、CUDA 12.8** 的特定配置中最高 **3000 GB/s**(memory-bound)与 **660 TFLOPS**(compute-bound)。这是仓库声明的受限 benchmark，不是通用吞吐承诺；README 同时列出 SM90 / SM100、dense / sparse 和 BF16 / FP8 cache 路径。
- 对比:[[025 FlashAttention-3：Hopper TMA、WGMMA 与 FP8|FA3]] 与 FlashMLA 适用的 attention 形状和 cache layout 不同；选型应从模型 schema、硬件、CUDA、dtype、batch 与上下文开始，而不是只按名称归类。
