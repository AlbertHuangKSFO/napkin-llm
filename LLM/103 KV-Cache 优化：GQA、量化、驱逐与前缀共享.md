[[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享]]:在不改模型质量的前提下压缩 [[102 KV-Cache|KV-Cache]] 显存的四类手段——**减头(GQA)、降精度(量化)、丢 token(驱逐)、共享前缀(prefix caching)**,分别从 KV 体积公式的不同维度下手,是把 decode 阶段 batch 开大、提吞吐的关键工程。

## ① 直觉:KV-Cache 是 decode 的显存大户,从四个维度去砍

回忆 [[102 KV-Cache|KV-Cache]]:自回归生成时,每个已处理 token 的 K、V 都缓存起来,避免重算。它的体积是

$$
\text{KV 字节} = 2 \times L \times H_{kv} \times d_{head} \times T \times b_{prec}
$$

(2 是 K 和 V,$L$ 层数,$H_{kv}$ KV 头数,$d_{head}$ 每头维度,$T$ 序列长度,$b_{prec}$ 每元素字节)。长上下文、大 batch 时,它能轻松超过模型权重本身,直接卡死 batch 上限——而 decode 是[[078 推理算力、吞吐与延迟、Roofline|访存受限]]的,batch 开不大就等于吞吐上不去。

四类优化各砍一个因子:

- **GQA**:砍 $H_{kv}$。多个 Query 头共享一组 KV 头。
- **量化**:砍 $b_{prec}$。FP16 → INT4/2bit。
- **驱逐**:砍 $T$。只保留重要 + 近窗 token,KV 近乎定长。
- **前缀共享**:砍「重复的 $T$」。多请求共享的前缀 KV 只存一份。

四者**正交**,可叠加:GQA + KV 量化 + prefix caching 是当下生产系统的标配组合。

**一个常被忽略的第五维:头维度 $d_{head}$ 与「潜空间压缩」。** 体积公式里还有 $d_{head}$ 这一项,DeepSeek 的 **MLA(Multi-head Latent Attention)** 正是从这维下手——不缓存完整的 K、V,而是缓存一个**低秩潜向量** $c$(维度远小于 $H_{kv}d_{head}$),用时再上投影还原 K、V。它把 KV 压到比 GQA 还小(DeepSeek-V2 报告 KV 压缩到 MHA 的约 1/14),且**质量优于 GQA**(因为低秩压缩比「直接砍头」保留更多信息)。详见 [[020 MLA 多头潜在注意力(DeepSeek)|MLA]]。所以严格说优化维度有五条:**减头(GQA)、降秩(MLA)、降精度(量化)、丢 token(驱逐)、去重复(前缀共享)**——前两条改架构、后三条纯推理期。

![[infer-MLA降秩vsGQA减头.png]]

**为什么 decode 阶段 KV 这么致命(再算一笔)。** decode 每生成 1 token,要把**全部模型权重 + 全部已缓存 KV** 从显存搬进计算单元,但只算 1 个 token 的注意力。算术强度(FLOP/Byte)极低,卡在访存带宽上(见 [[078 推理算力、吞吐与延迟、Roofline|Roofline]])。把 batch 开大是唯一的解法:$B$ 条请求共摊**同一次权重读取**,吞吐近似 $\propto B$。而 $B$ 的上限 = $\frac{\text{剩余显存}}{\text{单请求 KV 体积}}$——所以**压 KV 直接等于放大可用 batch、放大吞吐**。这就是本篇四类优化的终极动机。

## ② 例子:一条 32K 上下文请求的 KV 账单

设 LLaMA-2-13B:$L=40$、原始 $H=40$ 头、$d_{head}=128$、FP16($b=2$)、$T=32768$。

**无优化(MHA)**:

$$
2 \times 40 \times 40 \times 128 \times 32768 \times 2 \approx 27\text{ GB}
$$

一条请求就吃掉 27GB——单张 A100(80GB)放不下几条。逐项优化:

- **GQA(8 组)**:$H_{kv}: 40 \to 8$,KV ÷5 → **约 5.4 GB**。
- **再 KV 量化到 INT4**:$b: 2 \to 0.5$,再 ÷4 → **约 1.35 GB**。
- **再 StreamingLLM 驱逐**(只留 4 sink + 2K 窗 ≈ 2052 token):$T: 32768 \to 2052$,再 ÷16 → **约 85 MB**。

从 27GB 到 85MB,**约 320 倍**。当然驱逐会丢中段信息(有损),量化和 GQA 基本无损;实际按任务取舍。

**换个角度:能并发几条?** 设 A100-80G,扣掉 26GB 权重(13B 模型 FP16)后剩约 54GB 给 KV。
- 无优化(27GB/条):只能并发 **2 条**,batch 卡死、吞吐惨。
- GQA(5.4GB/条):并发 **10 条**,吞吐 ×5。
- GQA+INT4(1.35GB/条):并发 **40 条**,吞吐再 ×4。

可见 KV 压缩不是「省一点显存」,而是**直接决定能开多大 batch、跑多高吞吐**——同一张卡,优化后服务能力差一个数量级。这就是为什么所有生产引擎都把 KV 优化放在第一优先级。

![[infer-KV-Cache四类优化.png]]

## ③ 原理:四类各自的机制与代价

**① GQA(分组查询注意力)。** 详见 [[019 GQA 分组查询注意力|GQA]]。把 $H$ 个 Query 头分成 $G$ 组,每组共享一份 K、V。KV 体积 ÷ $(H/G)$。这是**训练时就定死的架构选择**(MHA→GQA→[[018 MQA 多查询注意力|MQA]] 是连续谱),不是推理期才加的,但它从根上决定了 KV 大小。LLaMA-2-70B、Qwen、Mistral 全用 GQA。

**② KV 量化。** 把缓存的 K、V 从 FP16 量化到 INT8/INT4/2bit 存储,注意力计算时反量化。关键发现(KIVI, 2024):**Key 应按通道(per-channel)量化、Value 应按 token(per-token)量化**——因为 Key 的某些通道有大 outlier(按通道分组才能隔离),Value 没有这种结构。2bit 下峰值显存 ÷2.6、batch 可 ×4,质量几乎无损。注意这与[[095 GPTQ|权重量化(GPTQ)]]不同:那是压模型权重,这是压 KV 激活缓存。

量化的几个易错点:① **反量化要在注意力 GEMM 前做**,所以量化省的是「存储」与「访存带宽」,不直接省计算——但 decode 是访存受限的,省带宽就是提速。② **scale/zero-point 也要存**(每组一个),组分得越细(粒度小)精度越高但元数据开销越大,2bit 下常用 group=128 平衡。③ **首尾 token 和 sink token 对量化误差更敏感**,有的实现对前几个 token 保留高精度。④ KV 量化与**权重量化正交**:可以权重 INT4(GPTQ/AWQ)+ KV INT4 同时上。常见组合 KV8(INT8,几乎零损、最稳妥)是生产默认;KV4 激进些,长上下文收益大但需测质量。

**③ 驱逐(eviction)。** 不是所有历史 token 都重要——观察发现注意力高度稀疏,少数 token 贡献绝大部分注意力(系统侧的驱逐与压缩实现见 [[LLM Infra/035 KV 驱逐与压缩：H2O 与注意力汇|KV 驱逐与压缩]])。两条主流策略:

- **H2O(Heavy-Hitter Oracle, 2023)**:按**累计注意力分数**动态保留「重击者(heavy hitter)」+ 最近 token,丢弃低分历史。20% 预算即可,吞吐最高 ×29。
- **StreamingLLM(2023)**:留**前 4 个 token(attention sink)+ 滑动窗口**。核心洞察:softmax 强制注意力和为 1,模型把多余注意力倾倒到最初几个 token(几乎与语义无关,纯当「泄洪口」)。删掉这些 sink,分布失稳、困惑度爆炸;留住它们,可稳定流式处理到 400 万 token。详见 [[107 长上下文推理：YaRN、位置插值与 StreamingLLM|长上下文推理]]。
- **SnapKV / PyramidKV(2024)**:观察到不同层注意力稀疏度不同——**浅层关注局部、深层才聚焦少数关键 token**,于是按层分配不同的 KV 预算(金字塔式:深层留得少),比一刀切的统一预算更省、更准。这类「层级感知驱逐」是 H2O 之后的精细化方向。

**驱逐的根本代价(必须讲清)**:一旦某 token 的 KV 被丢,它就**永久消失**,后续再也无法回看——这是有损且不可逆的。所以驱逐适合「近期信息主导」的任务(闲聊、流式日志),不适合「需要回看全文」的任务(长文档 QA、代码库理解)。H2O 这类「按注意力分数留」比 StreamingLLM「按位置留」更聪明,但也更可能误删——因为「过去注意力高」不代表「未来还需要」(贪心驱逐的固有风险)。

![[infer-StreamingLLM驱逐.png]]

**④ 前缀共享(prefix caching)。** 多请求常共享前缀:同一个长 system prompt、few-shot 示例、RAG 检索拼接的文档、或并行采样从同一前缀分叉。这些公共前缀的 KV **算一次、存一份**,多请求共指(写时复制,见 [[026 PagedAttention 与 KV 分页|PagedAttention]] 的 copy-on-write)。既省显存又省去重复 prefill 计算。这是 vLLM 的 prefix caching、SGLang 的 [[108 推理引擎：vLLM、TensorRT-LLM、llama.cpp、SGLang|RadixAttention]] 的核心价值——聊天/RAG/Agent 这类「共享前缀 + 短续写」场景收益巨大。

前缀共享有两层省:**省显存**(KV 只存一份)和**省计算**(prefill 不重算)。后者尤其值钱——一个 8K 的 system prompt,每来一条请求若都重新 prefill,就是 8K token 的算力白烧;命中前缀缓存则直接跳过,TTFT(首 token 时延)从几百毫秒降到几乎为零。关键工程点:① 前缀必须**逐 token 完全一致**(连空格、模板都不能差),所以对话模板要规范化;② 缓存用 **LRU / 引用计数**管理,共享前缀被所有引用方释放后才回收;③ 命中是「最长公共前缀」——SGLang 的前缀树能在**任意分叉点**共享(比 vLLM 早期「整段哈希」粒度更细)。一个反直觉点:把「变化的部分」(如用户问题)放在 prompt **末尾**、把「固定的部分」(system + 文档)放**开头**,前缀命中率最高——这是 prompt 工程里实打实影响成本的设计。

## ④ 代码:KV 体积估算 + KV 量化 + StreamingLLM 驱逐

```python
import torch

def kv_bytes(L, H_kv, d_head, T, batch=1, prec_bytes=2):
    """KV-Cache 字节数 = 2(K,V) × L × H_kv × d_head × T × batch × 精度。"""
    return 2 * L * H_kv * d_head * T * batch * prec_bytes

# LLaMA-2-13B, 32K 上下文
print(kv_bytes(40, 40, 128, 32768) / 1e9, "GB")   # ≈27  MHA
print(kv_bytes(40,  8, 128, 32768) / 1e9, "GB")   # ≈5.4 GQA(8 组)
print(kv_bytes(40,  8, 128, 32768, prec_bytes=0.5) / 1e9, "GB")  # ≈1.35 +INT4

# ❌ 错:Key、Value 都按 token 量化 —— Key 的通道 outlier 被混进同组,精度崩
# ✅ 对(KIVI):Key 按通道、Value 按 token,分别量化
def quant_per_channel(k):           # Key: [T, d] 按 d(通道)求 min/max
    mn = k.min(0, keepdim=True).values; mx = k.max(0, keepdim=True).values
    scale = (mx - mn) / 15                       # INT4: 16 级
    q = ((k - mn) / scale).round().clamp(0, 15)
    return q, mn, scale
def quant_per_token(v):             # Value: [T, d] 按 T(token)求 min/max
    mn = v.min(1, keepdim=True).values; mx = v.max(1, keepdim=True).values
    scale = (mx - mn) / 15
    q = ((v - mn) / scale).round().clamp(0, 15)
    return q, mn, scale

# StreamingLLM 驱逐:只保留 sink + 近窗的 KV 索引
def streaming_keep(T, n_sink=4, window=2048):
    sink = list(range(min(n_sink, T)))
    recent = list(range(max(n_sink, T - window), T))
    return sink + recent            # 其余位置的 K、V 直接丢弃,显存近乎定长

# H2O 驱逐:按累计注意力分数留「重击者」+ 近窗(内容相关,比固定位置聪明)
def h2o_keep(attn_scores, T, budget=512, recent=128):
    # attn_scores: [T] 每个 token 收到的累计注意力(所有 query 对它的注意力之和)
    recent_idx = set(range(max(0, T - recent), T))        # 近窗一定留
    heavy = sorted(range(T), key=lambda i: -attn_scores[i])  # 按累计注意力降序
    keep = set(recent_idx)
    for i in heavy:                                        # 补「重击者」直到预算用满
        if len(keep) >= budget: break
        keep.add(i)
    return sorted(keep)             # 与 StreamingLLM 区别:留谁取决于注意力,不只看位置

# 组合策略:可叠加,体积是各倍数连乘
def kv_after_all(L, H, d, T, batch=1):
    base = kv_bytes(L, H, d, T, batch)               # MHA FP16 基线
    gqa  = base / (H / 8)                             # GQA 8 组:÷(H/8)
    int4 = gqa / 4                                    # KV INT4:÷4(2/0.5)
    evict = int4 / (T / (4 + 2048))                   # 驱逐到 sink+窗:÷(T/2052)
    return base, gqa, int4, evict                     # 四者正交、连乘
```

## 面试高频

- **Q:KV-Cache 为什么是推理瓶颈?怎么压?** decode 访存受限,KV 占显存大头、限制 batch。四类压法:GQA 减 KV 头、量化降精度、驱逐丢 token、前缀共享去重复;前两者基本无损,后两者按场景取舍。
- **Q:GQA 和 KV 量化分别砍体积公式哪一项?** GQA 砍 $H_{kv}$(头数),KV 量化砍 $b_{prec}$(每元素字节);两者正交可叠。
- **Q:KIVI 为什么 Key 按通道、Value 按 token 量化?** Key 的某些通道存在大 outlier,按通道分组才能把它隔离不污染其他通道;Value 无此结构,按 token 量化即可,2bit 近乎无损。
- **Q:StreamingLLM 为什么必须保留最初几个 token?** 它们是 attention sink——softmax 把多余注意力倾倒于此;删掉后注意力分布失稳、困惑度爆炸。光留滑窗会崩,sink+滑窗才稳。
- **Q:H2O 和 StreamingLLM 都是驱逐,区别?** H2O 按累计注意力分数动态留「重击者」(内容相关);StreamingLLM 固定留头部 sink + 近窗(位置固定、更简单、适合无限流)。
- **Q:prefix caching 在什么场景收益最大?** 多请求共享长前缀:同一 system prompt、few-shot、RAG 文档、并行采样分叉——聊天、RAG、Agent。前缀 KV 算一次存一份,省显存 + 省重算。
- **Q:MLA 和 GQA 都压 KV,谁更强?** MLA 缓存低秩潜向量(降秩,砍 $d_{head}$ 维度),GQA 共享 KV 头(减头,砍 $H_{kv}$)。MLA 压得更狠(DeepSeek-V2 约 ÷14)且质量更好——低秩压缩比直接砍头保留更多信息。但 MLA 实现更复杂、需配套训练。两者都是训练期架构选择,不是推理期加的。
- **Q:压 KV 到底为了什么?** 为了开大 batch。decode 访存受限,batch 越大权重读取摊得越薄、吞吐越高;而 batch 上限 = 剩余显存 ÷ 单请求 KV。压 KV = 放大 batch = 放大吞吐,同卡服务能力可差一个数量级。
- **Q:四类优化哪些无损、哪些有损?** GQA(架构,基本无损)、MLA(架构,质量甚至更好)、量化(KV8 近无损、KV4 轻损)基本不丢信息;驱逐(丢 token)是**有损且不可逆**——丢了就回不来,只适合近期信息主导的任务。前缀共享完全无损(只是去重)。
- **Q:KV8 和 KV4 怎么选?** KV8(INT8)几乎零质量损失,是生产稳妥默认;KV4 更激进,长上下文/大 batch 收益大但需实测质量,对 sink 和首尾 token 敏感,常配 per-channel(Key)/per-token(Value)分量化。

## 关键事实

- KV-Cache 体积 $=2 L H_{kv} d_{head} T \cdot b_{prec}$;长上下文 / 大 batch 时常超过模型权重,是 decode 阶段 batch 上限的硬约束(vLLM 文档,2023)。
- GQA 在 MHA 与 MQA 之间插值,KV 头数 ÷$(H/G)$,LLaMA-2-70B 等普遍采用(Ainslie et al., 2023, arXiv:2305.13245,见 [[019 GQA 分组查询注意力]])。
- KIVI:Key 按通道、Value 按 token 的免微调 2bit 非对称量化,峰值显存 ÷2.6、batch ×4、吞吐 2.35–3.47×,质量近乎无损(Liu et al., ICML 2024, arXiv:2402.02750)。
- H2O(Heavy-Hitter Oracle):按累计注意力保留 heavy hitter + 近窗,20% 预算下吞吐最高 ×29、延迟 ×1.9(Zhang et al., NeurIPS 2023, arXiv:2306.14048)。
- StreamingLLM:保留前几个 attention sink + 滑窗,免微调稳定流式处理至 400 万 token;attention sink 源于 softmax 对初始 token 的强注意力(Xiao et al., 2023, arXiv:2309.17453)。
- 前缀共享(prefix caching)用写时复制让多请求共享前缀 KV,vLLM 的 Automatic Prefix Caching、SGLang 的 RadixAttention 均基于此(Kwon et al., 2023, arXiv:2309.06180;Zheng et al., 2024, arXiv:2312.07104)。
- MLA(Multi-head Latent Attention):缓存低秩潜向量替代完整 KV,DeepSeek-V2 将 KV 压至 MHA 的约 1/14 且质量优于 GQA,是「降秩」这一维度的代表(DeepSeek-AI, 2024, arXiv:2405.04434,见 [[020 MLA 多头潜在注意力(DeepSeek)]])。
- 层级感知驱逐(SnapKV/PyramidKV, 2024):不同层注意力稀疏度不同,按层分配 KV 预算(深层更省),比统一预算更优(Li et al., SnapKV, 2024, arXiv:2404.14469)。
- KV 量化粒度:Key per-channel(隔离通道 outlier)、Value per-token;KV8(INT8)近无损为生产默认,KV4 长上下文收益大但需测质量;与权重量化(GPTQ/AWQ)正交可叠加。
- 四类优化的本质都是放大 batch:decode 访存受限,吞吐近似正比于 batch,而 batch 上限受单请求 KV 体积制约;压 KV 一个数量级即放大服务能力一个数量级。
