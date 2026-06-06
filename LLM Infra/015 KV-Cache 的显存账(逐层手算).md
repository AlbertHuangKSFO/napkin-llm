[[015 KV-Cache 的显存账(逐层手算)|KV-Cache 显存账]]是把 [[LLM/102 KV-Cache|KV-Cache]] 到底吃多少显存**逐层手算清楚**:KV 缓存为每一层、每个 KV 头、每个历史 token 各存一份 Key 和一份 Value 向量,总字节由一条乘法式决定,随序列长度和 batch **线性增长**。它是 [[014 Decode 阶段：访存受限|Decode 访存受限]]里每步要搬的"另一半字节",也是长上下文/高并发服务的头号显存瓶颈——理解这本账才能算清一张卡能塞多少并发、多长上下文。

## 直觉

KV-Cache 是一本**不断加页的台账**:每生成 1 个 token,就在每一层都新增一页(这一层这个 token 的 K 和 V)。模型有多少层,就有多少叠台账同时加页;上下文越长,每叠越厚;并发请求越多,台账副本越多。

工业类比:酒店每来一位客人(token),每个楼层(层)都要给他留一个固定大小的储物格(K+V)。楼层数固定、格子大小固定,所以总占用 = 楼层数 × 客人数 × 格子大小 × 并发栋数(batch)——纯线性,没有任何摊薄。这也是为什么长上下文"贵":KV 占用随长度直接涨。

[[LLM/102 KV-Cache|GQA]] 的省法就藏在式子里:让多个 Query 头共享少数 KV 头,KV 头数从 64 降到 8,这本账直接瘦 8 倍。

## 例子

**逐层手算 Llama-3 70B**:80 层,Q 头 64 个,**KV 头 8 个**(GQA),head_dim 128,BF16(2 字节/元素)。

单个 token、单条序列的 KV 字节(把每一层加总):

$$
\underbrace{2}_{K,V}\times \underbrace{80}_{层}\times \underbrace{8}_{KV头}\times \underbrace{128}_{head\_dim}\times \underbrace{2}_{字节} = 327680\ \text{B} = 320\ \text{KB / token}
$$

- **4k 上下文**:$320\text{ KB}\times 4096 = 1.25$ GB / 序列。batch 8 →**10 GB**。
- **32k 上下文**:$320\text{ KB}\times 32768 = 10$ GB / 序列(正好 4k 的 8 倍)。batch 8 →**80 GB**(≈ 占满一张 A100-80G,且这还在 140 GB 权重之外)。

**对比若用 MHA(64 KV 头)**:每 token 变 $320\text{ KB}\times 8 = 2.56$ MB,32k 单序列就 80 GB——GQA 把它压回 10 GB,这就是 70B 能跑长上下文的关键。

## 原理

通用 KV 显存公式(GQA 用 $H_\text{kv}$=KV 头数;MHA 时 $H_\text{kv}=H$):

$$
M_\text{KV} = 2 \cdot L \cdot H_\text{kv} \cdot d_h \cdot S \cdot B \cdot b
$$

- $L$=层数,$H_\text{kv}$=KV 头数,$d_h$=head_dim,$S$=序列长,$B$=batch,$b$=字节/元素。
- 因子 2 来自 K 和 V 各一份。$H_\text{kv}\cdot d_h$ 常等于"KV 隐维",GQA 下它远小于 $d_\text{model}$。

线性性一目了然:**对 $S$、$B$ 都是一次项**,没有平方、没有摊薄。每 token 增量恒为 $2 L H_\text{kv} d_h b$。

省字节的三个旋钮:降 $H_\text{kv}$([[LLM/102 KV-Cache|GQA/MQA]])、降 $b$(KV 量化 FP8/INT8 把 $b$ 从 2 砍到 1 甚至 0.5)、限 $S$(滑窗/驱逐)。一张卡能容纳的最大 (并发 × 上下文) 就由 $M_\text{KV}\le \text{显存} - \text{权重}$ 框定。

## 图

![[mem-KV显存堆叠账.svg]]

![[mem-015GQA压KV八倍.svg]]

## 代码

KV 显存计算器 + 一张卡能塞多少并发的反解,附 `❌vs✅`:

```python
def kv_bytes(layers, kv_heads, head_dim, seq_len, batch=1, dtype_bytes=2):
    # ✅ 正确：用 KV 头数(GQA)而不是 Q 头数
    return 2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes

# Llama-3 70B：80 层、8 KV 头、head_dim 128、BF16
per_tok = kv_bytes(80, 8, 128, 1)                    # 327680 B = 320 KB/token
gb_4k   = kv_bytes(80, 8, 128, 4096)  / 1e9          # ≈ 1.25 GB / 序列
gb_32k  = kv_bytes(80, 8, 128, 32768) / 1e9          # ≈ 10.0 GB / 序列

# 反解：80GB 卡、扣掉 140GB 权重不可能 → 需多卡；单卡 A100 只放 KV 时能塞多少 32k 序列
def max_concurrency(total_gb, weight_gb, layers, kv_heads, head_dim, seq_len):
    free = (total_gb - weight_gb) * 1e9
    return int(free // kv_bytes(layers, kv_heads, head_dim, seq_len))

# ❌ 错误：用 Q 头数 64 估 KV，结果膨胀 8 倍，把显存预算算爆、并发拍小
wrong = kv_bytes(80, 64, 128, 32768) / 1e9           # ≈ 80 GB（误以为 MHA）
# ✅ GQA 实际 8 个 KV 头 → 10 GB，差 8 倍直接决定"这卡到底能不能跑"
```

`❌` 拿 Q 头数(64)算 KV,GQA 模型会高估 8 倍,直接把容量规划算错;`✅` 一律用 **KV 头数** $H_\text{kv}$。

## 面试高频

- **Q:KV 显存公式默写?** $2\cdot L\cdot H_\text{kv}\cdot d_h\cdot S\cdot B\cdot b$;因子 2=K+V,务必用 **KV 头数**(GQA)。
- **Q:为什么长上下文这么吃显存?** KV 对序列长 $S$ **线性**增长且无摊薄;32k 比 4k 直接 8 倍。
- **Q:Llama-3 70B 在 32k、batch 8 要多少 KV?** ≈ 80 GB(单序列 10 GB),在 140 GB 权重之外,需多卡。
- **Q:怎么省 KV 显存?** GQA/MQA 降 KV 头、KV 量化降字节、PagedAttention 减碎片、滑窗/驱逐限有效长度、prefix 共享复用公共前缀。
- **Q:KV 和权重哪个先成瓶颈?** 短上下文权重为主;上下文一长,KV 反超权重成主瓶颈(故长上下文服务首先卡 KV)。

## 关键事实

- 每 token KV 增量 = $2 L H_\text{kv} d_h b$,**恒定**;Llama-3 70B(GQA,BF16)= **320 KB/token**。
- GQA(KV 头 8 vs Q 头 64)把 KV 显存压到 MHA 的 **1/8**,是 70B 长上下文可行的关键(Llama-3 全系采用,2024)。
- FP8 KV 量化(2024–2025 主流支持,如 vLLM/TensorRT-LLM)把 $b$ 从 2 砍到 1,KV 再省一半,基本不掉精度。
