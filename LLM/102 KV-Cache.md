[[102 KV-Cache|KV-Cache]] 是自回归推理的核心加速:由于因果注意力下**历史 token 的 Key/Value 永不改变**,生成每个新 token 时只需为这一个新 token 算 Q/K/V,把历史 K/V **缓存起来直接复用**,避免每步重算整段历史——使逐步解码的每步计算量从 $O(L^2)$ 降到 $O(L)$;代价是 KV 显存随序列长度 $L$ **线性增长**,长上下文下成主要瓶颈,引出 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]]。

## 直觉

自回归生成是"已有的 $t$ 个 token → 预测第 $t+1$ 个"。第 $t+1$ 步要算注意力:新 token 的 Query 去和**所有历史 token 的 Key**点积、再加权它们的 **Value**。

关键观察:[[007 因果掩码与 padding 掩码|因果掩码]]保证每个 token 只看自己和左边,所以**历史 token 的 K、V 一旦算出就固定不变**(它们的输入没变)。若不缓存,第 $t+1$ 步要把前 $t$ 个 token 的 K/V **全部重算一遍**——纯属浪费,而且步步累积成 $O(L^2)$。

KV-Cache 就是:**把每层、每个历史 token 的 K 和 V 存进显存**;每生成一步,只为**新来的那一个 token** 算它的 Q/K/V,把新 K/V **追加(append)**进缓存,然后新 Q 和缓存里的全部 K/V 做注意力。于是每步只算 1 个新列,而不是重算整个矩阵。

代价很现实:缓存要占显存,且**正比于序列长度 $L$**——上下文越长、并发请求越多,KV 显存越爆。这就是 [[019 GQA 分组查询注意力|GQA]]、[[026 PagedAttention 与 KV 分页|PagedAttention]]、KV 量化等优化要省的东西。

## 例子

**两个阶段**。给定 prompt "今天天气" 生成后续:

- **Prefill(预填充)**:一次性并行算完 prompt 全部 4 个 token 的 K/V,填满缓存。这步是计算密集(大矩阵乘)。
- **Decode(逐步解码)**:每次只生成 1 个新 token。第 5 步只算 "真"(新 token)的 Q/K/V,K/V 追加进缓存(现 5 列),新 Q 与 5 列 K/V 算注意力 → 出第 6 个 token。第 6 步只算第 6 个……**每步只新增 1 列**。这步是访存密集(读巨大缓存)。

**省了多少**。生成 $L=1000$ 个 token:
- 无缓存:第 $t$ 步重算 $t$ 列,总计算 $\propto 1+2+\dots+L=\frac{L(L+1)}{2}\approx 5\times10^5$ 列。
- 有缓存:每步只算 1 列,总计 $L=1000$ 列。**约省 500 倍**的重复 K/V 计算(注意力本身仍要扫历史)。

**$O(L^2)\to O(L)$ 是怎么来的(逐步推)**。把「生成完整 $L$ 个 token」的总投影计算量加起来:
- **无缓存**:第 $t$ 步要对前 $t$ 个 token 重新投影 K/V,代价 $\propto t$;总和 $\sum_{t=1}^{L}t=\frac{L(L+1)}{2}=O(L^2)$。
- **有缓存**:第 $t$ 步只投影 1 个新 token,代价 $\propto1$;总和 $\sum_{t=1}^{L}1=L=O(L)$。

所以**整段生成的投影/前馈计算从 $O(L^2)$ 降到 $O(L)$**。注意:注意力打分本身(新 Q 点积历史 K)第 $t$ 步仍是 $O(t)$、整段仍是 $O(L^2)$——KV-Cache 省掉的是「对历史的重复投影和前馈」,这部分才是大头(投影/FFN 占 Transformer 绝大多数 FLOPs),所以实际加速达数倍到数百倍。

![[infer-KVCache增量复用.svg]]

## 原理

**注意力回顾**。第 $t$ 步,新 token 隐状态 $h_t$ 投影出 $q_t=h_tW_Q,\ k_t=h_tW_K,\ v_t=h_tW_V$。注意力输出

$$
o_t = \mathrm{softmax}\!\Big(\frac{q_t K_{1:t}^\top}{\sqrt{d_k}}\Big)V_{1:t},\qquad
K_{1:t}=\begin{bmatrix}k_1\\\vdots\\k_t\end{bmatrix},\ V_{1:t}=\begin{bmatrix}v_1\\\vdots\\v_t\end{bmatrix}
$$

因果性 ⇒ $k_1,\dots,k_{t-1}$ 与 $v_1,\dots,v_{t-1}$ **与之前各步完全相同**。故缓存它们,本步只算 $k_t,v_t$ 并 append:$K_{1:t}=[\,K_{1:t-1};\,k_t\,]$。

**复杂度**。生成第 $t$ 个 token:
- 无缓存:重算 $K_{1:t},V_{1:t}$(投影 $O(t\cdot d^2)$),整段全过一遍 ⇒ 单步 $O(t)$、全程 $\sum_t O(t)=O(L^2)$。
- 有缓存:只投影 1 个新 token($O(d^2)$ 常数级),注意力打分仍需扫 $t$ 个历史 K($O(t\cdot d_k)$)⇒ 单步注意力 $O(t)$,但**省掉了对历史的重复投影/前馈**,整体每步从"重算整段"降到"只算新列",这是实际数倍到数百倍的加速来源。

**KV 显存**。缓存大小

$$
\text{KV bytes} = 2 \times n_{\text{layers}} \times L \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes}
$$

(2 = K 和 V;$\times$ batch)。**线性正比于 $L$**,且每层都存。长上下文(如 128K)或高并发时,KV 缓存常**超过模型权重本身**成为显存第一瓶颈。

**代进真实数字感受一下**(Llama-2-7B 类:$n_{\text{layers}}=32$、$n_{\text{heads}}=32$、$d_{\text{head}}=128$、FP16=2 字节):

$$
\text{每 token KV} = 2\times32\times32\times128\times2\ \text{字节} = 1\,\text{MB/token}
$$

所以**单条 32K 上下文的 KV 缓存 = 32 GB**——比 7B 模型权重(FP16 约 14 GB)还大一倍多!并发 10 条就是 320 GB,远超单卡。这就是为什么长上下文/高并发下 KV 缓存是**第一显存瓶颈**,也是 GQA/MQA、KV 量化、PagedAttention 存在的理由。用 [[019 GQA 分组查询注意力|GQA]] 把 KV 头从 32 降到 8(共享 4:1),每 token KV 直接降到 0.25 MB,4 倍缩减。

这直接催生:

- **少存 K/V 头**:[[018 MQA 多查询注意力|MQA]]、[[019 GQA 分组查询注意力|GQA]] 让多个 Q 头共享一组 K/V → 缓存按头数比例缩小。
- **更省的内存管理**:[[026 PagedAttention 与 KV 分页|PagedAttention]] 用分页避免预留连续大块、减碎片(vLLM)。
- **压位宽 / 驱逐 / 前缀共享**:见 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]]。

注意:KV-Cache **不改变数学结果**(等价于全量重算),纯粹是工程上的重复计算复用。

## 代码

```python
import numpy as np

def attn(q, K, V, dk):                              # 单头注意力,q:(d,) K,V:(t,d)
    s = (K @ q) / np.sqrt(dk)
    w = np.exp(s - s.max()); w /= w.sum()
    return w @ V

# ❌ 无缓存:每生成一步都重算全历史的 K/V(O(L^2) 重复)
def decode_no_cache(hs, Wk, Wv, Wq, steps):
    dk = Wq.shape[1]
    for _ in range(steps):
        K = hs @ Wk                                 # 每步对【全历史】重投影 —— 浪费
        V = hs @ Wv
        q = hs[-1] @ Wq
        o = attn(q, K, V, dk)
        hs = np.vstack([hs, o])                     # 把新输出当下一步输入(示意)
    return hs

# ✅ 有缓存:历史 K/V 只算一次并缓存,每步只追加新 token 的 K/V
def decode_kv_cache(hs, Wk, Wv, Wq, steps):
    dk = Wq.shape[1]
    K_cache = hs @ Wk                               # prefill:一次性算完 prompt 的 K/V
    V_cache = hs @ Wv
    out = hs
    for _ in range(steps):
        h_new = out[-1]                             # 只处理最新 1 个 token
        k_new, v_new = h_new @ Wk, h_new @ Wv       # 只算 1 个新列
        K_cache = np.vstack([K_cache, k_new])       # append,不重算历史
        V_cache = np.vstack([V_cache, v_new])
        q = h_new @ Wq
        o = attn(q, K_cache, V_cache, dk)           # 新 Q 对【缓存的】全部 K/V
        out = np.vstack([out, o])
    return out

np.random.seed(0)
d = 8; hs = np.random.randn(4, d)
Wq, Wk, Wv = (np.random.randn(d, d) for _ in range(3))
a = decode_no_cache(hs.copy(), Wk, Wv, Wq, 3)
b = decode_kv_cache(hs.copy(), Wk, Wv, Wq, 3)
print("结果一致:", np.allclose(a, b))               # True:缓存不改变数学结果
```

## 面试高频

- **KV-Cache 缓存的是什么?为什么能缓存?** 缓存每层、每个历史 token 的 Key 和 Value。因果注意力下历史 token 的 K/V 不随新 token 改变,故可复用,无需每步重算。
- **缓存的是 Q 吗?** 不是。每步只需当前新 token 的 Q(用完即弃),要缓存的是历史 **K 和 V**(下步还要被新 Q 注意)。
- **KV-Cache 把复杂度从多少降到多少?** 逐步解码每步从"重算整段历史"($O(L^2)$ 累积)降到只算 1 个新列;注意力打分仍需扫历史 $O(L)$,但省去对历史的重复投影,是主要加速来源。
- **KV 显存怎么算?为什么是瓶颈?** $2\times$层数$\times L\times$头数$\times$头维$\times$字节$\times$batch,**线性正比 $L$** 且每层都存;长上下文/高并发时常超过权重本身。
- **怎么省 KV 显存?** 减 K/V 头([[019 GQA 分组查询注意力|GQA]]/[[018 MQA 多查询注意力|MQA]])、量化 KV、分页减碎片([[026 PagedAttention 与 KV 分页|PagedAttention]])、驱逐旧 token、前缀共享——见 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]]。
- **Prefill 和 Decode 阶段有何不同?** Prefill 并行算完 prompt 全部 K/V(计算密集);Decode 每步只生成 1 token、读巨大缓存(访存密集)。两阶段瓶颈不同,常分开调度/拆分(PD 分离)。
- **KV-Cache 会改变输出吗?** 不会,数学上等价于全量重算,纯工程优化。
- **给个具体显存数字。** Llama-2-7B(32 层、32 头、头维 128、FP16):每 token KV ≈ 2×32×32×128×2 字节 = 1 MB;32K 上下文单条就 32 GB,比 7B 权重(约 14 GB)还大。并发越高越爆,故是长上下文第一瓶颈。GQA(KV 头降 4 倍)可把它砍到 0.25 MB/token。
- **为什么 Prefill 是 compute-bound、Decode 是 memory-bound?** Prefill 并行处理整段 prompt,是大矩阵乘(算术强度高、吃算力);Decode 每步只算 1 个 token 但要把整个 KV 缓存和权重从显存读一遍(算术强度极低、吃带宽)。所以两阶段瓶颈不同,催生 PD 分离调度。见 [[078 推理算力、吞吐与延迟、Roofline|Roofline]]。
- **KV-Cache 缓存 Q 吗?为什么?** 不缓存。每步只用当前新 token 的 Q 去注意历史,用完即弃;要被后续步反复访问的是历史 K/V,所以只缓存 K 和 V。
- **量化 KV-Cache 有什么坑?** KV 量化到 int8/int4 能成倍省显存,但 K 比 V 更敏感(K 参与 softmax 打分,误差被指数放大),常对 K 用更高精度或更细粒度;且 KV 量化是在线动态的,要平衡精度与解量化开销。详见 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]]。
- **多条 beam / 多请求的 KV 怎么管?** 每个序列各自一份 KV;PagedAttention 用分页(像 OS 虚拟内存)按块分配、避免预留连续大块和碎片,还能让相同前缀(系统提示)的多请求**共享**前缀 KV 块,省显存。

## 关键事实

- KV-Cache 是 decoder-only 自回归推理的标配:因果掩码([[007 因果掩码与 padding 掩码|因果掩码]])保证历史 K/V 不变,故可缓存复用。
- 缓存的是 **K 和 V**(每层各一份),不缓存 Q;显存 $=2\times$层数$\times L\times$头数$\times$头维$\times$精度字节$\times$batch,随 $L$ **线性增长**。
- 解码分两阶段:**Prefill**(并行填 prompt 的 KV,计算密集)与 **Decode**(逐 token,访存密集);现代引擎常做 PD 分离与连续批处理(vLLM)。
- KV-Cache 把逐步解码的每步从"重算整段历史"降为"只算新列",是数倍至数百倍的实际加速;但带来线性显存开销,催生 [[019 GQA 分组查询注意力|GQA]]、[[026 PagedAttention 与 KV 分页|PagedAttention]]、KV 量化等优化([[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]])。
- 与 [[108 推理引擎：vLLM、TensorRT-LLM、llama.cpp、SGLang|推理引擎]]、[[104 投机解码与 Medusa、Lookahead|投机解码]] 配合:前者高效管理 KV 缓存,后者一次验证多个候选 token 进一步提速。
