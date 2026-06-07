[[051 Mooncake：KVCache 中心解耦架构|Mooncake]](Qin 等,Moonshot AI × 清华,FAST 2025 最佳论文,arXiv 2407.00079)是 Kimi 的生产级服务平台。它把 [[048 为何分离 prefill 与 decode|PD 分离]] 再推一步:不只分 [[013 Prefill 阶段：计算受限|Prefill]]/[[014 Decode 阶段：访存受限|Decode]] 集群,更以一个**分布式 [[LLM/102 KV-Cache|KV-Cache]] 池**为系统中心——复用 GPU 集群里被忽视的 CPU、DRAM、SSD、NIC 资源,把 KV 既当缓存又当共享存储。调度由 conductor 统管,核心是在最大化达标吞吐与守 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]] 之间平衡;面对真实**过载**场景,用基于预测的**早拒**保住 SLO。

## 直觉

DistServe/Splitwise 是"两条专线",Mooncake 则在两条专线中间架了一个**中央仓库(KVCache 池)**。prefill 算出的 KV 不是点对点直送 decode,而是先入库;decode、甚至后续命中相同前缀的新请求,都从仓库取货。这把"传 KV"升级成"**存 + 共享 KV**",顺带做了跨请求前缀复用。仓库还不用买新地——它把 GPU 机器里闲着的 CPU 内存、SSD、网卡拼成存储,等于用废料盖仓库。

过载是 Mooncake 区别于学术系统的关键:真实流量会超机器上限,这时与其让所有请求都变慢、全体超 SLO,不如**提前预测并拒掉**注定无法达标的请求,守住其余请求的 SLO——"早拒"是运营现实倒逼出的设计。

## 例子

论文/生产报告的量级:

- 长上下文场景吞吐 **最高 +525%**(KVCache 复用 + 分离的合力)。
- 相比此前系统,Kimi 在 **A800 上多扛 115%、H800 上多扛 107%** 的请求。
- 在严格 SLO 下,早拒策略让 Kimi **多处理 75% 请求**(而非让全体超时)。
- 存储复用例子:一台 8×H800 节点,GPU HBM 之外还有几百 GB DRAM + 数 TB NVMe SSD 长期闲置;Mooncake 把它们组成 KVCache 池,长对话的历史 KV 不必每轮重算,prefill 命中即跳过。

## 原理

Mooncake 的调度目标(conductor)在过载约束下最大化 goodput:

$$
\max\ \text{Goodput}\quad\text{s.t.}\quad
\text{TTFT}\le\tau_{\text{ttft}},\ \text{TPOT}\le\tau_{\text{tpot}},\ \text{load}\le \text{cap}
$$

当 $\text{load} > \text{cap}$ 不可行时,引入**预测早拒**:对到达请求估其完成所需资源与时间,若接纳会导致已有请求集体违约,则提前拒绝。KVCache 复用把 prefill 计算量从"每次重算"降为"只算未命中前缀":

$$
\text{FLOPs}_{\text{prefill}} \propto L_{\text{prompt}} - L_{\text{cached prefix}}
$$

命中率越高,prefill 越省 → 这正是"用更多存储换更少计算"(论文副标题:Trading More Storage for Less Computation)。

## 图

![[disagg-Mooncake架构.png]]

![[disagg-051KVCache池复用.png]]

![[disagg-051预测性早拒.png]]

## 代码

Mooncake 开源了 Transfer Engine 与 Mooncake Store;下面示意"以 KVCache 池为中心"的接入(KV 写入共享池、跨请求复用):

```python
# ❌ 纯点对点 PD:KV 只从本次 prefill 传给本次 decode,前缀无法跨请求复用
# decode.recv_kv(prefill.compute_kv(prompt))

# ✅ Mooncake:KV 进分布式池,既供本请求 decode,也供后续前缀命中
from mooncake.store import MooncakeStore
store = MooncakeStore(backends=["dram", "nvme"])   # 复用闲置 CPU/DRAM/SSD

def serve(prompt):
    prefix_kv = store.lookup(prompt)               # 跨请求前缀命中?
    new_kv = prefill.compute(prompt, reuse=prefix_kv)  # 只算未命中部分
    store.put(prompt, new_kv)                       # KV 入池,供他人复用
    return decode.generate(from_kv=new_kv)          # decode 集群续生成

# conductor 过载时:预测接纳后是否全体违约 → 早拒
if conductor.predict_violation(request):            # ✅ 守住其余请求 SLO
    reject(request)
```

`❌` 点对点只服务当前请求、丢失复用机会,且过载时一视同仁拖垮所有人;`✅` KV 进共享池被跨请求复用、用闲置存储扩容,conductor 预测性早拒保 goodput。

## 面试高频

- **Q:Mooncake 和 DistServe 的核心区别?** Mooncake 是 **KVCache 中心**:不仅分 prefill/decode 集群,还把 KV 放进一个复用 CPU/DRAM/SSD/NIC 的**分布式池**,支持跨请求前缀复用;DistServe 偏点对点 + 并行搜索。
- **Q:什么是早拒(early rejection)?为什么需要?** 真实流量会过载,Mooncake 用**预测**判断接纳新请求是否会导致已有请求集体违约,若是则提前拒绝,守住其余请求 SLO——"trade storage for computation" 之外的运营现实设计。
- **Q:Mooncake 的代表数字?** 长上下文吞吐 **最高 +525%**;A800/H800 多扛 **115%/107%** 请求;严 SLO 下多处理 **75%** 请求。FAST 2025 最佳论文。
- **Q:"用存储换计算"是什么意思?** KV 多存(DRAM/SSD),换 prefill 少算(命中前缀就跳过重算);存储便宜、算力贵,长对话场景尤其划算。

## 关键事实

- **Mooncake**,Qin 等,**Moonshot AI × 清华**,arXiv **2407.00079**,**FAST 2025 最佳论文**;是 **Kimi** 的生产服务平台。
- 三大支柱:① KVCache 中心的解耦(prefill/decode 集群 + 分布式 KV 池);② 复用 GPU 集群闲置 **CPU/DRAM/SSD/NIC**;③ conductor 调度 + **预测性早拒**应对过载。
- 数字:长上下文 **最高 +525% 吞吐**;A800 **+115%**、H800 **+107%** 请求量;严 SLO 下 **+75%** 请求。Transfer Engine 与 Mooncake Store 已开源,被 [[053 KV 传输：NIXL、点对点与带宽|KV 传输]] 生态广泛引用。
