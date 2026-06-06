[[036 KV 分层 offload：GPU、CPU、SSD(LMCache)]] 是把 KV-Cache **无损地**从 GPU 显存「沉」到更便宜更大的存储层(CPU DRAM → 本地 SSD → 远端盘/Redis)的系统手段:用得着就快速载回 HBM,用不着就让位给活跃请求。代表实现 **LMCache** 把 vLLM/SGLang 产生的 KV 抽出 GPU、按存储层级分级存放,并**跨请求、跨引擎复用前缀**。它和 [[035 KV 驱逐与压缩：H2O 与注意力汇|KV 驱逐]] 互补——驱逐是**有损丢弃**省 T 维,offload 是**无损搬走**换大容量;也是 [[032 前缀缓存：RadixAttention 树结构|前缀缓存]] 突破单卡显存上限后的自然延伸,并为 [[051 Mooncake：KVCache 中心解耦架构|Mooncake 架构]]、[[038 KV-aware 路由与跨引擎复用|KV-aware 路由]] 提供存储底座。

## 直觉类比
显存(HBM)像电脑的**内存条**:极快但很贵很小。当你开太多「文档」(请求的 KV),内存放不下。offload 就像操作系统的**分层缓存 + 交换**:刚用过的文档先丢到「内存条之外的 DRAM」(CPU),再不用就落到 **SSD**,冷门的丢到**远端网盘 / Redis**(全集群共享)。要用时按热度从最近的层取回。区别于「驱逐」直接把文档**删了**——offload 是**搬走但不删**,所以无损。

## 小数字例子
设一段 32K token 的长文档,其 KV 折算 ~16 GB。单张 80 GB 卡只想留活跃请求,不愿被一份长文档占住。
- 这份 KV 写到 **CPU DRAM**(假设 ~25 GB/s):搬 16 GB ≈ 0.64 s;但若该前缀被后续 100 个请求复用,省下的是 **100 次 prefill 重算**(每次几百毫秒到数秒)→ 净赚巨大。
- 若落 **本地 NVMe SSD**(~3 GB/s):载回 ≈ 5.3 s,适合复用频率没那么高、但仍想避免重算的温数据。
- 落 **Redis / 远端**:网络受限更慢,但全集群任意节点可命中——配合 KV-aware 路由,命中率才是收益关键。
核心权衡:**搬运延迟 vs 省下的 prefill 重算**。只要复用率够高、搬运用流水线掩盖,offload 就划算。

## 原理:为什么要把 KV 搬出 GPU
用户侧累积的 KV 总量增长极快,**远超 GPU 显存容量**;同一个 system prompt / RAG 文档被海量请求共用,重复 prefill 是纯浪费。LMCache 的做法:

1. **抽取**:把引擎(vLLM/SGLang)生成的 KV 从 GPU 显存里取出。
2. **分级存放**:CPU 内存 → 本地盘 → 远端盘 → Redis,形成存储金字塔。
3. **跨请求/跨引擎复用**:既支持 offload(跨查询前缀复用),也支持 [[048 为何分离 prefill 与 decode|PD 解耦]] 的跨引擎/跨 GPU KV 传输。
4. **高效搬运**:批量数据移动 + **compute/I-O 流水线**(搬 KV 与算 GEMM 重叠),走 Ethernet / RDMA / NVLink。

设命中前缀长度 $L_{hit}$、未命中需重算长度 $L_{miss}$,offload 的净收益近似:

$$
\Delta T \approx \underbrace{C_{\text{prefill}}\cdot L_{hit}}_{\text{省下的重算}}\;-\;\underbrace{\frac{S_{KV}(L_{hit})}{B_{\text{layer}}}}_{\text{搬运回 HBM 代价}}
$$

只要命中前缀够长、所在层带宽 $B_{\text{layer}}$ 够高(或被流水线掩盖),$\Delta T>0$ 即划算。

![[kv-分层offload存储金字塔.svg]]

![[kv-036offload对比驱逐手算.svg]]

## 配置 / 代码
```python
# vLLM + LMCache:把 KV offload 到 CPU/磁盘并跨请求复用前缀(配置示意)
# pip install lmcache; 通过环境变量 + KV connector 接入
import os
os.environ["LMCACHE_CHUNK_SIZE"]      = "256"      # KV 分块粒度(token)
os.environ["LMCACHE_LOCAL_CPU"]       = "True"     # 启用 CPU DRAM 层
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "40"    # CPU 缓存上限(GB)
os.environ["LMCACHE_LOCAL_DISK"]      = "file:///mnt/nvme/lmcache"  # 本地盘层
os.environ["LMCACHE_REMOTE_URL"]      = "redis://kv-pool:6379"      # 远端/Redis 层

from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          kv_transfer_config={"kv_connector": "LMCacheConnector",
                              "kv_role": "kv_both"})  # 既存又取
```

```text
❌ 只靠 GPU 显存放 KV:容量撑爆 → 要么 OOM,要么每次都重算公共前缀,prefill 算力白烧
✅ 分层 offload(CPU/SSD/Redis)+ 跨请求前缀复用:无损保住 KV,命中即取,prefill 重算省掉
```

## 面试高频
- **offload 和驱逐有什么本质区别?** 驱逐是**有损**——丢掉 token 的 KV 省 T 维;offload 是**无损**——把 KV 搬到更大更慢的层,要用再载回。前者省显存可能掉点,后者不掉点但吃搬运带宽/延迟。
- **为什么要分层?** 存储有速度/容量/价格金字塔:HBM 快小贵、CPU DRAM 中、SSD 大慢、远端/Redis 最大最慢但全集群共享。按 KV 热度放对层,均摊成本。
- **offload 的收益从哪来?** 主要是**省 prefill 重算**:公共 system prompt / RAG 文档只算一次,后续命中直接取 KV → TTFT 大降。复用率是关键。
- **搬运延迟怎么掩盖?** compute/I-O 流水线:取 KV 与计算重叠;批量搬运;走 RDMA/NVLink 高速网络。
- **LMCache 和 PagedAttention 是一回事吗?** 不是。PagedAttention 解决**单卡内**显存碎片;LMCache 解决 KV **超出 GPU 显存**后跨层存放与跨请求/跨引擎复用,是更上层的 KV 缓存层。

## 关键事实
- **LMCache**,Cheng/Liu et al. **2025**(arXiv 2510.09665,`LMCache/LMCache` 开源):首个高效开源 KV 缓存层,从 vLLM/SGLang 抽取 KV 出 GPU,跨引擎跨查询共享。
- 存储层级:**CPU 内存 → 本地盘 → 远端盘 → Redis**;搬运网络:**Ethernet / RDMA / NVLink**。
- 两大能力:**cache offloading**(跨查询前缀复用)+ **PD disaggregation**(跨引擎/GPU 的 KV 传输)。
- 性能来源:批量数据移动、compute/I-O 流水线、模块化 KV connector、一等控制 API。
- 与 [[035 KV 驱逐与压缩：H2O 与注意力汇|KV 驱逐]] 互补(无损搬 vs 有损丢);是 [[051 Mooncake：KVCache 中心解耦架构|Mooncake]] 式 KVCache 池与 [[038 KV-aware 路由与跨引擎复用|KV-aware 路由]] 的存储底座。
