[[037 Mooncake：KVCache 中心的存储池]] 是 Moonshot AI 为 **Kimi** 落地的生产级推理架构(论文 [[051 Mooncake：KVCache 中心解耦架构|Mooncake 架构]] 即指它):核心思想是**把整个 GPU 集群里闲置的 CPU、DRAM、SSD、NIC 攒成一个解耦的 KVCache 池**,让 KV 成为集群的「一等公民」而非困在单卡里。在此之上做 [[048 为何分离 prefill 与 decode|PD 解耦]](prefill 集群与 decode 集群分开),由一个 **KVCache 中心调度器**统一编排。它把 [[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|分层 offload]] 从「单机沉 KV」上升到「集群级共享池」,并天然需要 [[038 KV-aware 路由与跨引擎复用|KV-aware 路由]] 来命中池中已有的前缀。

## 直觉类比
传统部署像每个厨师(GPU)守着自己的小冰箱(显存),备好的高汤(KV)只有自己能用,别人要喝得重新熬。Mooncake 把整个后厨闲置的**大冰柜、储藏室、冷库**(集群里没用满的 CPU/DRAM/SSD)联网成一个**中央食材库**(KVCache 池):谁熬好的高汤都存进库,任何厨师要用同款高汤直接取,不必重熬。再把「熬汤的灶」(prefill,吃算力)和「装盘上菜的工位」(decode,吃带宽)**分开摆**,各自按需扩容。

## 小数字例子
设集群 100 张卡,每张卡配套机器还有大量没用满的 CPU 内存与本地 SSD。
- 假设每机闲置 **256 GB** DRAM + **2 TB** SSD → 全集群攒出约 **25 TB DRAM + 200 TB SSD** 的 KVCache 池,容量是单卡 80 GB HBM 的几百倍。
- 一份 32K token 长文档前缀(~16 GB KV)算一次存池,**全集群任意节点**之后都能命中复用,省下所有重复 prefill。
- 论文报告:在某些模拟长上下文场景吞吐相对基线最高 **+525%**;真实负载下 Mooncake 让 **Kimi 多处理约 75% 请求**。过载时用**预测式提前拒绝**保 SLO(不假设所有请求都会被处理)。

## 原理:为什么以 KVCache 为中心
LLM 推理两阶段资源画像截然不同:[[013 Prefill 阶段：计算受限|prefill]] **算力受限**(并行处理整段 prompt),[[014 Decode 阶段：访存受限|decode]] **访存受限**(每步 1 token、读全量 KV)。把两者塞同一组卡会互相拖累。Mooncake 的解法:

1. **PD 解耦**:prefill 集群与 decode 集群分开,各自按瓶颈维度扩容。
2. **KVCache 池**:prefill 算出的 KV 写入由集群闲置 CPU/DRAM/SSD/NIC 组成的解耦池;decode 从池取 KV 续写。KV 在 P→池→D 之间流动,成为架构中心。
3. **前缀去重/复用**:同前缀只存一份,跨请求/跨节点命中即取,免重算。
4. **KVCache 中心调度器**:在「最大化有效吞吐」与「满足延迟 SLO」之间权衡,过载时**预测式提前拒绝**。

调度目标可粗略写成在 SLO 约束下最大化 goodput:

$$
\max \ \text{goodput}\quad \text{s.t.}\quad \text{TTFT}\le \text{SLO}_{\text{ttft}},\ \ \text{TPOT}\le \text{SLO}_{\text{tpot}}
$$

KVCache 命中率越高,prefill 负担越轻,可服务的 goodput 越高——这正是 KVCache 中心设计的杠杆。

![[kv-Mooncake架构.png]]

![[kv-037闲置资源攒池手算.png]]

## 配置 / 概念示意
```text
# Mooncake 数据流(概念)
请求 → KVCache 中心调度器
        ├─ 查前缀命中(KVCache 池) ─ 命中→ 跳过/缩短 prefill
        ├─ 路由到 Prefill 集群 ─ 算 KV ─ 写入 KVCache 池(闲置 CPU/DRAM/SSD,经 NIC/RDMA)
        └─ 路由到 Decode 集群 ─ 从池取 KV ─ 自回归生成
   过载时:预测式提前拒绝(early rejection)保 SLO
```

```python
# 配套开源:kvcache-ai/Mooncake 提供分布式 KVCache store(Transfer Engine)
# vLLM/SGLang 等可把它当作跨节点 KV 后端接入(配置示意)
kv_transfer_config = {
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both",
    "kv_buffer_device": "cpu",          # 池主体用闲置 CPU DRAM
    "transfer_backend": "rdma",         # NIC/RDMA 高速搬运
}
```

```text
❌ prefill 与 decode 挤同一组卡 + KV 困在单卡:算力/带宽互相拖累,公共前缀反复重算
✅ PD 解耦 + 集群级 KVCache 池:各按瓶颈扩容,KV 全集群共享复用,goodput 在 SLO 下最大化
```

## 面试高频
- **「KVCache 中心」是什么意思?** 让 KV 成为架构的一等资源:prefill 产 KV、池存 KV、decode 取 KV,调度也围绕 KV 命中率展开,而非把 KV 当某张卡的私有附属。
- **池子里的存储从哪来?** 集群里**闲置**的 CPU、DRAM、SSD、NIC——本来就有但没用满,攒起来当解耦 KVCache,几乎零额外硬件成本。
- **Mooncake 和 LMCache 关系?** 都是把 KV 搬出 GPU 跨请求复用;LMCache 偏「KV 缓存层」组件,Mooncake 是整套 **PD 解耦 + 集群 KVCache 池 + 调度**的生产架构(Kimi 在用)。
- **为什么要 PD 解耦?** prefill 算力受限、decode 访存受限,资源画像相反;分开各自扩容、各自打满,避免互相干扰。
- **过载怎么办?** 预测式**提前拒绝**:预判某请求无法在 SLO 内完成就早拒,保住其余请求的 SLO(不假设全收)。

## 关键事实
- **Mooncake**,Qin et al. **2024**(arXiv 2407.00079,提交 6/24;`kvcache-ai/Mooncake` 开源):Kimi(Moonshot AI)生产推理平台。
- 核心:**KVCache 中心 + PD 解耦**;用集群**闲置 CPU/DRAM/SSD/NIC** 组成解耦 KVCache 池。
- 调度器在最大化有效吞吐与满足 SLO 间权衡;**预测式提前拒绝**应对过载。
- 性能:长上下文模拟场景吞吐最高 **+525%**;真实负载让 Kimi 多处理约 **75%** 请求。
- 是 [[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|分层 offload]] 的集群级形态,依赖 [[038 KV-aware 路由与跨引擎复用|KV-aware 路由]] 命中池中前缀,基于 [[013 Prefill 阶段：计算受限|prefill]]/[[014 Decode 阶段：访存受限|decode]] 资源画像差异(详见 [[048 为何分离 prefill 与 decode|PD 解耦]])。
