[[048 为何分离 prefill 与 decode|分离 prefill 与 decode]](PD 分离 / disaggregation)是把一次 LLM 推理的两个阶段——[[013 Prefill 阶段：计算受限|Prefill]] 与 [[014 Decode 阶段：访存受限|Decode]]——放到**不同 GPU 池**上分别执行的服务架构。动机是:两阶段的资源画像**正好相反**(prefill 吃算力、decode 吃带宽),放同一张卡用同一套 batch/并行策略会**互相伤害**;分离后各自调优、各自扩缩,在严 SLO 下显著抬高 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]]。这是 [[LLM/106 chunked prefill 与 prefill、decode 解耦|PD 解耦]] 在系统层的落地,也是 DistServe、Splitwise、[[051 Mooncake：KVCache 中心解耦架构|Mooncake]]、[[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|Dynamo]] 等系统的共同前提。

## 直觉

把推理想成餐厅:**prefill = 备菜**(切配、炒制,占满灶台火力,一次集中干完);**decode = 端菜**(一道道往外送,跑腿为主,不需要灶台)。让大厨既炒菜又跑堂(同卡合批),炒一大锅菜时所有上菜都得等——这就是 prefill 长任务**阻塞** decode、把 TPOT 抬高的画面。分离 = 大厨只在后厨备菜(prefill 卡),传菜员专职端菜(decode 卡),互不卡脖子。

更尖锐的反直觉:**对两阶段"好"的方向相反**。decode 喜欢大 batch(摊薄逐 token 的权重读取,提吞吐);prefill 讨厌大 batch(本就算力打满,再堆只会让 TTFT 更长)。同一个 batch 旋钮,一个想拧大、一个想拧小 → 单系统注定两头难顾。

**生活类比**:一家小馆子让大厨既炒菜又跑堂。来了一桌 8 人大单,大厨埋头切配炒制要 **5 分钟**(灶火全开,像 prefill 占满算力);这 5 分钟里,已经点好的几十桌只能干等上菜,本来 **3 秒**就能端上去的菜被拖到 **8 秒**(decode 的 TPOT 从 30ms 飙到 80ms)——一个大单堵死整条线。把工位拆开就解套了:大厨退进后厨**只管切配备料**(prefill 卡,缺人手就单加一个灶),专职传菜员**只管一道道端菜**(decode 卡,不被大单干扰),后厨备好的料递给传菜员(就是 KV 从 prefill 卡传到 decode 卡)。上菜稳稳 3 秒,快慢单互不卡脖子。

![[disagg-048类比厨房分工.png]]

## 例子

设两阶段在同一张 H100 上合批(chunked prefill 也只是把 prefill 切片插队,缓解但不消除):

- 一条 8k token 的长 prompt,prefill 约 **300 ms** 占满算力;这期间同卡上几十个正在 decode 的请求被挤,单步 decode 从 **30 ms** 涨到 **80 ms** → 这些请求 **TPOT 超标**。
- 分离后:prefill 卡专心算这条 8k,decode 卡上的几十个请求**不受干扰**,TPOT 稳在 30 ms。代价是这条 prompt 的 **KV 要从 prefill 卡传到 decode 卡**(见 [[053 KV 传输：NIXL、点对点与带宽|KV 传输]])。
- 资源配比:同一批流量下,prefill 算力利用率可能 90%、decode 只 30%;聚合时只能整机扩,分离后可单独加 prefill 卡——见 [[054 PD 配比与独立扩缩|PD 配比]]。

## 原理

两阶段的瓶颈由 [[004 算力 vs 带宽：Roofline 与算术强度|算术强度]] $I$ 与硬件 ridge point $I^\* = C_{\text{FLOPS}}/B_{\text{HBM}}$ 决定:

$$
I_{\text{prefill}} \gg I^\* \ \Rightarrow\ \text{计算受限},\qquad
I_{\text{decode}} \ll I^\* \ \Rightarrow\ \text{访存受限}
$$

合批同卡时,系统的有效时间是两阶段的**串行干扰**之和。设一步内 prefill 注入 $L_p$ 个 token、decode 服务 $B$ 个请求:

$$
T_{\text{step}} \approx \underbrace{\frac{2\,L_p\,N_{\text{params}}}{C_{\text{FLOPS}}}}_{\text{prefill 算力项}} + \underbrace{\frac{N_{\text{params}} + B\cdot S_{\text{kv}}}{B_{\text{HBM}}}}_{\text{decode 访存项}}
$$

prefill 项把 decode 的逐 token 间隔(TPOT)整体推高。分离后两项**落在不同卡并行**,各自只受自身瓶颈约束,于是可分别令 $\partial \text{TTFT}=0$、$\partial \text{TPOT}=0$ 调批,goodput 同时逼近上界。

## 图

![[disagg-PD动机对比.png]]

![[disagg-048两阶段资源画像.png]]

![[disagg-048-1P1D数据流.png]]

## 代码

vLLM 用 `kv_transfer_config` 把一个实例声明成 prefill 或 decode 角色(1P1D 雏形):

```python
# ❌ 聚合:一个引擎同时跑 prefill+decode,chunked prefill 只是缓解干扰
# llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
#           enable_chunked_prefill=True)   # 单卡,prefill 长任务仍抢 decode

# ✅ 解耦:起两个角色,KV 经连接器从 producer 传到 consumer
from vllm import LLM
from vllm.config import KVTransferConfig

prefill = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
    kv_transfer_config=KVTransferConfig(
        kv_connector="NixlConnector", kv_role="kv_producer"))   # 算 KV 的池

decode  = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
    kv_transfer_config=KVTransferConfig(
        kv_connector="NixlConnector", kv_role="kv_consumer"))   # 收 KV 生成的池
# 前置 router 选 1P1D,prompt→prefill 算 KV→NIXL 传→decode 续生成
```

`❌` 同卡合批,prefill 与 decode 抢同一份算力/带宽预算,SLO 紧时必然有一头超标;`✅` 拆成 producer/consumer 两池,各调各的 batch 与并行度,KV 用 NIXL 连接器搬运。

## 面试高频

- **Q:为什么要把 prefill 和 decode 分开?** 两阶段资源画像相反(prefill 计算受限、decode 访存受限),同卡合批会互相干扰、且 batch 旋钮方向相反,单系统难同时满足 TTFT 与 TPOT;分离后各自调优、独立扩缩,goodput 显著提升。
- **Q:chunked prefill 不是已经解决干扰了吗?** chunked prefill 把长 prefill 切片插进 decode 批,**缓解但不消除**干扰,且仍共享同卡的算力/带宽与并行策略;PD 分离是物理隔离,更彻底。
- **Q:分离的主要代价是什么?** prefill 算出的 KV 必须**跨卡/跨节点传到 decode**,引入带宽与延迟开销,需要高速网(RDMA/IB);小规模/低负载时这笔开销可能不划算。
- **Q:分离一定更快吗?** 不一定降单请求延迟,关键是在**严 SLO + 高负载**下提升达标吞吐(goodput);小流量时聚合反而更省。

## 关键事实

- PD 分离把**计算受限的 prefill** 与**访存受限的 decode** 解耦到不同 GPU 池,消除 prefill-decode 干扰,是 **DistServe(OSDI 2024)** 系统性论证的核心思想。
- 与 [[LLM/106 chunked prefill 与 prefill、decode 解耦|chunked prefill]] 的区别:后者在**单卡内**调度缓解干扰,前者在**集群层**物理分离;二者常被对比为"软隔离 vs 硬隔离"。
- 代表系统(2024–2025):DistServe、Splitwise(ISCA 2024)、Mooncake(FAST 2025)、NVIDIA Dynamo(GTC 2025);三者分别强调 goodput、异构硬件、KVCache 中心。
