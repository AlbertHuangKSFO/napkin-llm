[[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|NVIDIA Dynamo]](GTC 2025 发布,开源)是 NVIDIA 的数据中心级分布式推理框架,把 [[048 为何分离 prefill 与 decode|PD 分离]] 做成可生产、**引擎无关**的产品(底层可换 vLLM、TensorRT-LLM、SGLang)。四大组件:**SLO Planner**(按 TTFT/ITL 目标动态扩缩 [[013 Prefill 阶段：计算受限|Prefill]]/[[014 Decode 阶段：访存受限|Decode]] GPU)、**KV-aware Router**(查全局 [[LLM/102 KV-Cache|KV-Cache]] 映射、命中即免重算)、**NIXL**(低延迟点对点 [[053 KV 传输：NIXL、点对点与带宽|KV 传输]])、**KV Block Manager**(分级管 KV 块、向 CPU/存储卸载)。目标是把 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]] 的工程化推到集群规模。

## 直觉

DistServe/Mooncake 证明了 PD 分离的价值,Dynamo 则把它变成"开箱即用的操作系统"。四个组件各管一摊:**Router** 像前台带位——把请求引到已经缓存了相关 KV 的 GPU,省掉重算;**SLO Planner** 像值班经理——盯着 TTFT/ITL 是否达标,不够就调 prefill/decode 卡的配比;**NIXL** 像内部传送带——KV 在卡间高速点对点流动;**KV Block Manager** 像分级仓库——热 KV 留 HBM,温的下沉到 CPU 内存/存储。而且不绑定推理引擎,你原有的 vLLM/TRT-LLM 都能塞进来当后端。

## 例子

NVIDIA 公布的量级(2025):

- 在 NVIDIA **Blackwell** 上跑开源 **DeepSeek-R1**,服务请求数 **最高提升 30×**(相对未用 Dynamo 的基线)。
- 结合 wide expert parallel,在 **GB200 NVL72** 上把 MoE 模型吞吐**最高提升 7×**(相对 B200 系统)。
- KV-aware Router 的收益例子:多轮对话里,新一轮请求若被路由到**已持有前几轮 KV** 的 decode 卡,可直接续算,免去把历史 prompt 再 prefill 一遍。
- SLO Planner 的动作:监测到 ITL 逼近阈值且 decode 队列堆积 → 自动从 prefill 池借/扩几台到 decode 池,把配比从 2P2D 调到 1P3D(见 [[054 PD 配比与独立扩缩|PD 配比]])。

## 原理

SLO Planner 在 GPU 预算 $G$ 下,选 prefill/decode 实例数 $(n_p,n_d)$ 与并行/批配置,使 TTFT、ITL 双双达标:

$$
\text{find } (n_p, n_d)\ \text{s.t.}\ n_p+n_d\le G,\
\text{TTFT}\le \tau_{\text{ttft}},\ \text{ITL}\le \tau_{\text{itl}}
$$

它靠**部署前 profiling** 估出"并行度/批 → 延迟"的曲线,再在线按负载漂移调 $(n_p,n_d)$。KV-aware Router 维护一张全局 KV 位置表 $\mathcal{M}: \text{prefix}\mapsto \text{GPU}$,路由时最大化命中:

$$
\text{route}(r) = \arg\max_{g}\ \text{overlap}\big(\text{prefix}(r),\ \mathcal{M}^{-1}(g)\big)
\;\Rightarrow\; \text{省 prefill FLOPs}
$$

KV Block Manager 把 KV 切块分级(HBM→CPU→存储),NIXL 负责块在卡间的点对点搬运。

## 图

![[disagg-Dynamo组件.svg]]

![[disagg-052SLOPlanner闭环.svg]]

![[disagg-052KV感知路由.svg]]

## 代码

Dynamo 用声明式配置把组件拼起来,后端引擎可换:

```yaml
# ❌ 单体引擎:聚合部署,无全局 KV 路由、无 SLO 驱动的动态扩缩
# serve: { engine: vllm, mode: aggregated }

# ✅ Dynamo:四组件 + 引擎无关后端
router:        { type: kv_aware }          # 查全局 KV 映射,命中即免重算
planner:       { type: slo, ttft_ms: 200, itl_ms: 40 }  # 按 SLO 动态扩缩 P/D
kv_transfer:   { backend: nixl }           # 点对点低延迟搬 KV
kv_block_mgr:  { tiers: [hbm, cpu, storage] }  # KV 分级,温块下沉

prefill_pool:  { engine: trtllm,  role: prefill, min: 2, max: 8 }  # 独立扩缩
decode_pool:   { engine: vllm,    role: decode,  min: 2, max: 16 } # 引擎可不同
```

`❌` 单体聚合引擎缺全局 KV 视图与 SLO 闭环,过载只能整机扩;`✅` Router 免重算、Planner 按 TTFT/ITL 自动调 P/D 配比、NIXL 搬 KV、KVBM 分级,后端引擎可混用。

## 面试高频

- **Q:Dynamo 的四大组件分别干什么?** SLO Planner(按 TTFT/ITL 动态扩缩 prefill/decode GPU)、KV-aware Router(查全局 KV 映射、命中免重算)、NIXL(点对点低延迟 KV 传输)、KV Block Manager(KV 分级、向 CPU/存储卸载)。
- **Q:"引擎无关"是什么意思?** Dynamo 是编排/调度层,底层推理引擎可换 vLLM、TensorRT-LLM、SGLang,甚至 prefill 与 decode 用不同引擎。
- **Q:KV-aware Router 怎么省算力?** 维护全局 KV 位置表,把请求路由到已缓存其前缀 KV 的 GPU,直接续算,避免重复 prefill。
- **Q:Dynamo 的代表数字?** GTC 2025 发布;Blackwell 上 DeepSeek-R1 **最高 30×** 请求量;GB200 NVL72 上结合 wide-EP,MoE 吞吐 **最高 7×**(相对 B200)。
- **Q:SLO Planner 与普通自动扩缩有何不同?** 它是 **SLO 驱动 + PD 感知**:基于部署前 profiling 的延迟曲线,按 TTFT/ITL 目标分别调 prefill 与 decode 的实例数,而非按 CPU/GPU 利用率盲扩。

## 关键事实

- **NVIDIA Dynamo**,**GTC 2025** 发布,开源(ai-dynamo/dynamo);定位数据中心级分布式推理框架,**引擎无关**。
- 四大件:**SLO Planner / KV-aware Router / NIXL / KV Block Manager**;NIXL = NVIDIA Inference Xfer Library(见 [[053 KV 传输：NIXL、点对点与带宽|KV 传输]])。
- 数字:Blackwell 上 DeepSeek-R1 **最高 30×**;GB200 NVL72 + [[062 Wide-EP：DeepSeek、Kimi 在 H100、H200 上的部署|wide-EP]],MoE 吞吐 **最高 7×**(vs B200)。它把 DistServe/Mooncake 的研究范式工程化成产品。
