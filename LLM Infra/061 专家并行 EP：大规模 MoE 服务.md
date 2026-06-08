这是 [[LLM Infra]] 域里大规模 MoE 推理的起点,上承 [[056 推理并行总览：TP、PP、EP、DP、SP 各管什么|推理并行总览]],下接 [[062 Wide-EP：DeepSeek、Kimi 在 H100、H200 上的部署|Wide-EP]]。**专家并行(Expert Parallelism, EP)**只对 [[LLM/047 DeepSeek MoE：细粒度与共享专家|DeepSeek MoE]] 这类 MoE 模型有意义:把一层里的**多个专家(FFN)分散到不同 GPU**,每张卡只持有总专家的一部分权重;一个 token 经路由器选出 top-k 专家后,被 [[068 all-to-all：MoE、EP 的通信瓶颈|all-to-all]] 通信送到专家所在的卡上算、再送回来。它把 MoE 的"参数多但每 token 只激活一小撮"这一稀疏性,转化成**单卡权重显存的大幅下降**。EP 与训练侧的 [[LLM/049 专家并行 EP 与 MoE 部署|专家并行]] 同源,但推理无反向,通信只来自前向的 dispatch / combine。

把 MoE 层想成一个有 256 个专科门诊的大医院。**纯 [[057 张量并行推理：延迟换显存|TP 推理]]** 相当于让每个门诊都派半个医生到每栋楼——每栋楼都得备齐全部科室的设备,占地巨大却闲置(一个病人只看 8 个科)。**EP** 则是把整个科室成建制搬到不同楼:1 楼放 1–128 科、2 楼放 129–256 科,每栋楼只备自己那批科室的设备(权重);病人(token)按需被导诊台(router)分流到对应楼层就诊(dispatch),看完再回原楼层汇总(combine)。设备(显存)省了,代价是病人要在楼之间跑(网络通信)。

**生活类比**:一家医院全院共 **256 个专科**,但每位病人(token)挂号后,导诊台(router)只把他**分诊到对口的 8 个科**(top-8 激活),其余 248 个科他根本不去。EP 就是把科室成建制搬到 8 栋楼:每栋只放 256/8 = **32 个科**,只备自己那批科的设备——单栋楼的设备(权重显存)直接降到全院的 **1/8**。一批 4096 个病人按 top-8 分流,平均每个科收到 4096×8/256 = **128 人**,科室一次能多看几个人(GEMM batch 够大、Tensor Core 喂得满)。病人按需在楼间跑(dispatch all-to-all 送过去、combine 收回来)就是代价。技术对应:科室=专家、导诊台=router、楼=GPU、跑楼层=all-to-all 通信。

![[ipar-061类比医院分科室.png]]

小数字感受。DeepSeek-V3:每层 **256 个路由专家 + 1 个共享专家**,每 token 激活 **8 个路由专家**。纯 TP=8 时,每卡要存 256 个专家各 1/8 列,权重仍铺满全卡;改用 **EP=8**,每卡只完整持有 256/8 = **32 个专家**,该层路由专家权重显存直接降到约 **1/8**。再看每专家 batch:假设一批 4096 个 token、top-8、256 专家,平均每专家收到 $4096\times8/256=128$ 个 token——EP 把许多卡上的 token 聚到同一专家,GEMM 的 batch 维才够大、Tensor Core 才喂得满,这正是 [[062 Wide-EP：DeepSeek、Kimi 在 H100、H200 上的部署|Wide-EP]] 拉大 EP 度的根本动机。

$$
\text{每卡专家数}=\frac{N_{\text{expert}}}{\text{EP}},\qquad
\overline{\text{每专家 token}}=\frac{b\cdot s\cdot k}{N_{\text{expert}}}
$$

$$
\text{单 MoE 层通信量(dispatch+combine)}\approx 2\cdot b\cdot s\cdot k\cdot h\ \text{(激活字节)},\quad k=\text{top-k}
$$

通信量正比于**被激活的 token×k×隐藏维**,而**不依赖专家总数**——这是 EP 相对 TP 的关键优势:扩专家不加通信。

**共享专家放哪(手算)**。DeepSeek-V3 每层 256 路由 + **1 个共享专家**,共享专家对**所有** token 激活、不参与 top-k。EP=8 下 256 路由专家按 256/8=**32 个/卡**分散;但共享专家若也只放 1 张卡,那批卡的全部 token 都要 all-to-all 绕过去算它,等于给每个 token 凭空加一次跨卡往返。所以工程上把共享专家**在 8 张卡各复制一份**:每张卡先本地算完自己 token 的共享专家(零通信),再把 top-8 路由部分走 dispatch/combine。代价是共享专家权重 ×8(单专家通常远小于 32 个路由专家的总量,显存可忽略),换来共享部分零通信——典型的「小权重换大通信」取舍。

![[ipar-EP分布与路由.png]]

![[ipar-专家并行all-to-all.png]]

![[ipar-ep三段时序.png]]

```python
# SGLang:开专家并行服务 DeepSeek-V3(EP=8,与 TP/DP 组合)
# python -m sglang.launch_server \
#   --model-path deepseek-ai/DeepSeek-V3 \
#   --tp 8 --enable-ep-moe \           # MoE 走 EP,注意力走 TP/DP
#   --ep-size 8

# vLLM:数据并行注意力 + 专家并行 MoE
# vllm serve deepseek-ai/DeepSeek-V3 \
#   --data-parallel-size 8 \           # 注意力 DP(见 063 attn-DP)
#   --enable-expert-parallel            # MoE 自动按 DP×TP 展开为 EP
```

```text
❌ MoE 模型直接套 70B dense 的做法,纯 TP=8 切到底
   → 每卡仍存全部 256 专家的分片,显存没省;每专家 batch 太小,GEMM 利用率低
✅ MoE 层走 EP(每卡只放 32 专家),注意力层另配 DP/TP
   → 权重显存降到 1/EP,每专家聚合大 batch 喂满 Tensor Core
```


![[ipar-061EP对比TP显存.png]]

## 面试高频
- **EP 和 TP 切 MoE 有何本质区别?** TP 切每个专家的权重矩阵(列/行),每卡仍含全部专家;EP 把整个专家分配到不同卡,每卡只含一部分专家。EP 通信量∝激活而与专家总数无关,所以适合"专家很多"的场景。
- **EP 为什么能降显存却不一定降延迟?** 它把权重摊薄(降显存),但引入两次 all-to-all;在小 batch、跨节点时这通信可能成为延迟瓶颈,需 DeepEP 低延迟内核 + 通信计算重叠来掩盖。
- **EP 度越大越好吗?** 不是。EP 越大每卡专家越少、聚合 batch 越大(利于算力),但 all-to-all 的扇出与跨节点跳数也越大;还会放大**专家负载不均**,需 EPLB 做冗余专家与再均衡。
- **shared expert 在 EP 里怎么放?** 共享专家对所有 token 激活,通常在每卡复制或单独占卡,不参与 top-k 路由。

## 关键事实
- DeepSeek-V3(**2024**,arXiv:2412.19437):每层 256 路由专家 + 1 共享专家,每 token 激活 8 个路由专家。
- DeepSeek 官方推理系统(**2025** 开源周):dispatch / combine 两次 all-to-all 用 NVSHMEM 实现的 DeepEP 内核,分高吞吐(prefill)与低延迟(decode)两种模式。
- EP 通信量∝ $b\cdot s\cdot k\cdot h$,与专家总数无关——这是 EP 可"加宽专家"而不加通信的根本。
- vLLM / SGLang(**2025**)均以"DP 注意力 + EP MoE"为 DeepSeek 系标准部署形态(EP = DP × TP)。
