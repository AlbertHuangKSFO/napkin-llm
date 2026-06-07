> ⚠️ 实操篇:命令/配置需 GPU 环境实跑,本机仅校验语法。

[[129 PD 分离实验：Dynamo、llm-d 上手]]:把 [[048 为何分离 prefill 与 decode|PD 分离]] 从理论跑成实验——起独立的 prefill worker 与 decode worker,配 [[053 KV 传输：NIXL、点对点与带宽|NIXL]] 把 KV 从 prefill 卡直传 decode 卡,用 [[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|Dynamo]] 或 [[098 llm-d：K8s 原生分布式推理|llm-d]] 的最小例子,观察 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]] 变化。

## ① 类比:把「备料」和「炒菜」分到两个厨房

一个厨房里,大单备料(prefill,[[013 Prefill 阶段：计算受限|计算受限]])一上来就占满灶,正在炒的菜(decode,[[014 Decode 阶段：访存受限|访存受限]])只能干等——互相干扰(见 [[043 prefill、decode 干扰与 stall|干扰与 stall]])。**PD 分离开两个专用厨房**:备料厨房只管把料备好(算力密集、可大 batch),炒菜厨房只管持续出菜(带宽密集、低延迟),备好的料(KV-Cache)用传送带(NIXL)直接送过去。两边能**独立配比、独立扩缩**(几台备料配几台炒菜,见 [[054 PD 配比与独立扩缩|PD 配比]])。

## ② Dynamo:最小 PD 分离

```bash
# 装(NVIDIA 官方;镜像或 pip)
pip install ai-dynamo[all]

# 起一套 vLLM 后端的 disaggregated 服务(prefill + decode worker)
# 仓库 ai-dynamo/dynamo 的 examples 提供现成 graph + config
dynamo serve graphs.disagg:Frontend -f ./configs/disagg.yaml
```

```yaml
# disagg.yaml(节选示意:一个 prefill worker + 一个 decode worker)
Frontend:
  served_model_name: meta-llama/Llama-3.1-70B-Instruct
  router: kv                       # KV-aware 路由
VllmWorker:                        # decode worker
  tensor_parallel_size: 4
  remote_prefill: true             # 把 prefill 派给独立 worker
  conditional_disagg: true         # 短 prompt 可本地 prefill(更省)
PrefillWorker:                     # prefill worker
  tensor_parallel_size: 4
```

Dynamo 用 **NIXL** 把 KV 从 prefill 引擎 VRAM **非阻塞**直传 decode 引擎 VRAM,GPU forward 不被 KV 传输阻塞。

## ③ 原理:KV 怎么过去、为什么提 goodput

**KV 传输是命门。** prefill 算完整个 prompt 的 KV,要搬到 decode 卡。Dynamo 走 [[053 KV 传输：NIXL、点对点与带宽|NIXL]](点对点、非阻塞、可走 NVLink/IB/RDMA),传输与计算重叠,不阻塞 forward。搬运量 = KV 大小,所以**长 prompt 才值得分离**(短 prompt 直接本地 prefill 更省,故 `conditional_disagg`)。

**为什么提 goodput。** 分离后 prefill 可大 batch 冲算力、decode 专心保 TPOT,两相不再抢资源 → 同样 SLO 下能服务更多请求([[049 DistServe：goodput 优先的解耦|DistServe]] 的核心论点)。再配 KV-aware 路由(prefix 命中的请求路到有缓存的 decode 卡)进一步省。

**独立扩缩。** prefill 重则多加 prefill worker,decode 重则多加 decode worker,不必整机等比扩(见 [[054 PD 配比与独立扩缩|PD 配比]];Dynamo 的 SLO Planner 自动调,见 [[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|Dynamo Planner]])。

![[lab-PD分离部署.png]]

放大看 KV 如何经 NIXL 点对点直传 decode 卡、两侧如何独立配比扩缩,以及 Dynamo(框架)vs llm-d(K8s 原生)的形态差异:

![[lab-129KV传输.png]]

## ④ llm-d:K8s 原生 PD 分离

```bash
# llm-d 走 K8s + Helm;新路径用 llm-d-infra(deployer 已弃用)
git clone https://github.com/llm-d-incubation/llm-d-infra.git
cd llm-d-infra/quickstart

# 先装基础设施(Gateway API Inference Extension、CRD 等)
./install-deps.sh
helmfile apply -f examples/pd-disaggregation/helmfile.yaml
# ModelService 声明 P/D 分离:prefill / decode 各自 deployment + LeaderWorkerSet
```

```yaml
# ModelService 节选:声明 prefill / decode 两组角色
apiVersion: llm-d.ai/v1alpha1
kind: ModelService
spec:
  modelArtifacts:
    uri: hf://meta-llama/Llama-3.1-70B-Instruct
  decode:
    replicas: 2
    parallelism: { tensor: 4 }
  prefill:
    replicas: 1
    parallelism: { tensor: 4 }
  # 走 vLLM + Gateway API Inference Extension 做 KV-aware 路由
```

## ⑤ 观察 goodput:对比实验

```bash
# 对「聚合(单引擎)」 vs 「PD 分离」跑同一长上下文压测,比 goodput
vllm bench serve --backend vllm --model $MODEL \
  --dataset-name sharegpt --dataset-path ./sharegpt.json \
  --num-prompts 1000 --request-rate 20 \
  --metric-percentiles 99 | tee pd_disagg.txt
# 看:同 request-rate 下 p99 TTFT/TPOT 是否同时满足 SLO(=goodput 提升)
```

❌ 反模式:短 prompt、低并发也硬上 PD 分离——KV 传输开销 > 收益,反而更慢更复杂。
✅ 正解:**长 prompt / 高并发 / prefill 与 decode 明显互相干扰**时才上 PD;开 `conditional_disagg` 让短请求本地 prefill;用 goodput(而非裸吞吐)判定收益。

## 面试高频

- **「PD 分离怎么落地,KV 怎么过去?」** prefill/decode 各自独立 worker,prefill 算完 KV 用 NIXL 点对点非阻塞直传 decode 卡 VRAM,传输与计算重叠。
- **「什么时候该上 PD 分离?」** 长 prompt、高并发、prefill/decode 互相干扰致 goodput 上不去;短 prompt 低并发不值得(KV 传输开销大于收益)。
- **「PD 分离的核心收益是什么?」** prefill 冲算力、decode 保 TPOT,互不抢资源 → 同 SLO 下 goodput 更高;且可独立配比与扩缩。
- **「Dynamo 和 llm-d 部署形态差别?」** Dynamo 是框架(`dynamo serve` + graph/config,可裸机/K8s);llm-d 是 K8s 原生(Helm/ModelService + Gateway API Inference Extension 做 KV-aware 路由)。
- **「`conditional_disagg` 解决什么?」** 短 prompt 本地 prefill 更省,避免不必要的 KV 传输。

## 关键事实

- Dynamo:`pip install ai-dynamo[all]`;`dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml`;NIXL 非阻塞 KV 直传。
- 配置:decode worker `remote_prefill: true` + `conditional_disagg: true`;独立 PrefillWorker。
- llm-d:K8s 原生,新路径 `llm-d-infra` + helmfile(`llm-d-deployer` 已弃用);ModelService 声明 prefill/decode 两组。
- 收益判据是 goodput 不是裸吞吐;长 prompt/高并发才值得分离。
- KV 传输走 NIXL,可 NVLink/IB/RDMA,点对点非阻塞、与 forward 重叠。
