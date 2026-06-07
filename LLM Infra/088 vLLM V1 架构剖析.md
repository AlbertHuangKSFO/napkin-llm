[[088 vLLM V1 架构剖析]]:把 [[087 引擎全景：六大 runtime 选型|六引擎全景]]里的「通用首选」拆开看——vLLM **V1**(2025-01 发布的核心架构重写)如何用 **AsyncLLM 前端进程 ⇄ 隔离 EngineCore 执行循环**、**统一调度器**、**默认 chunked prefill**、**近零开销前缀缓存**把 [[LLM/026 PagedAttention 与 KV 分页|PagedAttention]] 和[[041 连续批处理：迭代级调度内幕|连续批]]组装成事实标准的 serving 系统。

## ① 类比:把「点单/上菜」和「后厨炒菜」拆成两个独立循环

V0 像一个人既在前台收银(tokenize、拼 prompt、回传)又在灶台炒菜(跑 GPU),两边互相等。**V1 把前台和后厨彻底分到两个进程**:前台(AsyncLLM)专心接客、切配、打包流式回传;后厨(EngineCore)只盯着一件事——调度 + 跑模型,灶火never 熄。两进程靠传菜窗口(IPC)对接,于是「切配」的 CPU 活和「炒菜」的 GPU 活能**重叠**,GPU 不再为 CPU 杂活停火。

![[eng-088V0对V1进程.png]]

## ② 小数字例子:V1 升级带来什么

- Red Hat / vLLM 报告:从 V0 切到 V1 引擎,多种负载下吞吐有可观提升(0.8.x 起 V1 成默认),CPU 开销大的多模态、tokenize 场景收益尤其明显。
- **前缀缓存近零开销**:V0 时代开 APC 在命中率低时反而拖慢(Python 对象/驱逐开销);V1 用常数时间驱逐 + 极少对象创建,**命中率 0% 也几乎不掉性能**,于是可以**默认常开**。
- **统一调度**:一个请求在某 step 处理多少 token 就是字典里一个数,prefill 和 decode 在同一 batch 里按 token 预算自由混合 → 不再有「prefill 撑爆一步、decode 饿着」的相位割裂。

![[eng-088统一调度token预算.png]]

## ③ 原理:四个关键设计

**1. AsyncLLM 进程 ⇄ EngineCore 隔离循环。** EngineCore 是一个**只含 scheduler + model executor 的纯执行循环**,被隔离到独立进程;tokenize、多模态预处理、detokenize、流式回传这些 CPU 密集活留在前端进程,两者 IPC 通信。好处:CPU 活与 GPU 核心循环**重叠**,吞吐上限被 CPU 杂活拖累的情况大幅缓解。

**2. 统一调度器,取消 prefill/decode 二分。** V1 把用户 prompt token 和模型生成 token **一视同仁**:调度决策表示为 `{request_id: num_tokens}`——这一步给每个请求安排处理多少 token。在固定 token 预算下,调度器动态分配,让计算受限的 [[013 Prefill 阶段：计算受限|prefill]] 和访存受限的 decode 自然交织(见[[043 prefill、decode 干扰与 stall|prefill/decode 干扰]]的背景)。

**3. chunked prefill 默认开。** [[042 chunked prefill：切块融合|chunked prefill]] 把长 prompt 切成块、与 decode token 拼进同一 batch,V1 **始终默认启用**,自动平衡 compute-bound(prefill)与 memory-bound(decode),避免长 prompt 独占一步、卡住正在解码的请求。

**4. 近零开销前缀缓存 + 零拷贝调度。** 前缀缓存数据结构优化为**常数时间驱逐**、极少 Python 对象创建;调度按 token 数下发、避免大对象搬运。配合 PagedAttention 的块表,KV 复用与显存碎片管理都做得很轻。

![[eng-vLLM-V1架构.png]]

底层仍是 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]]、[[041 连续批处理：迭代级调度内幕|连续批]]、[[044 调度器设计：waiting、running 队列与抢占|waiting/running 队列与抢占]]、[[045 抢占：重计算 vs swap|抢占 recompute/swap]]、[[032 前缀缓存：RadixAttention 树结构|前缀缓存]]这些概念,V1 是把它们用更干净的进程/调度结构重新组织。

## ④ 代码/配置:真实启动与关键开关

```bash
# 最小:一行起 OpenAI 兼容端点(V1 是默认引擎)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# 生产常用开关
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching        # V1 下近零开销,默认即可常开
```

```python
# 离线批量(LLM 类直接驱动 EngineCore)
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
out = llm.generate(["解释 PagedAttention"], SamplingParams(max_tokens=128))
```

❌ 反模式:还按 V0 心智手动 `--enable-chunked-prefill` 或纠结「该不该开前缀缓存」,甚至为了「省开销」关掉 APC。
✅ 正解:V1 下 chunked prefill 默认开、前缀缓存近零开销常开;调优重心转向 `--max-num-batched-tokens`(token 预算)、`--gpu-memory-utilization`、并行度,让统一调度器自己平衡 prefill/decode。

## 面试高频

- **「vLLM V1 相比 V0 改了什么?」** ① EngineCore 隔离进程 + AsyncLLM,CPU/GPU 重叠;② 统一调度器取消 prefill/decode 二分,决策是 `{req_id: num_tokens}`;③ chunked prefill 默认开;④ 前缀缓存重写为常数时间驱逐、近零开销。
- **「为什么 V1 敢把前缀缓存默认常开?」** V0 命中率低时 APC 反而拖慢;V1 优化数据结构后命中率 0% 也几乎不掉,故可常开。
- **「统一调度器 vs 传统 prefill/decode 分阶段,好处?」** 同一 batch 内按 token 预算混合 compute-bound 与 memory-bound 工作,提升 GPU 利用、减少相位 stall。
- **「EngineCore 隔离进程解决什么瓶颈?」** tokenize/detokenize/多模态/流式这些 CPU 活不再阻塞 GPU 核心循环,二者重叠提吞吐。
- **「vLLM 和 [[094 OpenAI 兼容 API 与引擎抽象|OpenAI API]] 的关系?」** 前端是 FastAPI 实现的 OpenAI 兼容层,SSE 流式;底层是 EngineCore。

## 关键事实

- vLLM **V1 架构** 2025-01-27 官方博客发布;约 0.8.1(2025-04 前后)起成为默认引擎。
- EngineCore = scheduler + model executor 的隔离执行循环;前端做 tokenize/多模态/detokenize/流式。
- 调度决策表示为 `{request_id: num_tokens}`,**chunked prefill 始终默认开**。
- 前缀缓存:常数时间驱逐、极少 Python 对象 → 命中率 0% 也近零开销。
- 仓库 `vllm-project/vllm`;OpenAI 兼容 server 基于 FastAPI,SSE 格式 `data: {...}` + `data: [DONE]`。
