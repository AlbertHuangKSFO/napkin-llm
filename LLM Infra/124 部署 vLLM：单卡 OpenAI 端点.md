> ⚠️ 实操篇:命令/配置需 GPU 环境实跑,本机仅校验语法。

[[124 部署 vLLM：单卡 OpenAI 端点]]:把 [[088 vLLM V1 架构剖析|vLLM]] 从「架构图」变成「一行命令起一个 [[094 OpenAI 兼容 API 与引擎抽象|OpenAI 兼容]] 端点」——`pip install vllm` → `vllm serve <model>` → `curl /v1/chat/completions`,顺带 Docker 起法与常用 flag。这是 125–130 一切实操的**地基**。

## ① 类比:一条命令开一家「外卖窗口」

`vllm serve` 干的事像开一个**和 OpenAI 完全一样的外卖窗口**:你的客户端代码(原本调 OpenAI)只要把 `base_url` 改成你的地址,**一个字都不用改**就能点单。窗口背后是 [[088 vLLM V1 架构剖析|EngineCore]] 在炒菜([[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]] + [[041 连续批处理：迭代级调度内幕|连续批]]),但客户看到的只是 `POST /v1/chat/completions`。

## ② 三步起服务

```bash
# 1. 装
pip install vllm

# 2. 起(V1 是默认引擎,chunked prefill / 前缀缓存默认开)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# 3. 测(另开终端)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "用一句话解释 KV-Cache"}],
    "max_tokens": 64
  }'
```

看到 JSON 里有 `choices[0].message.content` 就成了。`model` 字段必须和启动时的模型名一致。

## ③ 原理:这一行背后发生了什么

**1. 拉权重 + 起引擎。** vLLM 从 HuggingFace 下载权重(gated 模型需 `HF_TOKEN`),按 [[011 单卡能放多大模型：参数与 KV 显存预算|显存预算]] 在 GPU 上分配权重 + [[015 KV-Cache 的显存账(逐层手算)|KV-Cache]] 池(PagedAttention 的 block 池)。

**2. 起两个进程。** [[088 vLLM V1 架构剖析|V1]] 把前端(FastAPI/OpenAI 层、tokenize、流式)与 EngineCore(调度+执行)拆成独立进程,IPC 通信,CPU 活与 GPU 循环重叠。

**3. 暴露 OpenAI 兼容路由。** `/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/v1/embeddings`(若是 embedding 模型)。流式靠 SSE:`data: {...}` 多行 + `data: [DONE]` 收尾。

![[lab-vLLM单卡部署.svg]]

放大看这一行命令背后的 V1 双进程结构(前端 FastAPI / EngineCore)与 OpenAI 兼容路由:

![[lab-124双进程架构.svg]]

## ④ 常用 flag 速查

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --max-model-len 8192 \              # 上下文上限,直接影响 KV 显存
  --gpu-memory-utilization 0.90 \     # 允许占用的显存比例(默认 0.9)
  --max-num-seqs 256 \                # 并发序列数上限
  --max-num-batched-tokens 8192 \     # 单步 token 预算(调度核心旋钮)
  --dtype auto \                      # auto/bfloat16/float16
  --api-key sk-local-xxx \            # 给端点加鉴权(可选)
  --served-model-name llama3-8b       # 对外暴露的模型别名
```

- `--max-model-len` 太大 → KV 池吃满显存 → 并发掉;按真实需求设。
- `--gpu-memory-utilization` 留余量给激活/碎片,0.90 是稳妥起点。

## ⑤ 完整脚本:Docker 起 + Python 客户端

```bash
# Docker(官方镜像 vllm/vllm-openai)
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.90
```

```python
# 客户端:复用 openai SDK,只改 base_url
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local-xxx")

resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "解释连续批处理"}],
    max_tokens=128,
    stream=True,                       # 走 SSE 流式
)
for chunk in resp:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

❌ 反模式:Docker 忘了 `--ipc=host`(vLLM 多进程共享内存会 OOM/崩)或漏挂 HF 缓存卷(每次重拉权重)。
✅ 正解:`--gpus all --ipc=host` + 挂 `~/.cache/huggingface`;`--shm-size` 不足时显式加大。

## 面试高频

- **「vLLM 怎么暴露成 OpenAI 兼容?」** `vllm serve` 起 FastAPI 前端,实现 `/v1/chat/completions` 等路由,流式走 SSE;客户端只改 `base_url` 即可复用 OpenAI SDK。
- **「`--max-model-len` 和 `--gpu-memory-utilization` 分别影响什么?」** 前者定上下文上限→决定单序列 KV 显存;后者定 vLLM 可占显存比例→决定 KV 池大小→决定最大并发。
- **「Docker 跑 vLLM 必加哪两个参数?」** `--gpus all` 给 GPU,`--ipc=host`(或大 `--shm-size`)给多进程共享内存,否则崩。
- **「V1 起服务为什么是两个进程?」** 前端(tokenize/流式/OpenAI 层)与 EngineCore(调度+执行)隔离,CPU/GPU 重叠提吞吐。

## 关键事实

- 装:`pip install vllm`;起:`vllm serve <model> --port 8000`;V1 默认引擎,前缀缓存/chunked prefill 默认开。
- 官方镜像 `vllm/vllm-openai:latest`;Docker 必带 `--gpus all --ipc=host` + 挂 HF 缓存。
- 路由:`/v1/chat/completions`、`/v1/completions`、`/v1/models`;流式 SSE `data: {...}` + `data: [DONE]`。
- 核心 flag:`--max-model-len`、`--gpu-memory-utilization`(默认 0.9)、`--max-num-batched-tokens`、`--served-model-name`、`--api-key`。
- 客户端复用 `openai` SDK,仅改 `base_url`。
