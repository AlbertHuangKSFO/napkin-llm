[[094 OpenAI 兼容 API 与引擎抽象]]:[[087 引擎全景：六大 runtime 选型|六引擎]]为什么能「随便换」的根因——它们都实现同一套 **OpenAI 兼容 `/v1/chat/completions` 契约 + SSE 流式**,于是引擎变成可热插拔的实现;应用只认接口、网关统一鉴权路由,换引擎只改 `base_url`。这是[[088 vLLM V1 架构剖析|引擎四层抽象]]里「① API 趋同」的具体兑现,也是跨域链接 [[LLM/094 LLM.int8 与离群值|量化]] 之外、本域与生产对接的关键一篇。也对应实操 [[LLM/124 部署 vLLM：单卡 OpenAI 端点|vLLM 部署]]。

## ① 类比:USB 标准口

不同品牌的硬盘、键盘、网卡能插同一个 USB 口,因为**接口标准化**,设备内部怎么实现不重要。`/v1/chat/completions` 就是 LLM 推理界的 USB 标准口:vLLM、SGLang、TGI、TensorRT-LLM、llama.cpp 内部架构天差地别,但都「长出一个 OpenAI 形状的口」。于是你的应用(openai SDK、LangChain、自研网关)像「插 U 盘」一样,改个 `base_url` 就换引擎,代码一行不动。

![[eng-094引擎热插拔.png]]

## ② 小数字例子:契约统一带来的实操收益

- **换引擎零改代码**:`base_url` 从 `http://vllm:8000/v1` 改成 `http://sglang:30000/v1`,应用逻辑、SDK 调用全不变——A/B 两引擎、灰度迁移都靠改一个地址。
- **网关统一管控**:一个 OpenAI 兼容网关在前面统一**鉴权、限流、路由、计费、日志**,后面挂 N 个不同引擎/不同模型,对客户端只暴露一个标准口。
- **本地无缝替云**:把 `base_url` 指向 llama.cpp 的 `:8080/v1`,同一份调 OpenAI 的代码直接跑本地模型,离线开发/省成本。

## ③ 原理:契约 + 流式 + 抽象

**1. 统一契约。** `POST /v1/chat/completions` 已成对话式 LLM 的**事实标准**(取代旧 `/v1/completions` 的纯文本补全):
- 请求:`messages`(role/content 的结构化对话)、`model`、`stream`、`tools`(函数调用)、采样参数(temperature/top_p/max_tokens…)。
- 响应:`choices[].message`(非流式)或 `choices[].delta`(流式)。
结构化的 role-based 对话 + 内建上下文管理,是它压过 legacy 文本补全的原因。

**2. 流式 SSE。** 流式用 **Server-Sent Events**:服务端逐 token 推 `data: {…}` 行,结束发 `data: [DONE]`。vLLM/SGLang/TGI 都按此实现,客户端可边生成边渲染(关系到 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TTFT/ITL]] 等服务指标)。

![[eng-094SSE流式契约.png]]

**3. 引擎抽象 = 四层里的 ① 趋同。** 各引擎 ②调度 ③显存 ④后端实现差异巨大(见 [[088 vLLM V1 架构剖析|vLLM V1]]/[[089 SGLang：RadixAttention、HiCache 与前端|SGLang]]/[[090 TensorRT-LLM：编译式极致优化|TRT-LLM]]/[[091 TGI 与 Hugging Face 生态|TGI]]/[[092 llama.cpp、GGUF：CPU 与端侧|llama.cpp]]),但 **①API 层全部对齐 OpenAI 形状**。vLLM 的兼容 server 基于 FastAPI、是 HTTP 客户端与 EngineCore 的桥;SGLang 有专门的 `serving_chat.py` 处理 `/v1/chat/completions`;TGI 由 Rust router 暴露。正因 ① 可互换,「引擎」才成为可替换组件。

![[eng-OpenAI兼容API.png]]

## ④ 代码/配置:同一份代码打不同引擎

```bash
# 三家引擎,都起在 OpenAI 兼容口
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000        # vLLM  -> :8000/v1
python -m sglang.launch_server --model-path <model> --port 30000  # SGLang -> :30000/v1
llama-server -m model-Q4_K_M.gguf --port 8080                  # llama.cpp -> :8080/v1
```

```python
# 应用代码完全不变,只换 base_url 就切引擎
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")  # 换成 :30000/v1 即用 SGLang
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "用一句话解释 PagedAttention"}],
    stream=True,                       # 流式 SSE
)
for chunk in resp:                     # 逐 token 到 data:[DONE]
    print(chunk.choices[0].delta.content or "", end="")
```

❌ 反模式:把引擎私有接口/私有字段硬编进应用,换引擎就得改一大片代码,被单一引擎锁死。
✅ 正解:**只依赖 OpenAI 兼容契约 + SSE**;前面架一个统一网关管鉴权/限流/路由,后面引擎随负载自由替换、混部、A/B,应用零改。

## 面试高频

- **「为什么六大引擎能互相替换?」** 因为 ①API 层都实现 OpenAI 兼容 `/v1/chat/completions` + SSE 流式;②③④ 差异再大,对应用都被这层标准口屏蔽,改 `base_url` 即换。
- **「/v1/chat/completions 比 /v1/completions 强在哪?」** 结构化 role-based `messages`、内建上下文管理、tools 函数调用;legacy 只做纯文本补全。
- **「流式怎么实现、和指标什么关系?」** SSE 逐 token 推 `data:{…}` + `data:[DONE]`;直接影响 TTFT(首 token)与 ITL(token 间隔)体验。
- **「网关层放这一层有什么用?」** 统一鉴权/限流/路由/计费/观测,后端多引擎多模型对客户端只露一个标准口,便于灰度与混部。
- **「这层抽象对应引擎架构的哪部分?」** 四层抽象的 ①API/前端;②调度 ③显存 ④后端才是各引擎真正差异与性能护城河所在。

## 关键事实

- `POST /v1/chat/completions` 是对话式 LLM 的**事实标准**,取代 legacy `/v1/completions`;请求用结构化 `messages`,支持 `stream`、`tools`、采样参数。
- 流式 = **SSE**:`data: {…}` 多行 + 终止 `data: [DONE]`;vLLM/SGLang/TGI 均实现。
- vLLM 的 OpenAI 兼容 server 基于 **FastAPI**(HTTP ↔ EngineCore 的桥);SGLang 有专门 `serving_chat.py`;TGI 由 Rust router 暴露。
- 引擎可热插拔的根因:各引擎 ①API 层对齐 OpenAI 形状,②③④ 实现差异被屏蔽 → 改 `base_url` 即换引擎。
- 实操参考 [[LLM/124 部署 vLLM：单卡 OpenAI 端点|vLLM 单卡 OpenAI 端点]]。
