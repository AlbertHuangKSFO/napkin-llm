[[119 流式输出：SSE、token streaming 与背压]] 关心**怎么把模型一边解码、一边把 token 推给用户**,而不是憋到整段生成完才一次性返回。自回归推理是逐 token 产出的,**流式(streaming)**就是把每个新 token 立刻封成事件推回客户端,让用户「看着字一个个蹦出来」。主流做法是 **SSE(Server-Sent Events)**:一次 HTTP 请求,服务器持续推回 `data:` 行,直到结束;OpenAI / Anthropic / Gemini 的流式接口都用它,见 [[094 OpenAI 兼容 API 与引擎抽象|OpenAI 兼容 API]]。它直接决定**首 token 时延(TTFT)**这一关键体验指标(见 [[105 SLO、SLA 设计：为推理定指标|SLO 设计]]),也带来一个反直觉的坑——**背压(backpressure)**:当客户端/网络消费 token 比引擎产得慢,SSE 没有应用层流控,缓冲会无界膨胀。流式还和 [[114 结构化输出：guided decoding 与 XGrammar|结构化输出]]、[[116 长上下文服务：streaming、attention sink 与 KV 预算|长上下文流式]] 协同,是 [[100 推理网关与智能路由(cache-aware)|推理网关]] 必须正确处理的一段链路。

## 直觉类比
非流式像**等厨师把整桌菜全做完再一起端上**:你盯着空桌干等几十秒。流式像**做好一道端一道**:第一道(首 token)上得越快,你越觉得「响应了」,哪怕全程总时长一样。背压则像**传菜员太慢、桌子放不下**:后厨还在猛出菜(引擎猛产 token),前厅消化不了,菜堆在传菜口(缓冲)越积越高——SSE 这条「单行传送带」没法让前厅喊「后厨慢点」,堆到爆就翻车(浏览器标签崩/服务端内存涨)。WebSocket 则是**双向对讲**:前厅停下不接,TCP 接收窗满,后厨自然被迫放慢。

## 小数字例子
设引擎稳定 **80 token/s**,一条回答 **800 token**。
- **非流式**:用户等约 $800/80=10$ s 才看到任何字,TTFT ≈ 10 s,体验「卡死」。
- **流式**:首 token 约 **40 ms** 就到(prefill 后第一个 decode),用户立刻看到字往外蹦,主观「秒回」,总时长仍约 10 s 但体验天差地别。
- **背压场景**:客户端在弱网/后台 tab 只能消费 **20 token/s**,引擎仍按 80 产 → 每秒积压 $80-20=60$ token。10 s 内堆 ~600 token 在缓冲里;若不设限,长会话下连接缓冲可涨到 MB 级,SSE 无流控时累积可拖崩标签或吃满服务端连接内存。

## 原理:逐 token 推送与背压
解码每步产一个 token,流式即在每步**立刻 flush** 一个事件。SSE 报文形如:

```text
data: {"choices":[{"delta":{"content":"你"}}]}\n\n
data: {"choices":[{"delta":{"content":"好"}}]}\n\n
data: [DONE]\n\n
```

设引擎产出速率 $r_{\text{gen}}$、客户端消费速率 $r_{\text{con}}$,缓冲积压随时间近似:

$$
B(t) \approx \max\!\big(0,\ (r_{\text{gen}} - r_{\text{con}})\cdot t\big)
$$

只要 $r_{\text{gen}} > r_{\text{con}}$ 且**无流控**,$B(t)$ 线性发散 → OOM/崩溃。**WebSocket** 经 TCP 流控天然回压:客户端停读 → 接收窗(receive window)填满 → 发送端阻塞 → 可反压到引擎暂停解码;**SSE 没有这条反向通道**,需在应用层手动模拟:监测发送缓冲(Node 的 `drain`、WS 的 `bufferedAmount`),满则暂停产 token。首 token 时延 TTFT 与每 token 间隔(ITL)是流式两大体验指标:

$$
\text{TTFT} = T_{\text{prefill}} + T_{\text{first-decode}}, \qquad \text{ITL} \approx \frac{1}{r_{\text{gen}}}
$$

![[srv-SSE流式背压.svg]]

并排看 SSE 单向传送带(无反向通道 → 缓冲发散)与 WebSocket 借 TCP 流控天然回压的本质差异:

![[srv-119SSE对比WebSocket.svg]]

## 配置 / 代码
```python
# FastAPI:用 SSE 逐 token 推送 + 简单背压保护
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import anyio, json

app = FastAPI()

@app.post("/v1/chat/completions")
async def stream(req: dict):
    async def gen():
        async for tok in engine.generate(req):       # 引擎逐 token 产出
            chunk = {"choices": [{"delta": {"content": tok}}]}
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            # 背压:让出事件循环,若下游写缓冲满,await 会自然阻塞→反压到引擎
            await anyio.sleep(0)
        yield "data: [DONE]\n\n"
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 关代理缓冲,逐块直达
    )
```

```text
❌ 非流式憋完整段才返回 → TTFT 爆(用户等十几秒);或 SSE 猛 flush 不看下游消费 → 慢客户端缓冲无界 → 标签崩/服务端内存涨,代理层还偷偷缓冲(Nginx proxy_buffering)让「流式」失效
✅ 逐 token flush、小块、关代理缓冲;监测发送缓冲满则暂停产 token;设单连接产出上限/超时;需取消/批准/多端交互再上 WebSocket 借 TCP 流控天然背压
```

## 面试高频
- **为什么 LLM 接口几乎都用 SSE 而不是 WebSocket?** LLM 是「一次请求→一长串单向 token」,正好是 SSE 的设计场景:普通 HTTP、`text/event-stream`、`EventSource` 自动重连、穿透代理简单;WebSocket 的双向能力在纯出字场景用不上,反而更重。
- **流式如何改善体验,关键指标是什么?** 把整段等待拆成「首 token 很快 + 逐字到达」,关键是 TTFT(首 token 时延)和 ITL(token 间隔);总时长可能不变但主观「秒回」。
- **什么是背压,SSE 为什么扛不住?** 引擎产 token 比客户端消费快时积压;SSE 是单向传送带,客户端无法在同连接上喊「慢点」,无应用层流控则缓冲无界膨胀 → 崩。
- **WebSocket 怎么天然背压?** 全双工 + TCP 流控:客户端停读 → 接收窗满 → 发送端阻塞 → 可反压到引擎暂停解码;SSE 需在应用层手动监测发送缓冲来模拟。
- **生产里怎么防背压翻车?** 小块切分勤 flush、关代理缓冲(`X-Accel-Buffering: no`)、监测发送缓冲满则暂停产、设单连接产出上限与超时、tab 隐藏时缓发;必要时 WS 控制 + SSE 数据混用。
- **为什么 Agent 工作流又开始上 WebSocket?** Agent 需在会话中回送信号(取消生成、批准工具调用、转向),SSE 同连接做不到双向,故常用 WS 控制通道 + SSE/WS 数据通道。

## 关键事实
- **SSE**(Server-Sent Events)是 LLM 流式事实标准:OpenAI、Anthropic、Google Gemini 的流式接口均用 SSE;2025–2026「先 SSE,需双向再 WebSocket」成为主流共识(从早年「默认 WebSocket」反转)。
- **SSE 无背压机制**:服务器只管发、客户端只管收,无法在同连接上让客户端喊「慢点」;产快于消费且无流控 → 缓冲膨胀可拖崩浏览器标签。
- **WebSocket 借 TCP 流控天然背压**:客户端停读 → 接收窗填满 → 发送端自然放慢;生产中常 WS 做控制(取消/批准/转向)、SSE 做数据,二者并用。
- 实践要点(2025):小块切分、`text/event-stream`、勤 flush、关代理缓冲(Nginx `proxy_buffering off` / `X-Accel-Buffering: no`)、监测 `bufferedAmount`/`drain` 做应用层背压。
- 首 token 时延 TTFT 是流式核心体验指标,与 ITL(每 token 间隔)一起进 SLO,见 [[105 SLO、SLA 设计：为推理定指标|SLO 设计]];长会话流式还需配 [[116 长上下文服务：streaming、attention sink 与 KV 预算|长上下文流式]] 控 KV 预算。
