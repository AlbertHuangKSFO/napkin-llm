[[15 Function Calling 工具调用|Function Calling 工具调用]] 是给模型一组带 JSON schema 的工具,让它输出**结构化调用**(工具名 + 参数)、由 runtime 真正执行、再把结果(observation)回灌上下文继续——它是把 LLM 接到外部世界、让 [[03 Agent 核心循环|Agent 核心循环]] 跑起来的那根「神经」。

模型本身从不执行任何东西:它只产生「我想调 `get_weather`、参数是 `{"city":"SF"}`」这个**意图**,执行权、副作用、错误处理全在 runtime 手里。这条边界是理解工具调用一切行为的钥匙。

## 本质:把「自由文本」夹成「结构化意图」

LLM 默认只会吐自由文本。要让它驱动真实动作,历史上有两条路:

- **纯 prompt 约定 + 正则解析**(早期 [[09 ReAct|ReAct]] 的做法):让模型按 `Action: Search[query]` 这种文本格式输出,再用正则抠出来。脆弱——模型一手抖格式(多个空格、漏方括号、把参数写成自然语言)解析就崩。
- **Function Calling**:把工具用 **JSON schema** 显式声明给模型,模型被**约束**着输出一个结构化对象(`{name, arguments}`),由平台保证它是合法 JSON、参数名对得上 schema。鲁棒得多,且能让模型「知道有哪些工具、各要什么参数」。

洞见在于:**模型不需要真的会执行,它只需要会「填表」**。给它一张表(schema:工具叫什么、干什么、要哪几个字段、各是什么类型),它负责判断「这次要不要填表、填哪张、每格填什么」,填完交给 runtime 去办。这就把不可控的自然语言,夹成了可被代码消费的结构化意图。

于是工具调用承担了 agent 里最关键的一次「**意图 → 动作**」转换:模型的语言能力 + runtime 的执行能力,在 schema 这个契约上对接。

## 机制:一轮工具调用分五步

![[Function Calling 工具调用.svg]]

1. **定义 tool schema**:开发者把每个工具声明成一段 JSON schema——`name`(模型据此选)、`description`(模型据此**判断该不该用、什么时候用**,极其重要,见 [[16 工具设计与工具层|工具设计与工具层]])、`parameters`(JSON Schema 描述每个参数的名/类型/是否必填/枚举)。这些 schema 随请求一起发给模型。
2. **模型决策**:模型读完用户消息 + 工具列表,自己决定三件事:**要不要调工具**(也可能直接答)、**调哪个(些)**、**每个参数填什么值**。它把决策编码成响应里的 `tool_calls` 字段(而非普通文本)。
3. **runtime 执行**:harness 解析出 `tool_calls`,**真正去调**对应的本地函数 / REST API / DB 查询 / 代码执行器。⚠️ 模型给的参数是不可信输入,执行前要校验、要做最小权限与沙箱。
4. **结果回灌**:把执行返回的原始结果包成一条 `tool`/`tool_result` 消息(带上对应 `tool_call_id` 做配对),**追加进对话历史**,连同之前的上下文重新发给模型。这步就是 [[09 ReAct|ReAct]] 里的 Observation。
5. **模型继续或收尾**:模型读到结果后,要么基于它再发起新的 `tool_call`(进入下一轮,这就是循环),要么觉得信息够了、直接产出最终自然语言答案(响应里不再有 `tool_calls`,循环终止)。

控制这个循环的是**外层 harness**,不是模型——和 [[09 ReAct|ReAct]] 完全同构:模型每轮只负责「下一步 tool_call 或收尾」,harness 负责执行、回灌、设最大轮数防死循环。事实上,**今天绝大多数 [[09 ReAct|ReAct]] 实现就是用 Function Calling 落地的**:把「解析 Action 文本」换成「解析 tool_calls 字段」,机制不变、鲁棒得多。

### 并行 tool calls

![[Function Calling 工具调用-并行调用.svg]]

现代模型支持在**一次响应里给出多个 `tool_calls`**。当几个调用**互不依赖**(比如同时查 SF 和 NYC 的天气),runtime 可以**并发执行**、把多条结果**一起回灌**,省掉一轮模型往返、显著降延迟。每个 call 带唯一 `id`,result 按 `id` 配对回去。

注意边界:只有**无依赖**的调用才能并行。若 `call_2` 的参数依赖 `call_1` 的结果(先查用户 id 再用 id 查订单),模型只能**串行**——先发 call_1、拿到结果、再发 call_2,这就是两轮。是否能并行由模型自己判断,但 schema 设计得好(工具粒度合理)能帮它更容易识别出可并行的批次。

### 流式与错误回灌

- **流式(streaming)**:`tool_calls` 的参数 JSON 可以**逐 token 流式**返回(`arguments` 字段一片片拼)。好处是 UI 能尽早显示「正在调用 X 工具」,但 runtime 必须等参数 JSON **完整**才能执行——半截 JSON 不能解析。
- **错误回灌**:工具执行失败(参数非法、API 超时、404)**不要直接抛给用户、也不要静默吞掉**。正确做法是把错误**当成一条 tool_result 回灌**(`{"error": "city 'Xyz' not found, did you mean ...?"}`),让模型**读到错误、自我纠正**(改参数重试、换工具、或换思路)。这正是 agent 自愈能力的来源,和 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 同源——错误信息越可读、可恢复,模型纠错越准(见 [[16 工具设计与工具层|工具设计与工具层]])。

## 来源 / 出处

- **OpenAI** 在 **2023 年 6 月**为 `gpt-3.5-turbo` / `gpt-4` 上线 **function calling**(最初是 `functions` + `function_call` 字段,2023 年 11 月演进为 `tools` + `tool_calls`,并支持并行)。这是工业界第一次把「模型输出结构化函数调用」做成一等公民 API。
- **Anthropic** 的 **tool use**(Claude)采用同构设计:请求里传 `tools`(每个含 `name`/`description`/`input_schema`),模型回 `tool_use` 块,你回 `tool_result` 块。语义与 OpenAI 一致,字段名不同。
- 两家(及后来 Google 等)都把工具定义建立在 **JSON Schema** 之上,这也成了 [[17 MCP 模型上下文协议|MCP 模型上下文协议]] 标准化工具暴露的基础。

## 可跑的最小实现

一个 `get_weather` 工具的完整 schema + 一轮请求 / 响应 / 回灌(OpenAI 风格,Anthropic 仅字段名不同):

```python
from openai import OpenAI
import json
client = OpenAI()

# ① 工具 schema —— description 与参数描述是模型选用的唯一依据,务必写清
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市当前天气。当用户问某地天气/温度/是否下雨时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名,如 'San Francisco'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "温度单位,默认 celsius"}
            },
            "required": ["city"]      # city 必填,unit 可选
        }
    }
}]

# 真正的执行函数(runtime 侧,模型看不到)
def get_weather(city, unit="celsius"):
    return {"city": city, "temp": 18, "unit": unit, "desc": "多云"}

messages = [{"role": "user", "content": "旧金山现在多少度?"}]

# ② 第一轮:模型决定调不调、填什么参数
resp = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
msg = resp.choices[0].message
messages.append(msg)                       # 把含 tool_calls 的助手消息加回历史

# ③ runtime 执行每个 tool_call,④ 结果回灌(按 tool_call_id 配对)
for call in msg.tool_calls or []:
    args = json.loads(call.function.arguments)   # 模型填的参数(不可信,生产需校验)
    try:
        result = get_weather(**args)
    except Exception as e:
        result = {"error": str(e)}              # 错误也回灌,让模型自纠
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,                 # 与请求配对
        "content": json.dumps(result, ensure_ascii=False)
    })

# ⑤ 第二轮:模型读到结果,产出自然语言答案(此轮通常不再有 tool_calls)
final = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
print(final.choices[0].message.content)   # "旧金山现在 18°C,多云。"
```

模型第一轮的响应里 `tool_calls` 长这样(这是模型「填的表」,不是它执行的结果):

```json
{
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": { "name": "get_weather", "arguments": "{\"city\":\"San Francisco\"}" }
  }]
}
```

要点:① schema 的 `description` 决定模型用不用、用对不对,是工具好坏的命门;② 模型给的 `arguments` 是**字符串化的 JSON、且不可信**,执行前 `json.loads` + 校验;③ 回灌必须带 `tool_call_id` 配对;④ 这段循环往复就成了完整的 [[03 Agent 核心循环|Agent 核心循环]]。

## 对比:文本协议 vs 结构化 Function Calling

| 维度 | 文本协议(早期 ReAct) | Function Calling(结构化) |
|---|---|---|
| 工具如何告知模型 | 写进 prompt 的散文描述 | JSON Schema(`tools` 字段) |
| 模型输出形态 | `Action: Search[q]` 字符串 | `tool_calls` 结构化对象 |
| 解析方式 | 正则 / 字符串切割,**脆弱** | 平台保证合法 JSON,**鲁棒** |
| 参数类型/必填 | 全靠模型自觉 | schema 约束、可校验 |
| 并行调用 | 几乎做不到 | 原生支持多 call |
| 流式 | 难 | 参数可逐 token 流 |
| 今天地位 | 教学/演示 | **生产默认** |

## 何时用 / 坑 / 反模式

**该用**:任何要让模型「碰外部世界」的场景——查数据、调 API、读写文件、跑代码、检索([[24 Agentic Search：grep vs 向量检索|Agentic Search：grep vs 向量检索]])。它是 agent 接外部的标准接口。

**坑**:
- **把模型当执行器**:模型只产意图,**所有副作用与权限校验必须在 runtime 兜住**。直接 `eval(模型给的代码)` 或拿模型给的 SQL 不校验就执行,是高危反模式。
- **错误不回灌**:工具报错直接抛异常中断,模型失去自纠机会。应包成 tool_result 回灌。
- **工具太多**:几十上百个工具塞进一次请求,模型**选择困难**、还吃光上下文。对策见 [[16 工具设计与工具层|工具设计与工具层]](分层、按需加载)与 [[17 MCP 模型上下文协议|MCP 模型上下文协议]]。
- **description 含糊**:模型不知道何时该用,要么漏调要么乱调。description 要写「**什么情况下调我**」。
- **不设 max_steps**:循环不收敛会烧 token、甚至死循环。harness 必须设最大轮数护栏。
- **参数当可信**:模型可能填出越权路径(`../../etc/passwd`)、危险命令。最小权限 + 沙箱 + 白名单。

**反模式**:用工具调用做**纯文本格式化**(比如「请按 JSON 返回」)——那该用结构化输出 / response_format,不是工具。工具是为了「做事」,不是为了「换个输出格式」。

## 关键事实

- Function Calling = 给模型**带 JSON schema 的工具** → 模型输出**结构化调用意图**(`tool_calls`)→ runtime **执行** → 结果(observation)**回灌** → 模型继续或收尾。
- **模型从不执行任何东西**,它只「填表」(选工具 + 填参数);执行权、副作用、权限、校验全在 runtime。这条边界是安全模型的根基。
- 工具的 **`description`** 是模型判断「用不用、何时用」的唯一依据,直接决定成败,详见 [[16 工具设计与工具层|工具设计与工具层]]。
- 支持**并行 tool calls**(无依赖时一次发多个、并发执行、一起回灌,省往返);有依赖只能串行。
- **错误也要回灌**(包成 tool_result),让模型自纠——这是 agent 自愈能力来源,与 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 同源。
- 它是 [[09 ReAct|ReAct]] 的现代落地方式(把文本 Action 换成结构化 tool_call),也是 [[17 MCP 模型上下文协议|MCP 模型上下文协议]] 把工具标准化暴露给模型的底层形态。
- 出处:OpenAI function calling(2023-06)、Anthropic tool use,均建立在 **JSON Schema** 之上。

## 主流开源实现 / Python 库

- **官方 SDK**:`openai`(`tools` + `tool_calls`)、`anthropic`(`tools` + `tool_use`/`tool_result`)——最底层、最直接,生产首选起点。
- **`567-labs/instructor`**(pip `instructor`,⚠️ owner 已从 `jxnl` 迁到 `567-labs`):基于 Pydantic 把工具调用包成「类型安全的结构化抽取」,自动校验 + 重试 + 流式,月下载 300 万+。**纯结构化输出/抽取场景的安全默认**。
- **`pydantic/pydantic-ai`**(pip `pydantic-ai`):需要完整 agent run、内建可观测性与 trace 时选它;比 instructor 更偏「跑 agent」而非「抽一次表」。
- **`dottxt-ai/outlines`**(pip `outlines`):**约束解码**路线——用有限状态机在生成时屏蔽非法 token,从源头保证只产出合法 schema(开源权重模型上尤其有用),与上面「事后校验」思路互补。
- 当下首选:接官方 SDK 自己管循环;只要结构化输出用 `instructor`;要约束解码兜底用 `outlines`。

## 工业界实践

### 各家 API 形态(同构、字段名不同)

| 厂商 | 工具定义字段 | 模型返回 | 结果回灌 | 并行 |
|---|---|---|---|---|
| **OpenAI** | `tools: [{type:"function", function:{name,description,parameters}}]` | `message.tool_calls[]`(含 `id`) | `role:"tool"` + `tool_call_id` | ✅ |
| **Anthropic (Claude)** | `tools: [{name, description, input_schema}]` | `content` 里的 `tool_use` 块(含 `id`) | `tool_result` 块 + `tool_use_id` | ✅ |
| **Google (Gemini)** | `function_declarations` | `functionCall` part | `functionResponse` part | ✅ |
| **开源权重(vLLM/Ollama 等)** | 多走 OpenAI 兼容层 | 视模型而定,质量参差 | 同 OpenAI | 视模型 |

三家都建立在 **JSON Schema** 之上,这也是 [[17 MCP 模型上下文协议|MCP]] 标准化工具暴露的基础。跨厂商时用 **LiteLLM** 抹平差异(统一成 OpenAI 格式),或用 LangChain/LlamaIndex 的工具抽象。

### 生产架构:工具调用循环长什么样

```
用户请求 → [系统 prompt + 工具 schema] → LLM
   → tool_calls? ──no──→ 返回最终答案
        │ yes
        ▼
   工具路由器(按 name 分发)→ 鉴权/限流/参数校验/沙箱
        ▼
   并发执行无依赖的 calls(线程池/async)→ 各自超时与重试
        ▼
   裁剪返回体(token-efficient)→ tool_result 回灌(按 id 配对)
        ▼
   回到 LLM(轮数 +1,超 max_steps 熔断)
```

要点:**循环控制权在外层 harness**(见 [[23 Agent Harness 概览|Agent Harness 概览]]),模型每轮只产「下一个 tool_call 或收尾」。这与 [[09 ReAct|ReAct]] 完全同构——今天绝大多数 ReAct 实现就是用 Function Calling 落地的。

### 工程化实践与最佳实践

- **路由与分发**:用 `{name: fn}` 字典做工具路由,别用一长串 `if/elif`。每个工具包一层「校验 → 执行 → 裁剪返回 → 异常转 tool_result」的标准 wrapper。
- **超时与重试**:每个工具单独设超时(外部 API 易卡);幂等的读类工具可自动重试,写类工具重试前先确认幂等(见 [[16 工具设计与工具层|工具设计与工具层]] 幂等原则)。
- **并发执行**:无依赖的多 call 用线程池(同步)或 `asyncio.gather`(异步)并发,把多条结果**一起回灌**,省一轮往返、显著降延迟。
- **参数校验当一等公民**:模型给的 `arguments` 是**字符串化 JSON 且不可信**——`json.loads` 后用 Pydantic / JSON Schema 校验,失败把校验错误**回灌**(而非抛异常),让模型自纠重填。`instructor` 把这套自动化(校验失败自动带错误重试)。
- **可观测**:每个 tool_call 记录 name/参数/耗时/成败/返回大小,trace 到 LangSmith/Langfuse/Phoenix/OpenTelemetry。**工具失败率、平均轮数、超 max_steps 比例**是 Agent 健康度的核心指标(见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]])。
- **约束解码兜底**:开源模型工具调用质量参差时,用 `outlines` / vLLM 的 guided decoding / OpenAI 的 Structured Outputs(`strict:true`)从源头保证只产合法 schema。

### 踩坑(生产实录)

- **`arguments` 是字符串不是对象**:OpenAI 的 `function.arguments` 是 JSON **字符串**,必须 `json.loads`;忘了会当 dict 用直接崩。
- **流式时参数半截不能执行**:`tool_calls` 的 `arguments` 逐 token 流式返回,必须等**完整**才能 `json.loads` 执行;UI 可先显示「正在调用 X」但 runtime 不能用半截 JSON。
- **工具太多致选择困难**:几十上百个工具塞一次请求,模型选错、还吃光上下文。对策:分层 / 按需加载(见 [[18 工具检索与动态加载|工具检索与动态加载]])、[[17 MCP 模型上下文协议|MCP]] 动态发现。
- **`tool_call_id` 配对漏了**:回灌的 tool_result 必须带对应 `id`,漏了或配错会让模型对不上结果而混乱。
- **模型幻觉工具名/参数**:可能调一个不存在的工具或填非 schema 字段。路由器要对未知工具名返回可读错误回灌,而非崩溃。
- **不设 max_steps**:循环不收敛会烧 token 甚至死循环,harness 必须设最大轮数硬护栏。
- **错误直接抛/静默吞**:都断了模型自纠链路。正确做法一律包成 tool_result 回灌(见下方面试 Q4)。

## 面试高频

**Q1:Function Calling 的完整流程?模型到底执行了什么?**
标准答:① 开发者把工具声明成带 JSON Schema 的 `tools`,随请求发给模型;② 模型决定「要不要调、调哪个、参数填什么」,编码进 `tool_calls`;③ runtime 解析并**真正执行**对应函数/API;④ 把结果包成 tool_result(带 `tool_call_id`)回灌进历史;⑤ 模型读结果后再发起新 call(进下一轮)或产出最终答案(无 tool_calls,循环终止)。**关键:模型从不执行任何东西**,它只产生「意图(填表)」,执行权、副作用、权限、校验全在 runtime——这条边界是安全模型的根基。

**Q2:早期 ReAct 的文本协议 vs 现代 Function Calling,差在哪?**
标准答:文本协议让模型按 `Action: Search[q]` 输出再用**正则解析**,模型一手抖格式就崩,脆弱;Function Calling 用 **JSON Schema 显式声明**工具,平台保证输出合法 JSON、参数名对得上 schema,鲁棒得多,且原生支持并行 call 与参数流式。今天文本协议只剩教学价值,Function Calling 是生产默认,也是 ReAct 的现代落地方式(把「解析 Action 文本」换成「解析 tool_calls 字段」)。

**Q3:并行 tool calls 的条件是什么?** 只有**互不依赖**的调用才能并行(如同时查 SF 和 NYC 天气),runtime 并发执行、一起回灌、省一轮往返。若 `call_2` 的参数依赖 `call_1` 的结果(先查用户 id 再用 id 查订单),只能**串行**(两轮)。是否并行由模型判断,但工具粒度设计合理能帮它更易识别可并行批次。

**Q4(高频陷阱):工具执行报错了怎么办?** **不要直接抛给用户,也不要静默吞掉**——把错误**当成一条 tool_result 回灌**(如 `{"error":"city 'Xyz' not found, did you mean...?"}`),让模型读到错误、自我纠正(改参数重试/换工具/换思路)。这正是 Agent 自愈能力的来源,与 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]] 同源;错误信息越可读可恢复,模型纠错越准(见 [[16 工具设计与工具层|工具设计与工具层]])。

**Q5:决定工具调用成败最关键的字段是什么?** 工具的 **`description`**——它是模型判断「用不用、何时用」的**唯一依据**。要写「**什么情况下调我**」(触发条件)而非只写「是什么」。含糊的 description 会导致漏调或乱调。详见 [[16 工具设计与工具层|工具设计与工具层]]。

**Q6(安全陷阱):为什么说「把模型当执行器」是高危反模式?** 模型只产意图、且给的参数**不可信**(可能填出越权路径 `../../etc/passwd` 或危险命令/SQL)。所有副作用与权限校验**必须在 runtime 兜住**:最小权限 + 沙箱 + 白名单 + 参数校验。直接 `eval(模型给的代码)` 或拿模型 SQL 不校验就执行,是典型的工具滥用与意外代码执行风险。

**Q7:用工具调用来「让模型按 JSON 格式输出」对吗?** 不对,这是反模式。纯文本格式化该用**结构化输出 / `response_format` / Structured Outputs**,不是工具。工具是为了「**做事**」(碰外部世界),不是为了「换个输出格式」。

## 知识拓展

- **谱系定位**:Function Calling 是把 LLM 接到外部世界、让 [[03 Agent 核心循环|Agent 核心循环]] 跑起来的「神经」,是 [[09 ReAct|ReAct]] 的现代落地形态;上层是 [[16 工具设计与工具层|工具设计与工具层]](喂什么样的工具)、[[17 MCP 模型上下文协议|MCP]](标准化暴露)、[[18 工具检索与动态加载|工具检索与动态加载]](工具太多时按需加载);错误回灌思想接 [[13 Reflection 与 Reflexion|Reflection 与 Reflexion]];安全侧接最小权限/沙箱/人审闸门。
- **机制对照**:
  - **结构化输出(Structured Outputs / JSON mode)**:与工具调用底层同源(都约束模型产合法 schema),但目的是「输出格式」而非「做事」。OpenAI 的 `strict:true` 用约束解码保证 100% schema 合规(2024)。
  - **约束解码(constrained decoding)**:`outlines`、vLLM guided decoding、llama.cpp grammar——在生成时用有限状态机/语法**屏蔽非法 token**,从源头保证 schema 合法,与「事后校验」(instructor)互补,开源模型尤其有用。
  - **Computer Use / 浏览器 Agent**:工具调用的极端形态——工具是「点击/输入/截屏」,模型输出坐标与动作(见 [[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]])。
- **前沿与趋势(带年份)**:
  - OpenAI function calling(2023-06)→ `tools`+并行(2023-11)→ Structured Outputs `strict`(2024)。
  - Anthropic tool use(Claude),后推出 **MCP**(2024-11)把工具暴露标准化为开放协议(见 [[17 MCP 模型上下文协议|MCP 模型上下文协议]])。
  - **Toolformer**(Schick et al., 2023):用自监督让模型**自己学会何时调 API**,是「工具调用能力训练化」的早期工作。
  - **Gorilla / APIBench**(Patil et al., 2023)、**ToolLLM / ToolBench**(Qin et al., 2023):大规模 API 调用能力的训练与评测基准。
  - **Berkeley Function-Calling Leaderboard (BFCL)**:工业界事实上的工具调用能力排行榜,衡量模型选工具/填参数/并行/多轮的准确率。
- **边界 / 反模式**:① 把模型当执行器、参数当可信;② 错误不回灌断自愈;③ 工具爆炸致选择困难;④ description 含糊;⑤ 不设 max_steps;⑥ 拿工具调用做纯文本格式化。
