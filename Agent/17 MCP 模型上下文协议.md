[[17 MCP 模型上下文协议|MCP 模型上下文协议]](Model Context Protocol)是一套**标准化「应用怎么给 LLM 喂上下文与工具」的开放协议**:client-server 架构,server 暴露三种原语——**Tools / Resources / Prompts**,host 应用里的 client 连上去消费。一句话——**它是 AI 应用接外部能力的「USB-C 口」**。

它和 [[15 Function Calling 工具调用|Function Calling 工具调用]] 的关系:Function Calling 是「模型怎么调一个工具」的**底层机制**;MCP 是「工具怎么被标准化地暴露、发现、连接」的**上层协议**。MCP server 暴露的 tool,最终还是通过 Function Calling 的形态被模型调用——MCP 解决的是「**工具从哪来、怎么接进来**」。

## 本质:用一个标准口子,消灭 N×M 集成爆炸

![[MCP 模型上下文协议-NxM 集成.svg]]

在 MCP 之前,要让某个 AI 应用接某个外部系统(GitHub、Slack、数据库…),得**为这对组合专门写一份集成**。N 个应用 × M 个工具 = **N×M 套各不相同的定制代码**,每加一个应用或一个工具都要重写一遍,且彼此不复用——这就是集成爆炸。

MCP 的洞见和 **LSP(Language Server Protocol)之于编辑器、USB-C 之于外设**完全同构:**定一个标准协议,把两两定制变成各连一次标准口**。

- **工具方**只需写一个 **MCP server**(把自己的能力按协议暴露一次),**所有**支持 MCP 的应用就都能用它。
- **应用方**只需写一个 **MCP client**(实现协议),就能接**所有** MCP server。

于是 N×M 塌缩成 **N+M**:N 个应用各实现一次 client,M 个工具各实现一次 server,中间靠协议对接。这是 MCP 最根本的价值主张——**把集成从「平方级」降到「线性级」**,并催生一个可共享、可复用的 server 生态(社区已有成百上千个现成 MCP server)。

## 机制:client-server 架构与三原语

![[MCP 模型上下文协议.svg]]

**角色**:
- **Host(宿主)**:用户实际用的 AI 应用——Claude Desktop、IDE 插件、或你自己写的 agent。它管理若干 client、持有 LLM、对接用户。
- **Client(客户端)**:host 内部的连接器,**与一个 server 保持 1:1 连接**。host 想连多个 server,就开多个 client。
- **Server(服务端)**:一个独立进程/服务,把某个系统的能力按 MCP 协议暴露。每个 server 聚焦一个域(GitHub server、文件系统 server、数据库 server…)。

**server 暴露的三种原语**(这是 MCP 的核心抽象,各自有不同的「控制权归属」):

1. **Tools(工具)——模型控制**:模型可以**主动调用**的动作,带 JSON Schema 描述输入(`create_issue`、`run_query`、`send_message`)。这正是 [[15 Function Calling 工具调用|Function Calling 工具调用]] 那套 schema,只是来源换成了 server。**有副作用、能改世界**,所以由模型在推理中决定何时调。
2. **Resources(资源)——应用控制**:可被读取的**上下文数据**,用 URI 寻址(`file:///path`、`postgres://table/row`)。它是**只读**的、给模型当背景信息(一份文档、一段数据库记录、一个目录树)。通常由 host 应用决定把哪些 resource 塞进上下文,而非模型乱读。
3. **Prompts(提示)——用户控制**:server 预制的**可复用提示模板 / 工作流**(「帮我 review 这个 PR」「按这个模板写周报」),通常作为用户可显式触发的命令(如斜杠命令)。它把「怎么用好这个 server」的最佳实践固化下来。

这三者的「**Tools 模型控,Resources 应用控,Prompts 用户控**」的控制权划分,是 MCP 设计里很讲究的一点——它清晰地分开了「谁决定这块上下文/动作何时进入对话」。

**传输与协议**:
- 底层用 **JSON-RPC 2.0** 做消息格式(请求/响应/通知)。
- 两种**传输**:**stdio**(server 作为本地子进程,通过标准输入输出通信——适合本地工具,如文件系统、本地脚本)、**HTTP + SSE**(Server-Sent Events,适合远程 server;后续演进为 Streamable HTTP)。
- **发现与调用流程**:client 连上 server 后,先 `initialize` 握手协商能力,再 `tools/list` / `resources/list` / `prompts/list` **发现**对方提供了什么,host 把发现到的 tools 转成模型可见的工具列表;模型决定调用时,client 发 `tools/call`,server 执行并返回结果,host 把结果回灌给模型(这步就接回 [[15 Function Calling 工具调用|Function Calling 工具调用]] 的回灌)。

MCP server/client 这一层,正是 [[16 工具设计与工具层|工具设计与工具层]] 里说的「**工具层常由 MCP client 实现**」——MCP client 把多个 server 的能力发现、汇聚、统一成 agent 看到的工具集。

## 来源 / 出处

- **Anthropic 于 2024 年 11 月开源 MCP**(规范 + SDK + 一批参考 server),定位是「给 LLM 应用接外部数据与工具的开放标准」。
- 2025 年迅速成为**事实行业标准**:**OpenAI**(Agents SDK / 产品)、**Google**(及 DeepMind 相关)、以及大量厂商、IDE、agent 框架纷纷支持 MCP。它从「Anthropic 的协议」变成了跨厂商的通用层。
- 规范、SDK(Python/TypeScript 等)与 server 目录均公开;社区贡献了海量现成 server(GitHub、Slack、Postgres、文件系统、浏览器…)。

## 可跑的最小实现

**最小 MCP server**(Python,FastMCP 风格伪码),暴露一个 tool 和一个 resource:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")          # 一个聚焦「天气」域的 server

# 暴露一个 Tool(模型控制:模型可主动调,有 schema)
@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询指定城市当前天气。用户问某地天气/温度时调用。

    Args:
        city: 城市名,如 'San Francisco'
        unit: 温度单位 celsius / fahrenheit,默认 celsius
    """
    # 函数签名 + docstring 会被自动转成 MCP 的 tool schema(含 JSON Schema 参数)
    return {"city": city, "temp": 18, "unit": unit, "desc": "多云"}

# 暴露一个 Resource(应用控制:只读上下文数据,URI 寻址)
@mcp.resource("weather://stations")
def list_stations() -> str:
    """所有气象站清单,供模型了解可查范围。"""
    return "SF, NYC, Tokyo, London"

if __name__ == "__main__":
    mcp.run(transport="stdio")           # 本地以子进程方式跑,经 stdin/stdout 通信
```

**client 侧**(host 应用如何连上并调用,伪码):

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["weather_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                 # ① 握手协商能力

            tools = await session.list_tools()          # ② 发现:有哪些 tool
            # → host 把这些 tool 转成模型可见的工具列表(Function Calling 形态)

            # ③ 模型决定调用后,client 代发 tools/call
            result = await session.call_tool(
                "get_weather", {"city": "San Francisco"}
            )
            print(result.content)        # → 回灌给模型继续(接 Function Calling 回灌)
```

要点:① server 端用函数签名 + docstring **自动生成 tool schema**——所以 [[16 工具设计与工具层|工具设计与工具层]] 的原则(清晰命名、docstring 写「何时用」、参数带描述)在这里同样适用;② client `initialize` 后先 `list` 发现、再 `call` 调用,发现是动态的;③ stdio 传输适合本地,远程换 HTTP+SSE,server 代码几乎不变。

## 对比:有 / 无 MCP

| 维度 | 无 MCP(两两定制) | 有 MCP(标准协议) |
|---|---|---|
| 集成数量 | N×M(平方级) | N+M(线性级) |
| 加一个新工具 | 每个应用各改一遍 | 写一个 server,所有应用即用 |
| 加一个新应用 | 每个工具各接一遍 | 写一个 client,接所有 server |
| 复用 | 几乎为零 | 共享 server 生态 |
| 暴露的抽象 | 各自为政 | 统一三原语 Tools/Resources/Prompts |
| 传输 | 各写各的 | stdio / HTTP+SSE,JSON-RPC |
| 类比 | 每种设备配专用线 | USB-C / LSP |

## 何时用 / 坑 / 反模式

**该用**:你的 agent 要接**多个**外部系统、或想**复用**社区现成集成、或想让你的工具被**别家应用**也能用——MCP 是当下最标准的答案。它也让 [[23 Agent Harness 概览|Agent Harness 概览]] 的工具层有了通用接入方式。

**坑 / 安全**:
- **MCP server 是要执行真实动作的进程**,装第三方 server 等于在你的 host 里引入可执行代码 + 外部连接——**供应链与权限风险真实存在**。只装可信来源、最小权限、敏感操作加确认。
- **[[05 Prompt Injection 提示注入|prompt injection]] 经 resource/tool 结果回流**:server 返回的内容会进模型上下文,恶意 server 或被污染的数据可注入指令。要把 server 输出当不可信处理(详见 [[20 Agentic 供应链与 MCP 安全|MCP 安全]])。
- **工具过多**:连一堆 server 会把工具列表撑爆,触发 [[16 工具设计与工具层|工具设计与工具层]] 说的选择困难 + 上下文膨胀。按需连、按需暴露。
- **凭证管理**:OAuth / token 在 client 或 server 侧管理,别让模型看到密钥。

**反模式**:
- 把 MCP 当成「**又一个 REST 网关**」——它的价值是**标准化的发现 + 三原语抽象 + 生态复用**,不是单纯转发 HTTP。只为一个私有工具、且永不复用,直接写本地 tool 函数可能更简单。
- 把本该是只读上下文的东西做成 Tool(让模型乱调),或把有副作用的动作塞进 Resource——**搞反了三原语的控制权归属**。

**与相关概念的边界**:MCP 解决「工具/上下文怎么标准接入」;[[25 Agent Skills(SKILL.md)|Agent Skills(SKILL.md)]] 解决「怎么把一段可复用的能力/流程打包给 agent」;二者互补——skill 可以内部调用 MCP 暴露的工具。

## 关键事实

- MCP = **Model Context Protocol**,Anthropic **2024-11** 开源,2025 年成跨厂商(OpenAI/Google 等跟进)**事实标准**。
- 价值:把集成从 **N×M 降到 N+M**,思路同 **LSP / USB-C**——一个标准口消灭两两定制,催生可复用 server 生态。
- 架构:**Host(含若干 Client)↔ Server**,每个 **Client 与一个 Server 1:1**,Host 可同时连多个 Server。
- 三原语 + 控制权:**Tools(模型控,有副作用的动作)/ Resources(应用控,只读上下文,URI 寻址)/ Prompts(用户控,可复用模板)**。
- 底层 **JSON-RPC 2.0**;传输 **stdio(本地)/ HTTP+SSE(远程)**;流程 **initialize 握手 → list 发现 → call 调用 → 结果回灌**。
- 与 [[15 Function Calling 工具调用|Function Calling 工具调用]] 互补:MCP 管「工具怎么标准暴露/发现/接入」,Function Calling 管「模型怎么调一个工具」;MCP tool 最终以 Function Calling 形态被调用。
- **工具层常由 MCP client 实现**(见 [[16 工具设计与工具层|工具设计与工具层]]),是 [[23 Agent Harness 概览|Agent Harness 概览]] 接外部能力的标准方式。
- server 的 tool schema 由函数签名 + docstring 自动生成,故 [[16 工具设计与工具层|工具设计与工具层]] 的好工具原则在 MCP server 上同样成立。
- 安全要点:第三方 server = 引入可执行代码 + 外部连接,有供应链/注入/权限风险,需可信来源 + 最小权限 + 敏感操作确认。

## 主流开源实现 / Python 库

- **`modelcontextprotocol/python-sdk`**(pip `mcp`):官方 Python SDK,server/client 都能写,协议演进的权威实现——**追协议最新规范、做底层控制时首选**。
- **`PrefectHQ/fastmcp`**(pip `fastmcp`,⚠️ owner 已从个人 `jlowin` 迁到组织 `PrefectHQ`,2026 年发布 FastMCP 3.0 GA):Pythonic 高层封装,装饰器写 server/client,生态据称支撑约 70% 的 MCP server;FastMCP 1.0 曾被并入官方 SDK。**绝大多数业务 server 的首选**(上面最小实现用的 `from mcp.server.fastmcp import FastMCP` 即官方 SDK 内置的 FastMCP 1.0 一脉)。
- 当下首选:日常写 server 用 `fastmcp`(开发体验最好);要贴最新协议细节或定制传输层用官方 `mcp`。

## 工业界实践

MCP 在 2025 年完成了从「Anthropic 的一个协议」到「跨厂商基础设施层」的转变,工业界落地已经成体系,踩坑也踩出了一套最佳实践。

### 协议本身的演进(规范跟着生产需求长)

- **传输从 SSE 收敛到 Streamable HTTP**:最初远程传输是 HTTP + SSE,但生产里 SSE 的「长连接 + 双通道」对负载均衡、横向扩展极不友好。**2025-03 规范 deprecate 了纯 SSE,推出 Streamable HTTP**(单端点、可选 SSE 流、支持无状态),这才真正解锁了一波远程 server 的规模化部署。stdio 仍是本地子进程(文件系统、本地脚本)的首选。
- **规范版本按日期命名**:`2024-11`(初版)→ `2025-03`(Streamable HTTP)→ `2025-06-18`(授权大改)→ **`2025-11-25`**(MCP 一周年版,引入实验性 **Tasks** 能力、把异步长任务标准化)。看 MCP server 文档先认版本号。
- **授权(Authorization)是生产化的最大补丁**:`2025-06-18` 起把 **MCP server 正式定义为 OAuth 2.1 资源服务器**,强制 **PKCE**;`2025-11-25` 进一步要求 **RFC 8707 Resource Indicators**(token 绑定具体 server,防「拿 A 的 token 去骗 B」的 confused-deputy 攻击)、**RFC 9728 Protected Resource Metadata**(server 通过 `.well-known` 广告自己的授权服务器位置),并用 **Client ID Metadata Documents(CIMD)** 取代繁琐的动态客户端注册(DCR)。一句话:**远程公开 server 现在必须跑完整 OAuth 2.1 流程**,凭证永远不进模型上下文。

### MCP Registry 与生态规模

- **官方 MCP Registry 于 2025-09-08 上线预览**(开放的 server 目录 + API),解决「server 满天飞、没处发现、版本与可信度难辨」的问题——相当于 MCP 的「npm registry」。到 2026 年初已收录近两千个条目。这把「N+M」里的 M 端从「自己写」推向「目录里找现成的」。
- **超出 Tools 的三原语都在被用**:除了最热的 Tools,生产里 Resources 常用来挂「只读上下文」(知识库快照、配置),Prompts 被 IDE/Claude Desktop 做成斜杠命令。`2025-11-25` 还在推 **elicitation**(server 反向向用户要输入,如缺参数时弹个表单)、**roots**(client 告诉 server 可访问的文件根)、**sampling**(server 反过来请求 client 侧 LLM 补全)等双向能力。

### 典型生产架构与运维

```text
[Host: IDE / Claude Desktop / 自研 Agent]
        │  (一个 Client ↔ 一个 Server,1:1)
        ├── stdio ──→ 本地 server(文件系统、git、本地 DB)
        └── Streamable HTTP + OAuth2.1 ──→ 远程 server(GitHub/Slack/内部 API)
                                              ↑
                              [MCP Gateway / 反向代理]:鉴权、限流、审计、
                              工具聚合与筛选(防工具爆炸)、多 server 路由
```

- **规模化靠网关**:server 一多(几十个、几百个工具),直接全连会撞 [[18 工具检索与动态加载|工具检索与动态加载]] 说的工具爆炸 + token 爆。生产里普遍在 host 和一堆 server 之间架 **MCP Gateway**(如 `microsoft/mcp-gateway`),统一做鉴权、限流、审计日志、按需暴露工具子集、跨 server 路由。
- **可观测性**:把 `tools/call` 当 RPC 调用埋点——每次调用记 server、tool、参数(脱敏)、耗时、成功/失败、token 占用。MCP 的 JSON-RPC 结构天然适合做 trace span,接 [[38 Agent 评估与可观测性|Agent 评估与可观测性]] 那套(OpenTelemetry / Langfuse)。
- **成本/延迟**:远程 server 多一跳网络 RTT;工具 schema 常驻上下文每轮重发,工具多了直接抬高每轮 token 单价——这正是 `2026-01` Anthropic **Tool Search**(`defer_loading`)要解决的(详见 [[18 工具检索与动态加载|工具检索与动态加载]])。

### 踩坑与最佳实践

- **第三方 server = 引入可执行代码 + 外部连接**:供应链风险真实。生产里只用 Registry 里可信来源、固定版本(别用 `latest`)、最小权限、敏感操作(删库、转账)强制人工确认。
- **server 输出当不可信处理**:tool/resource 返回会进上下文,是 [[05 Prompt Injection 提示注入|prompt injection]] 的注入面(详见 [[20 Agentic 供应链与 MCP 安全|MCP 安全]])。
- **别把三原语用反**:有副作用的动作 → Tool;只读上下文 → Resource;用户触发的模板 → Prompt。把只读数据做成 Tool 会让模型乱调、把危险动作塞进 Resource 会绕过审批。
- **token 治理**:连 server 按需连、按需暴露工具,配合工具检索;别一股脑把十几个 server 几百个工具全挂上。

## 面试高频

**Q1:MCP 和 Function Calling 是什么关系?是不是一回事?**
不是。Function Calling 是「**模型怎么调一个工具**」的底层机制(模型按 JSON Schema 生成调用、框架执行、结果回灌);MCP 是「**工具怎么被标准化地暴露、发现、连接进来**」的上层协议。MCP server 暴露的 tool,最终仍以 Function Calling 形态被模型调用。一句话:**Function Calling 管「怎么调」,MCP 管「工具从哪来」**。
- 追问「那没有 MCP 就不能 Function Calling 了?」——能。Function Calling 不依赖 MCP;MCP 只是给工具的来源/接入做了标准化,解决 N×M 集成爆炸。

**Q2:MCP 到底解决了什么核心问题?为什么类比 USB-C / LSP?**
解决 **N×M 集成爆炸**:N 个应用 × M 个工具,每对组合都要定制集成。MCP 定一个标准协议,工具方写一次 server、应用方写一次 client,集成从平方级 N×M 塌缩成线性级 **N+M**,并催生可复用的 server 生态。同构于 LSP(一个协议让任意编辑器接任意语言)、USB-C(一个口接任意外设)。
- 陷阱:别答成「MCP 是个 API 网关 / REST 封装」。它的价值是**标准化的发现 + 三原语抽象 + 生态复用**,不是转发 HTTP。

**Q3:MCP 的三原语是什么?控制权各归谁?**
**Tools(模型控制)**——有副作用、可改世界的动作,模型推理中主动调;**Resources(应用控制)**——只读上下文数据,URI 寻址,由 host 决定塞哪些进上下文;**Prompts(用户控制)**——可复用提示模板/工作流,用户显式触发(如斜杠命令)。记忆点:**Tools 模型控、Resources 应用控、Prompts 用户控**。
- 追问「为什么要分控制权?」——清晰界定「谁决定这块上下文/动作何时进入对话」,避免模型乱读数据或乱执行动作。

**Q4:MCP 的传输层有哪几种?各自适用?**
底层 **JSON-RPC 2.0**。传输:**stdio**(server 作本地子进程,经 stdin/stdout,适合本地工具)、**Streamable HTTP**(适合远程 server,单端点、可选 SSE 流、对负载均衡友好)。注意:**早期的纯 HTTP+SSE 已在 2025-03 被 deprecate**,被 Streamable HTTP 取代。
- 陷阱:还在背「HTTP+SSE」的就暴露了知识停留在 2024;能点出 Streamable HTTP 和 deprecation 是加分项。

**Q5:MCP 调用的完整流程?**
client `initialize` 握手协商能力 → `tools/list` / `resources/list` / `prompts/list` **发现**对方能力 → host 把 tools 转成模型可见工具列表 → 模型决定调用 → client 发 `tools/call` → server 执行返回 → host 把结果**回灌**给模型(接回 Function Calling 回灌)。关键词:**握手 → 发现 → 调用 → 回灌**,发现是动态的。

**Q6:用 MCP 有哪些安全风险?怎么防?**
① **供应链**:第三方 server 是可执行进程 + 外连,只用可信来源、固定版本、最小权限;② **prompt injection**:server 返回内容会进上下文,当不可信处理;③ **凭证泄露**:OAuth token 在 client/server 侧管,绝不进模型上下文;④ **confused deputy**:用 RFC 8707 Resource Indicators 把 token 绑死到具体 server;⑤ **工具爆炸**:按需连、网关筛选。
- 陷阱:只答「prompt injection」不够全;能讲到 OAuth 2.1 / PKCE / Resource Indicators 说明跟到了 2025 规范。

## 知识拓展

- **MCP 一周年与 Tasks(2025-11-25)**:一年从开源协议长成事实标准。`2025-11-25` 版引入**实验性 Tasks**——把「异步长任务」标准化(server 接活后返回 task id,client 轮询/订阅进度),呼应 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]] 的长任务编排需求。这是协议从「同步 RPC」迈向「异步作业」的关键一步。
- **MCP vs A2A——两个不同层**:MCP 是「**agent ↔ 工具/数据**」的纵向接入协议;[[30 A2A 协议|A2A 协议]] 是「**agent ↔ agent**」的横向协作协议。两者互补不竞争:一个 agent 用 MCP 接外部能力、用 A2A 和别的 agent 对话。面试常把二者混为一谈,要能划清。
- **MCP 与 Agent Skills 的边界**:MCP 解决「工具/上下文怎么标准接入」;[[25 Agent Skills(SKILL.md)|Agent Skills(SKILL.md)]] 解决「怎么把一段可复用的能力/流程打包给 agent」。二者互补——一个 skill 内部可以调 MCP 暴露的工具;skill 是「方法论 + 流程」,MCP server 是「能力的来源」。
- **反模式回顾**:把 MCP 当「又一个 REST 网关」;只为单个私有、永不复用的工具也硬上 MCP(直接写本地 tool 函数更省);三原语控制权用反(只读数据做成 Tool、危险动作塞进 Resource)。
- **前沿方向(2026 路线图)**:规范侧在解决 Streamable HTTP 的**有状态会话 vs 横向扩展**矛盾(无状态化、标准化 `.well-known` 能力发现)、推进 elicitation/roots/sampling 等**双向交互**、以及 Registry 的可信度与签名机制。趋势是 MCP 从「连工具」走向「连一个有状态、可异步、可信任的能力网络」。
- 关联:[[15 Function Calling 工具调用|Function Calling 工具调用]](底层调用机制)、[[16 工具设计与工具层|工具设计与工具层]](工具层常由 MCP client 实现)、[[18 工具检索与动态加载|工具检索与动态加载]](MCP server 一多就要检索式按需暴露)、[[23 Agent Harness 概览|Agent Harness 概览]](MCP 是 harness 接外部能力的标准方式)。
