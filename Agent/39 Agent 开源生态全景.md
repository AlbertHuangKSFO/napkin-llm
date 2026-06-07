[[39 Agent 开源生态全景|Agent 开源生态全景]]不是一份"框架排行榜",而是一张**按职责分层的地图**:它告诉你某个需求该去哪一层找轮子、同层有哪些互斥竞品、跨层怎么搭、以及——最重要的——**哪些层你现在根本不该碰**。

## 本质:为什么要一张全景,而不是一份清单

OSS agent 生态有两个让人头疼的特征:**碎**和**快**。

碎,是因为"做一个 Agent"被拆成了十几种正交的职责:把 [[03 Agent 核心循环|Agent 核心循环]]收成代码(编排)、给模型装手([[15 Function Calling 工具调用|Function Calling 工具调用]]/[[17 MCP 模型上下文协议|MCP 模型上下文协议]])、让它跨会话记事([[19 Agent 记忆系统|Agent 记忆系统]])、上线后能看见它在干嘛([[38 Agent 评估与可观测性|Agent 评估与可观测性]])、把提示当参数自动调([[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]])、不够了就改权重([[32 Agentic RL 与训练|Agentic RL 与训练]])、跑几小时不能死([[34 Agent 部署与持久化执行|Agent 部署与持久化执行]])……每一项都有三五个竞品库在抢,且彼此**不在一个层面**。

快,是因为这堆库平均每周一个版本,**org 还经常改名搬家**。autogen 搬成了 `ag2ai/ag2`,phidata 改名 `agno-agi/agno`,`jlowin/fastmcp` 随 3.0 并入 `PrefectHQ/fastmcp`,`princeton-nlp/SWE-agent` 独立成 `SWE-agent/SWE-agent`,`volcengine/verl` 转到了 `verl-project/verl`。你昨天 star 的 URL,今天可能 301 跳转。

所以"罗列一百个库"毫无用处——三个月就过期,而且没回答真正的问题:**我这个需求,该用哪个?现在该不该上?** 全景图的价值在于把这些库**摁进固定的层**,层是稳定的(职责不变),库是流动的(谁火了换谁)。你记住"记忆这一层 2026 主流是 mem0/letta/zep",比记住某个具体版本号有用得多。

下面这张图是全文的骨架,后面每一层都对它做展开:

![[Agent 开源生态全景.png]]

读图三句话:**颜色只分层、不分优劣**;**同层多个库≈互斥竞品**(选一个就够),**跨层多个库≈互补搭配**(可以叠);一个真实项目通常横跨 2~4 层,而不是把某一层装满。

## 按层拆解的生态全景

### ① 编排框架——决定你"怎么写"的骨架

**定位**:把 [[03 Agent 核心循环|Agent 核心循环]](想—调工具—观察—再想)、状态管理、[[22 多智能体系统|多智能体系统]]协作收敛成一套你能调用的代码抽象。这是你接触最多、锁定最深的一层,选错了重写成本最高。所有具体写法对比见 [[37 Agent 框架对比|Agent 框架对比]]。

代表库(owner/repo · pip 名 · 一句话选型):

- **`langchain-ai/langgraph`**(`langgraph`)——把 Agent 显式建成**有状态的图**,节点+边+checkpoint。要可控性、复杂分支循环、Human-in-the-Loop 人审、[[34 Agent 部署与持久化执行|中断恢复]]时选它。代价是抽象重、上手陡。2026 已到 1.x,生产派首选。
- **`crewAIInc/crewAI`**(`crewai`)——"一队角色 agent 各司其职"的心智模型,搭原型极快。从零自研、不依赖 LangChain。要快速做"研究员+写手+审稿"这类协作 demo 时选它。
- **`ag2ai/ag2`**(`ag2`,**前身 `microsoft/autogen`**)——多 Agent 对话编排的老牌,AutoGen 0.2 代码无改即可迁移。研究多智能体交互、群聊式协作时选它。
- **`huggingface/smolagents`**(`smolagents`)——核心逻辑约 1000 行,主打 **CodeAct**(让模型写 Python 代码当动作,而非填 JSON)。要极简、少黑盒、模型无关时选它。
- **`pydantic/pydantic-ai`**(`pydantic-ai`)——把 Pydantic 的类型安全带进 Agent,FastAPI 式手感,2026 V2 在 beta。重结构化输出、强类型、想看清每一步时选它。
- **`agno-agi/agno`**(`agno`,**前身 phidata**,pip 名已从 `phidata` 换成 `agno`)——主打多模态 + 性能,实例化极快。要轻量高吞吐 agent 平台时选它。
- **`openai/openai-agents-python`**(`openai-agents`)——OpenAI 官方轻量多 Agent 框架,内建 handoff、guardrail、tracing,2026 加了 sandbox agents(跑命令/改代码的长程任务)。吃 OpenAI 全家桶时最顺。
- **`google/adk-python`**(`google-adk`)——Google 官方,2026 ADK 2.0 GA 带图工作流+协作 agent,与 Vertex/Gemini 深度绑定。在 GCP 上做生产时选它。
- **`run-llama/llama_index`**(`llama-index`,核心 `llama-index-core`)——RAG 出身,如今 Workflows 1.0 是事件驱动的 agent 编排。强 [[36 Agentic RAG|Agentic RAG]]、文档密集场景时选它。

**这一层的选型逻辑**见专门的决策图:

![[Agent 开源生态全景-选型.png]]

一句话压缩:**要可控 → langgraph;要快搭协作 → crewAI/ag2/agno;要轻量类型安全 → pydantic-ai/smolagents;绑定某云 → 用那家官方(openai-agents/adk)**。

### ② 工具 / 协议——Agent 的"手"

**定位**:让 Agent 调外部能力,以及让 Agent 之间、Agent 与工具之间用**标准协议**对话,避免 NxM 的私有集成爆炸。底层原理见 [[15 Function Calling 工具调用|Function Calling 工具调用]]、[[16 工具设计与工具层|工具设计与工具层]];工具多了怎么按需加载见 [[18 工具检索与动态加载|工具检索与动态加载]]。

- **`modelcontextprotocol/python-sdk`**(`mcp`)——[[17 MCP 模型上下文协议|MCP 模型上下文协议]]官方 Python SDK,服务端/客户端两用。要把数据源/工具标准化暴露给任意模型应用时用它。
- **`PrefectHQ/fastmcp`**(`fastmcp`,**原 `jlowin/fastmcp`**,随 3.0 官方迁入 PrefectHQ,旧址 301 跳转,pip 名不变)——写 MCP server 的最快路径,2.0 起补齐 client/proxy/OpenAPI 集成,号称撑起全语言 70% 的 MCP server。日常造 MCP 工具就用它,别从官方 SDK 裸写。
- **`a2aproject/a2a-python`**(`a2a-sdk`)——[[30 A2A 协议|A2A 协议]]官方 Python SDK,Google 捐给 Linux Foundation,2026 实现规范 1.0(兼容 0.3)。让不同框架/不同公司的 Agent 跨服务互通时用它。注意:**MCP 是"Agent↔工具",A2A 是"Agent↔Agent"**,两者互补不冲突。
- **`ComposioHQ/composio`**(`composio`)——1000+ 托管工具集 + 托管 OAuth + 工具检索 + 沙箱。**最大价值是免你自写第三方 SaaS 的鉴权与集成**(Gmail/Slack/GitHub/Notion…),按 `composio-langchain`/`composio-crewai` 等适配各框架。

选型逻辑:**对外暴露工具/数据 → MCP(写 server 用 fastmcp);连一堆现成 SaaS、不想碰 OAuth → composio;让多个独立 Agent 互相调用 → A2A**。

### ③ 记忆——跨会话"记住"

**定位**:解决 [[19 Agent 记忆系统|Agent 记忆系统]]的核心痛点——上下文窗口装不下全部历史,需要把长期事实持久化、按需召回。与 [[20 上下文工程|上下文工程]]、[[21 上下文压缩与卸载|上下文压缩与卸载]]是一体两面(记忆是"卸载到外部"的极端形态)。

- **`mem0ai/mem0`**(`mem0ai`,注意 pip 名带 ai)——嵌入式记忆层,几行代码给任意 Agent 加"记住用户偏好"。最省事,适合大多数应用直接挂上。
- **`letta-ai/letta`**(`letta`,**前身 MemGPT**)——把"自编辑记忆"做成有状态的 Agent 服务,内建记忆分页(把记忆当虚拟内存换入换出,即 [[19 Agent 记忆系统|MemGPT 分页]])。要服务端有状态 Agent、持续自我改进时选它。
- **`getzep/zep`**(`zep`,底层 `getzep/graphiti`)——时序知识图谱记忆,自动建图并推理状态随时间的变化。关系密集、需要"谁在何时改了什么"这类时序查询时选它。

选型逻辑:**省事嵌入式 → mem0;要有状态服务/自改进 → letta;关系与时序重 → zep(graphiti)**。注意:记忆这一层**不是每个项目都要**,只有"需要跨会话记住用户"才上。

### ④ 可观测 / 评估——上线前的"眼睛"

**定位**:trace 出每一步在干嘛 + 给 Agent 表现打分。这是 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]的工程落地,**一旦要上线或排查"为什么错了",这层就是刚需**。

- **`langfuse/langfuse`**——开源可观测平台标杆,可自托管,trace/metrics/prompt 管理/dataset 全有,接 OpenTelemetry。要自托管、数据不出门时首选。
- **`Arize-ai/phoenix`**(`arize-phoenix`)——ML 级严谨度的可观测+评估,本地可跑。团队有 ML 评估经验、要更工程化指标时选它。
- **`confident-ai/deepeval`**(`deepeval`)——"LLM 版 pytest",2026 的 4.0 主打 agent-native 评估流。要把评估写成单测、卡进 CI 时选它。
- **`explodinggradients/ragas`**(`ragas`)——专攻 [[36 Agentic RAG|RAG]] 评估的指标库(忠实度、召回相关性等)。做检索增强、要量化检索质量时选它。
- 还有:**`AgentOps-AI/agentops`**(`agentops`,agent 监控+成本追踪,原生接 crewAI/openai-agents 等)、**`traceloop/openllmetry`**(基于 OpenTelemetry 的 GenAI 可观测扩展)、**LangSmith**(LangChain 官方**托管** SaaS,框架集成最深,但自托管仅 Enterprise)。

选型逻辑:**要自托管 trace → langfuse;评估写成单测 → deepeval;RAG 专项打分 → ragas;ML 级严谨 → phoenix;吃 LangChain 全家桶且能付费托管 → LangSmith**。

### ⑤ 提示优化——把提示当"参数"调

**定位**:不靠人手调 prompt,而是用程序自动搜索/进化出更好的提示。这是 [[31 Agent 提示词优化(DSPy)|Agent 提示词优化(DSPy)]]的工具落地。

- **`stanfordnlp/dspy`**(`dspy`)——"编程而非提示":你写 signature 和模块,优化器(MIPROv2、2026 的 **GEPA**)自动生成指令与 few-shot。GEPA 用反思式进化,论文称在多任务上比 MIPROv2 高 13%、比 GRPO 高 20% 且少 35x rollout(ICLR 2026)。要系统化、可复现地优化 pipeline 时选它。
- **`zou-group/textgrad`**(`textgrad`)——"文本梯度反传":用 LLM 给出的文本反馈当梯度,反向优化提示/解。发表于 Nature。要对单点提示/解做精细优化时选它。

选型逻辑:**优化整条 Agent pipeline → dspy(先 MIPROv2,够狠上 GEPA);优化单个提示/答案 → textgrad**。注意:**这层是"提升手段"不是"运行时依赖"**——调好了把产物塞回 ① 即可,生产时不需要带着优化器跑。

### ⑥ 训练 / RL——改权重,不止改提示

**定位**:当提示优化触顶仍不够时,直接对模型做 RL 后训练,让 Agent 从经验里学。原理与奖励范式见 [[32 Agentic RL 与训练|Agentic RL 与训练]]、[[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进(Ralph loop)]]。

- **`verl-project/verl`**(`verl`,**原 `volcengine/verl`**,HybridFlow 架构)——灵活高效的 RL 后训练框架,2026 是工业界跑大规模 agentic RL 的主力之一。
- **`huggingface/trl`**(`trl`)——HF 官方,DPO/GRPO/PPO 全套 trainer,生态最熟、上手最低。入门 RLHF/RLAIF 首选。
- **`OpenRLHF/OpenRLHF`**(基于 Ray)——2026 已转型为 **Agentic RL 框架**,支持 PPO/DAPO/REINFORCE++/异步 RL/VLM。要可扩展分布式 agentic RL 时选它。
- **`OpenPipe/ART`**(Agent Reinforcement Trainer)——专为"多步 Agent"用 GRPO 做在岗训练,亮点是 **RULER** 自动生成奖励(免手写 reward)。要给 agent 做 RL 但懒得设计奖励函数时选它。

选型逻辑:**入门/通用 → trl;工业大规模 → verl;分布式 agentic → OpenRLHF;给 agent 在岗 RL 且省奖励工程 → ART**。提醒:**这是整张图最贵、最该最后考虑的一层**——先穷尽 ① 编排、上下文工程、⑤ 提示优化,真不够再动权重。

### ⑦ 垂直成品 Agent——别自建,先套现成的

**定位**:已经把 ①~④ 攒成可用产品的"成品 Agent",编码与浏览器两大方向最成熟。原理见 [[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]]、[[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]]。

- **`All-Hands-AI/OpenHands`**(**org 已迁 `OpenHands/OpenHands`**,旧址跳转)——开源 AI 软件工程师平台,2026 走 1.x + software-agent-sdk。要一个能开 PR、跑命令、改代码的成品时选它。
- **`princeton-nlp/SWE-agent`**(**org 已迁 `SWE-agent/SWE-agent`**)——拿 GitHub issue 自动修复,开源里 [[28 代码 Agent 与 SWE-bench|SWE-bench]] 的 SOTA 常客;mini-swe-agent 用 100 行 Python 拿到 65% verified。研究/刷榜导向时选它。
- **`Aider-AI/aider`**(`aider-chat`)——终端里的 AI 结对编程,直接在你本地 git 仓改文件。要"命令行里和模型一起写代码"时选它。
- **`browser-use/browser-use`**(`browser-use`)——让 Agent 控制浏览器自动化网页任务,2026 近 10 万 star。要操作真实网页(下单、填表、抓取)时选它。

选型逻辑:**写代码/修 bug → OpenHands、aider;刷 SWE-bench/做研究 → SWE-agent;操作网页 → browser-use**。铁律:**自建编排前先问"有没有现成成品能套"**,成品省 80% 工作,代价是定制性下降。

### ⑧ 部署 / 成本——让它"死了能续、贵了能省"

**定位**:长程任务的容错、状态持久化,以及跨模型的成本/延迟控制。对应 [[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]、[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。

- **`temporalio/sdk-python`**(`temporalio`)——持久执行引擎:进程中途挂了,在新进程精确重建状态、从断点续跑,变量与 API 调用全程可追溯。跑几小时不能中途死的长程 Agent 用它。
- **`BerriAI/litellm`**(`litellm`)——统一网关,一套 OpenAI 格式调 100+ 模型,带成本追踪、护栏、负载均衡、日志。多模型/省钱/做 routing 时用它。
- **`langgraph` checkpoint**——若已用 langgraph,它自带状态快照即可做轻量中断恢复,简单场景无需上 temporal。

选型逻辑:**进程级容错/超长程 → temporal;轻量中断恢复 → langgraph checkpoint;多模型统一接口与省钱 → litellm**。

## 选型决策建议(把上面压成一条路径)

1. **从 ① 编排起步**,按可控性/协作/轻量三选一,先跑通最小闭环(见 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]]:很多需求其实是 [[05 Routing|Routing]]/[[04 Prompt Chaining|Prompt Chaining]] 这类固定 workflow,根本不需要全自主 Agent;人工介入点见 Human-in-the-Loop)。
2. **要连工具就上 ②**,优先 MCP(用 fastmcp 写)和 composio(免 OAuth),别手撸集成。
3. **以下都"等痛点出现再加"**:需要跨会话记忆 → ③;要上线/排错 → ④;提示触顶 → ⑤(再不行才 ⑥);超长程不能死 → ⑧。
4. **能套成品就套**:编码/浏览器场景直接用 ⑦,别重造。
5. **框架是负债**:每加一层就多一份升级成本和锁定风险,层数越少越好维护。

## 坑(只挑反直觉的)

- **别一上来全栈**。新手最常见的错是"langgraph + mem0 + langfuse + dspy + temporal"一把梭,结果连最小闭环都没跑通。**90% 的项目只需 ①+②**,上线再加 ④,长程再加 ⑧。
- **org 改名是常态,认 pip 名别只认 URL**。autogen→`ag2`、phidata→`agno`(pip 名也从 phidata 换成 agno)、`jlowin/fastmcp`→`PrefectHQ/fastmcp`、`princeton-nlp/SWE-agent`→`SWE-agent/SWE-agent`、`volcengine/verl`→`verl-project/verl`、`All-Hands-AI/OpenHands`→`OpenHands/OpenHands`。GitHub URL 会 301,但 **pip 包名通常稳定**——以 pip 名为准。
- **同层选一个就够,别叠**。记忆层 mem0/letta/zep 是竞品不是搭档;可观测 langfuse/phoenix/LangSmith 同理。叠装只会让 trace 重复、依赖打架。
- **⑤/⑥ 不是运行时依赖**。提示优化和训练是"离线提升手段",产物(优化后的提示/权重)塞回 ① 就行,生产服务里不该带着 dspy 优化器或 trl trainer 跑。
- **MCP ≠ A2A,别混**。MCP 解决"Agent↔工具/数据"(② 的 NxM 集成),A2A 解决"Agent↔Agent"互通;一个对下,一个对旁。
- **"成品 Agent"有定制天花板**。OpenHands/aider 省事,但深度定制其内部循环会很别扭;需求一旦偏离主线,可能反而该回 ① 自建。
- **可观测要趁早接,别等出事**。trace 是事后排查 [[22 多智能体系统|多智能体系统]]为什么跑飞的唯一抓手,上线前没接好,出问题就只能瞎猜。
- **注意 [[05 Prompt Injection 提示注入|Prompt Injection]] 与 [[01 AI 安全总览与三层栈|AI 安全]]**:工具层(②)和成品 Agent(⑦)一旦能跑命令/访问 SaaS,被注入的攻击面就打开了,[[24 沙箱、最小权限与人审闸门|沙箱与权限边界]]要先于功能设计。

## 关键事实

- **生态稳定的是"层",流动的是"库"**:记住每层的职责和 2026 主流三五个库,比记版本号有用得多。
- **层数 = 复杂度 = 负债**:从 ① 起步,按"痛点出现"逐层加,是唯一可持续的扩张方式。
- **同层互斥、跨层互补**:这是读全景图的核心规则,也是判断"该不该再加一个库"的标尺。
- **⑤⑥是提升手段、⑦是成品、其余是运行时积木**:三类东西心智模型不同,别用同一把尺子衡量。
- **以 pip 名为锚**:`langgraph`/`crewai`/`ag2`/`smolagents`/`pydantic-ai`/`agno`/`openai-agents`/`google-adk`/`llama-index` · `mcp`/`fastmcp`/`a2a-sdk`/`composio` · `mem0ai`/`letta`/`zep` · `langfuse`/`arize-phoenix`/`deepeval`/`ragas`/`agentops` · `dspy`/`textgrad` · `verl`/`trl` · `aider-chat`/`browser-use` · `temporalio`/`litellm`。URL 会变,这些名字大体不变。

相关:[[37 Agent 框架对比|Agent 框架对比]](① 的代码级对照)、[[23 Agent Harness 概览|Agent Harness 概览]](Anthropic 视角的 harness 分层)、[[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]]、[[29 Deep Research Agent|Deep Research Agent]](一个横跨多层的完整案例)。

## 工业界实践

**真实项目长什么样:不是「装满八层」,而是「按痛点逐层加」。** 90% 的生产 agent 横跨 **①+②** 起步(编排 + 工具),上线再补 **④**(可观测),长程再补 **⑧**(持久化)。下面是几个有代表性的真实组合:

```
企业知识库问答:    langgraph(①) + llama-index(检索) + mcp/composio(②)
                   + ragas+langfuse(④)            ← 不需要 ③⑤⑥⑦⑧
研究/内容流水线:   crewai 原型(①) → 生产重写 langgraph(①) + langfuse(④)
长程编码 agent:    套成品 OpenHands/aider(⑦),需要时 temporal(⑧)兜容错
带记忆的助手:      pydantic-ai(①) + mem0(③) + composio(②) + langfuse(④)
多模型省钱网关:    任意①  +  litellm(⑧) 统一接口 + 成本追踪 + routing
```

**各层的 2026 生产「默认值」(选一个就够,同层互斥):**
- **① 编排**:可控/生产派 **langgraph**(已 1.x,配 LangGraph Platform 做托管持久化);快搭协作 **crewai**;极简少黑盒 **smolagents**(CodeAct);类型安全 **pydantic-ai**;绑云用官方(**openai-agents** / **google-adk**)。
- **② 工具/协议**:对外暴露工具写 MCP server 用 **fastmcp**(已迁 PrefectHQ,撑起全语言约 70% 的 MCP server);连一堆 SaaS 免 OAuth 用 **composio**;Agent↔Agent 互通用 **a2a-sdk**(已进 Linux Foundation,规范 1.0)。**记牢:MCP 对下(Agent↔工具),A2A 对旁(Agent↔Agent)**。
- **③ 记忆**:省事嵌入式 **mem0**;有状态服务/自改进 **letta**(前身 MemGPT);时序图谱 **zep**(底层 graphiti)。**不是每个项目都要**。
- **④ 可观测**:自托管 **langfuse**(2026.1 被 ClickHouse 收购,能力延续);CI 卡门禁 **deepeval**;RAG 专项 **ragas**;ML 级严谨 **phoenix**;托管最丝滑 **LangSmith**;eval-first 实验 **Braintrust**。
- **⑤ 提示优化**:整条 pipeline 用 **dspy**(优化器 MIPROv2,够狠上 **GEPA**,ICLR 2026 称比 MIPROv2 高 13%);单点提示用 **textgrad**(发表于 Nature)。
- **⑥ 训练/RL**:入门 **trl**;工业大规模 **verl**(原 volcengine);分布式 agentic **OpenRLHF**;省奖励工程 **ART**(RULER 自动生成奖励)。
- **⑦ 成品 agent**:编码 **OpenHands** / **aider** / **SWE-agent**(mini 版 100 行拿 SWE-bench 65% verified);浏览器 **browser-use**(近 10 万 star)。
- **⑧ 部署/成本**:进程级容错 **temporal**;轻量中断恢复用 langgraph checkpoint;多模型统一接口 + 省钱 **litellm**。

**规模化与成本的硬约束:**
- **层数 = 复杂度 = 负债**:每加一层多一份升级成本和锁定风险。verl/trl(⑥)是整张图最贵、最该最后碰的——先穷尽 ① 编排、上下文工程、⑤ 提示优化,真不够再动权重。
- **⑤/⑥ 不是运行时依赖**:dspy/textgrad/trl 是**离线提升手段**,优化好的提示/权重塞回 ① 即可,生产服务里别带着优化器/trainer 跑。
- **多模型成本控制走 litellm 网关**:一套 OpenAI 格式调 100+ 模型 + 成本追踪 + routing + 负载均衡,是省钱(见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]])和切换模型的统一入口。

**安全是横切关注点,不是某一层。** 工具层(②)和成品 agent(⑦)一旦能跑命令/访问 SaaS,[[05 Prompt Injection 提示注入|Prompt Injection]] 的攻击面就打开了——[[24 沙箱、最小权限与人审闸门|沙箱、最小权限与人审闸门]] 要**先于功能**设计。

## 面试高频

**Q1:面对几百个 agent 库怎么不被淹?给个方法论。**
**记「层」不记「库」**:生态稳定的是按职责分的八层(编排/工具协议/记忆/可观测评估/提示优化/训练/成品/部署),流动的是各层的具体库。记住每层职责 + 2026 主流三五个库,比记版本号有用得多。读图三规则:**颜色只分层不分优劣;同层多库≈互斥竞品(选一个);跨层多库≈互补搭配(可叠)**。
- *追问:org 老改名怎么办?* **认 pip 名别只认 URL**:autogen→`ag2`、phidata→`agno`、`jlowin/fastmcp`→`PrefectHQ/fastmcp`、`volcengine/verl`→`verl-project/verl`,GitHub URL 会 301,pip 包名通常稳定。

**Q2:MCP 和 A2A 有什么区别?会不会冲突?**
不冲突,**互补**:**MCP(Model Context Protocol)解决 Agent↔工具/数据**的标准化接入(避免 NxM 私有集成爆炸),对下;**A2A(Agent-to-Agent)解决 Agent↔Agent** 跨框架/跨公司互通,对旁。一个连工具,一个连同侪。
- *陷阱:「都是协议,选一个就行?」* 错,职责正交,大型系统可能两者都用:用 MCP 给每个 agent 装工具,用 A2A 让这些 agent 互相调用。

**Q3:一个新项目该从哪层起步?什么时候才加更多层?**
从 **① 编排**起步,按可控/协作/轻量三选一跑通最小闭环;要连工具就上 **②**(优先 MCP + composio,别手撸集成);**其余都等痛点出现再加**:跨会话记忆→③、上线/排错→④、提示触顶→⑤(再不行才 ⑥)、超长程不能死→⑧;编码/浏览器场景**先问有没有现成成品**(⑦)能套,省 80% 工作。
- *陷阱:新手最常见的错* 一上来「langgraph + mem0 + langfuse + dspy + temporal」一把梭,连最小闭环都没跑通。**框架是负债,层数越少越好维护。**

**Q4:⑤提示优化、⑥训练、⑦成品 这三类和其余层有什么本质不同?**
⑤⑥是**离线提升手段**(产物塞回 ① 用,生产不带着跑)、⑦是**成品**(直接套用,定制天花板低)、其余是**运行时积木**(真正在服务里跑的依赖)。三类心智模型不同,别用同一把尺子衡量。

## 知识拓展

**生态的两个底层趋势:**
- **协议化收敛**:框架内部「怎么写」在分化(图/角色/对话/CodeAct/类型安全),但框架**之间**靠 MCP + A2A 标准化互通——长期看「锁定在某框架」的风险被协议层稀释,可互操作性比单框架优劣更重要。
- **评估/训练在 agent 化**:④ 层出现 Agent-as-a-Judge(用 agent 评 agent 轨迹)、⑥ 层 ART 用 RULER 自动生成奖励——「评」和「练」本身也在变成 agent 任务,与 [[38 Agent 评估与可观测性|评估]]、[[32 Agentic RL 与训练|Agentic RL]] 接壤。

**反模式(只挑反直觉的):**
- **同层叠装**:记忆层 mem0/letta/zep 是竞品不是搭档,可观测 langfuse/phoenix/LangSmith 同理;叠装只会让 trace 重复、依赖打架。
- **成品 agent 深度定制其内部循环**:OpenHands/aider 省事,但需求一旦偏离主线,改它内部循环很别扭,可能反而该回 ① 自建。
- **为「全栈」而全栈**:八层不是成就清单;真实项目通常只横跨 2~4 层。

**边界:** 这张全景是「OSS 视角」;闭源/托管侧另有平行栈(LangSmith、Braintrust、各云 agent 平台、Claude Agent SDK 的 HaaS 等),选型时 OSS 自托管的「数据不出门 + 无锁定」要和托管的「省心 + 能力更全」做权衡——同一权衡也出现在 [[37 Agent 框架对比|Agent 框架对比]] 与 [[23 Agent Harness 概览|Agent Harness 概览]]。

**相关链接:** ① 的代码级对照见 [[37 Agent 框架对比|Agent 框架对比]];一个横跨多层的完整案例见 [[29 Deep Research Agent|Deep Research Agent]];harness 分层视角见 [[23 Agent Harness 概览|Agent Harness 概览]];安全横切见 [[05 Prompt Injection 提示注入|Prompt Injection 提示注入]] 与 [[24 沙箱、最小权限与人审闸门|沙箱、最小权限与人审闸门]]。
