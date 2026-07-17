[[18 Agent 间通信安全与级联失败|本篇]]把 OWASP ASI 的两条相邻条目焊在一起看:**ASI07 不安全 agent 间通信**是「点」上的病灶,**ASI08 级联失败**是这个病灶沿网络扩散的「面」。在单 LLM 时代,一次失陷止于一个进程;在[[22 多智能体系统|多智能体系统]]里,agent 之间靠消息互相驱动,一个节点被攻陷会像多米诺一样把错误**扇出(fan-out)**给所有信任它的 peer,比人类介入更快。

## ASI07:不安全的 agent 间通信
Agent 之间交换的不只是文本,还有**目标、部分计划、工具结果、信誉信号**,载体可能是消息总线、HTTP、gRPC、[[17 MCP 模型上下文协议|MCP]]、[[30 A2A 协议|A2A]] 或共享内存。一旦认证、完整性或语义一致性薄弱,攻击者就能:
- **伪造 agent**:冒充某个 peer 发指令(身份不可信)。
- **篡改消息**:中途改掉计划或工具结果(完整性不可信)。
- **重放委派令牌**:把一次合法的授权重复利用(无新鲜性校验)。
- **协议降级**:逼通信退回到无加密/弱认证的旧通道。
- **毒化路由**:污染信誉信号,让消息被导向恶意节点。

本质是:很多多 agent 框架默认**peer 之间无条件互信**——这正是它要被攻破的地方,与[[16 Agent 身份与权限滥用(非人类身份 NHI)|非人类身份]]问题同源。

## ASI08:级联失败
一个 agent 的错误或失陷不会停在原地,而是**沿自动化管道传播并逐级放大**:下游 agent 信任上游 → 接受毒消息 → 据假信号触发真实动作 → 再把错误转发给更多 peer。扇出是指数级的:一变三、三变九,跨团队、跨租户外溢,**比人能反应更快**。这也是为什么级联失败常常顺手演变成[[12 Unbounded Consumption 成本型 DoS|成本型 DoS]]——失控的 agent 互相触发,把 API 调用与算力预算烧穿(链 #12 成本放大)。

**指数扇出手算**。设每个 agent 把毒消息转给的下游数(扇出度)$d=3$,信任链传播 $L=5$ 跳。第 $k$ 跳被波及的 agent 数是 $d^k$:第 1 跳 $3^1=3$,第 2 跳 $3^2=9$,…第 5 跳 $3^5=243$。即一个源头节点 5 跳内就拉爆 $243$ 个末端 agent;累计被触达 $\sum_{k=0}^{5}3^k=\tfrac{3^{6}-1}{3-1}=364$ 个节点。每个节点的"真实动作"都是一次 API 调用+算力开销,这 364 次几乎在同一波传播里同时发生——这正是它"快于人类介入"、且顺手烧穿预算变成成本型 DoS 的算术根源。把扇出度从 3 压到 $d=1$(fan-out cap),末端就从 $3^5=243$ 塌缩到 $1^5=1$,指数项被掐死成线性——这就是 fan-out 上限为何是核心防御。

![[sec-级联失败.png]]

## 对比表

| 维度 | ASI07 不安全 agent 间通信 | ASI08 级联失败 |
| --- | --- | --- |
| 性质 | 单条链路的「点」病灶 | 沿网络扩散的「面」效应 |
| 触发 | 消息被伪造/篡改/重放 | 一个节点失陷或出错 |
| 关键词 | 认证、完整性、新鲜性 | fan-out、放大、传播速度 |
| 后果 | 单 agent 被误导 | 跨团队/跨租户系统性崩塌 |
| 主防御 | mTLS、消息签名、零信任 | 熔断、fan-out 上限、隔离舱 |

## 防御
通信边界与扩散路径要**同时**设闸,否则只堵一头无效:
- **治 ASI07(通信边界)**:agent 间用 **mTLS** 双向认证;**消息签名**保完整性;**零信任**——任何 peer 消息默认不可信、按策略校验;用 nonce / 时间戳**防重放**;禁止协议降级。
- **治 ASI08(扩散路径)**:**熔断器(circuit breaker)**——检测到异常立即断开、停止向下游外发;**fan-out 上限**——限制单个 agent 能扇出的消息数,配速率与预算配额;**隔离舱(bulkhead)**与跨租户硬隔离,防止外溢;**幂等键去重**避免重复触发;**超时退避 + 健康检查 + 自动回滚**;**人审闸门**对不可逆动作兜底(链 [[19 人机信任利用与 Rogue Agents|人审]]、[[24 沙箱、最小权限与人审闸门|人审闸门]])。
- **可观测兜底**:全量消息日志 + 异常基线告警,失陷可追溯、可回放(链 [[25 监控、可观测与事件响应|监控]])。

核心一句话:**任一 agent 都不得无条件信任 peer;失陷必须被「断」住,而不是被「转发」。**

## 关键事实(含出处)
- ASI07/ASI08 准确名称为 **Insecure Inter-Agent Communication / Cascading Failures**,出自 **OWASP Top 10 for Agentic Applications(2025-12-09 发布)**。来源:[OWASP Gen AI Security Project](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/)、[OWASP ASI 初始](https://owasp.org/www-project-top-10-for-large-language-model-applications/initiatives/agent_security_initiative/)。
- ASI07 官方描述明确点名「spoof agents、replay delegation tokens、downgrade protocols、poison routing」,通信载体含 MCP / A2A。
- ASI08 强调「one fault propagates across agents and automations, amplifying … faster than humans can intervene」,即级联放大快于人类介入。

## 工业界实践
工业界把这块当「分布式系统 + 零信任 + 弹性工程」的交叉:通信边界用 service mesh 那套硬认证,扩散路径用微服务弹性模式(circuit breaker / bulkhead)那套兜底。

**1. 治通信边界(ASI07):零信任 + 可验证身份**
- **mTLS 双向认证**靠 [[16 Agent 身份与权限滥用(非人类身份 NHI)|SPIFFE/SPIRE]] 落地:每个 agent 拿 SVID,互调时双向验证书,身份不可伪造。生产里常跑在 service mesh(Istio / Linkerd)上,mTLS 由 sidecar 自动接管。
- **消息签名 + 新鲜性**:消息体 + 委派令牌签名保完整性;带 **nonce + 时间戳**防重放,服务端维护已用 nonce 窗口拒重复。
- **A2A 协议的现实坑**:Google 2025-04 发布的 **Agent2Agent(A2A)**(现归 Linux Foundation)支持 OAuth2 / API Key / mTLS,但**用 Agent Card 声明认证方式却不强制校验卡本身的真伪**——agent 冒充、卡篡改、重放是真实风险,需实现方额外加固(保护 Agent Card 端点、out-of-band 取凭证、永不把 secret 塞进 JSON-RPC payload)。这就是「协议给了能力,默认互信仍要你自己关掉」的典型。

```python
# agent 收到 peer 消息:零信任校验四件套(身份/完整性/新鲜性/授权)
def on_peer_message(msg):
    verify_mtls_peer_cert(msg.conn)                 # ① 身份:SVID 双向认证
    if not verify_sig(msg.body, msg.sig, peer_pubkey(msg.from)):
        drop("integrity_fail")                       # ② 完整性:消息签名
    if msg.nonce in seen_nonces or expired(msg.ts):
        drop("replay_or_stale")                      # ③ 新鲜性:nonce+时间戳防重放
    if not policy.allow(msg.from, msg.intent):
        drop("unauthorized")                         # ④ 授权:按策略校验,默认不信
    handle(msg)
```

**2. 治扩散路径(ASI08):弹性 + 隔离**
把微服务的弹性模式搬到 agent 网络:
- **熔断器(circuit breaker)**:检测到下游异常/失陷,立即「断」而非「转发」,停止外发。
- **fan-out 上限 + 配额**:限制单 agent 单位时间能扇出的消息/触发数,配速率限制与预算配额——这同时挡住 [[12 Unbounded Consumption 成本型 DoS|成本型 DoS]](agent 互相触发烧穿预算)。
- **隔离舱(bulkhead)+ 跨租户硬隔离**:一个舱崩了不拖垮全局,防跨团队/跨租户外溢。
- **幂等键去重**:防重复触发同一不可逆动作。
- **超时退避 + 健康检查 + 自动回滚**;不可逆动作前置**人审闸门**兜底。

```python
# 扇出闸:熔断 + 上限 + 隔离,核心是「失陷必须被断住,不是被转发」
breaker = CircuitBreaker(fail_threshold=5, reset_timeout=30)
def fan_out(agent, downstream, msg):
    if agent.fanout_count(window="1m") >= FANOUT_CAP:
        alert("fanout_cap_hit"); return             # 限制扇出,防级联+成本DoS
    if breaker.is_open(downstream):
        return                                       # 下游异常已熔断,直接不发
    if msg.idem_key in already_sent: return          # 幂等去重,防重复触发
    try:
        send(downstream, msg, timeout=5)
    except Exception:
        breaker.record_failure(downstream)           # 攒够失败 → 熔断该路径
```

**3. 可观测兜底**
全量消息日志(含 from/to/intent/sig 校验结果)+ 异常基线告警,失陷可追溯、可回放(链 [[25 监控、可观测与事件响应|监控]]);给消息打 trace_id 串起跨 agent 调用链,定位级联起点。

## 面试高频

**Q1:单 agent 失陷和多 agent 级联失败,危害量级差在哪?**
标准答:单 LLM 时代一次失陷止于一个进程;多 agent 里 agent 靠消息互相驱动,一个节点被攻陷会沿信任链**扇出(fan-out)指数放大**——一变三、三变九,跨团队跨租户外溢,且**比人类介入更快**。ASI07 是「点」病灶(单条链路被伪造/篡改/重放),ASI08 是「面」效应(沿网络扩散放大)。
- 追问「为什么不能只防一头」:只堵通信边界(ASI07)挡不住已失陷节点的合法扩散;只堵扩散(ASI08)挡不住源头被伪造。必须同时设闸。

**Q2:ASI07 有哪些攻击手法?对应防御?**
标准答:伪造 agent(→mTLS 身份认证)、篡改消息(→消息签名)、重放委派令牌(→nonce+时间戳)、协议降级(→禁止降级、强制 TLS≥1.2)、毒化路由信誉信号(→信誉信号也要签名+校验)。根本前提:**多 agent 框架默认 peer 间无条件互信,这正是要被关掉的默认。**

**Q3:circuit breaker / bulkhead / fan-out cap 分别防什么?**
标准答:熔断器——检测异常立即断开停止外发,防级联沿单条路径蔓延;隔离舱——资源/故障分舱,一舱崩不拖垮全局,防横向外溢;fan-out 上限——限制单 agent 扇出量,既防级联放大又防成本型 DoS。一句话记忆:**失陷必须被「断」住,而不是被「转发」。**
- 陷阱:有人把这三者混为一谈——熔断是「时间维度」(失败累积就断路径),隔离舱是「空间维度」(资源分区互不影响),fan-out cap 是「数量维度」(限扇出度数)。

**Q4(追问)为什么级联失败常演变成成本型 DoS?**
标准答:失控 agent 互相触发,每次触发都是真实 API 调用 + 算力消耗,指数扇出会把 token/算力预算烧穿(链 #12)。所以 fan-out cap + 预算配额是「一石二鸟」的防御。

## 知识拓展
- **框架定位**:OWASP **ASI07 Insecure Inter-Agent Communication / ASI08 Cascading Failures**(Top 10 for Agentic Applications,2025-12-09);MITRE ATLAS 2025-10 新增的 agent 技术含跨 agent 传播/协同。
- **协议**:**A2A(Agent2Agent)**(Google 2025-04 发布,现 Linux Foundation 治理,Apache-2.0)、[[17 MCP 模型上下文协议|MCP]];A2A 安全研究见 arXiv:2504.16902《Building A Secure Agentic AI Application Leveraging A2A》。
- **复用的成熟工程**:零信任(NIST SP 800-207)、service mesh mTLS(Istio/Linkerd + SPIFFE)、弹性模式(circuit breaker / bulkhead,源自 Hystrix / Polly / Resilience4j)。
- **对抗演化**:攻击从「伪造单条消息」→「劫持信任链做级联」→「跨租户外溢」→「协同串谋(见 [[19 人机信任利用与 Rogue Agents|流氓 agent]] 的 collusion)」;防御从「加 mTLS」→「零信任默认不信 peer」→「弹性兜底让失陷可被断住」→「跨 agent 可观测+可回放定位级联起点」。

## 相邻
- 上游:[[16 Agent 身份与权限滥用(非人类身份 NHI)|身份滥用]]、[[17 Memory 与 Context Poisoning|记忆投毒]]
- 下游:[[19 人机信任利用与 Rogue Agents|人机信任与流氓 agent]]、[[20 Agentic 供应链与 MCP 安全|MCP 安全]]
- 跨域:[[22 多智能体系统|多智能体系统]]、[[26 Sub-agents 与 Agent Teams|Sub-agents]]
- 框架定位:[[26 安全框架与治理地图|安全框架地图]]、[[01 AI 安全总览与三层栈|三层栈]]
