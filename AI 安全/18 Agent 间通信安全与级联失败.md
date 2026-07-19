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
一个 agent 的错误或失陷不会必然停在原地,而可能沿自动化管道传播:下游过度信任上游 → 接受毒消息 → 据假信号触发真实动作 → 再把错误转发给更多 peer。只有在持续分叉、无去重/熔断、各边独立成功等严格前提下,节点数才会按指数增长;实际图还受拓扑、授权、队列、重试与人工闸门约束。即使不是指数,未受控的重试和扇出也会演变成[[12 Unbounded Consumption 成本型 DoS|成本型 DoS]]——失控的 agent 互相触发,把 API 调用与算力预算烧穿(链 #12 成本放大)。

**指数扇出玩具上界手算。** 假设这是无环树、每个节点恰好把同一毒消息发给 $d=3$ 个**彼此不同**的下游、所有下游都接受且没有去重/熔断/授权拒绝,传播 $L=5$ 跳。第 $k$ 跳的节点数才是 $d^k$:第 1 跳 $3$,第 2 跳 $9$,第 5 跳 $3^5=243$;累计为
$$\sum_{k=0}^{5}3^k=\frac{3^{6}-1}{3-1}=364.$$
这只是用于设计 fan-out cap、队列预算与熔断阈值的**最坏拓扑模型**,不是对所有多 agent 系统的经验定律。把允许扇出从 3 限到 1,在这个模型里末端上界从 $243$ 降到 $1$;真实系统还必须以每跳授权、幂等键、速率/成本配额和终态监测来验证是否真的止血。

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
- **可观测兜底**:按风险分级记录最小可审计消息元数据(认证主体、授权决定、消息/载荷哈希、trace_id、状态、时间与拒绝原因),并把原文载荷与凭证隔离、脱敏、限时保留和严格授权访问;再用异常基线告警定位级联起点(链 [[25 监控、可观测与事件响应|监控]])。

核心一句话:**任一 agent 都不得无条件信任 peer;失陷必须被「断」住,而不是被「转发」。**

## 关键事实(含出处)
- ASI07/ASI08 准确名称为 **Insecure Inter-Agent Communication / Cascading Failures**,出自 **OWASP Top 10 for Agentic Applications(2025-12-09 发布)**。来源:[OWASP Gen AI Security Project](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/)、[OWASP ASI 初始](https://owasp.org/www-project-top-10-for-large-language-model-applications/initiatives/agent_security_initiative/)。
- ASI07 官方描述明确点名「spoof agents、replay delegation tokens、downgrade protocols、poison routing」,通信载体含 MCP / A2A。
- ASI08 强调「one fault propagates across agents and automations, amplifying … faster than humans can intervene」,即级联放大快于人类介入。
- **A2A 最新正式规范为 1.0.0**。Agent Card 可选择用 JWS 签名;对签名卡,客户端应从受信任密钥库或安全渠道取得公钥、做 RFC 8785 规范化后验签,至少验证一枚可信签名再信任卡,且拒绝过期/撤销密钥。签名不替代每次请求的 TLS 与授权范围检查。来源:[A2A Protocol Specification 1.0.0, §§7–8](https://a2a-protocol.org/dev/specification/)(核验:2026-07-18)。

## 工业界实践
工业界把这块当「分布式系统 + 零信任 + 弹性工程」的交叉:通信边界用 service mesh 那套硬认证,扩散路径用微服务弹性模式(circuit breaker / bulkhead)那套兜底。

**1. 治通信边界(ASI07):零信任 + 可验证身份**
- **mTLS 双向认证**靠 [[16 Agent 身份与权限滥用(非人类身份 NHI)|SPIFFE/SPIRE]] 落地:每个 agent 拿 SVID,互调时双向验证书,身份不可伪造。生产里常跑在 service mesh(Istio / Linkerd)上,mTLS 由 sidecar 自动接管。
- **消息签名 + 新鲜性**:消息体 + 委派令牌签名保完整性;带 **nonce + 时间戳**防重放,服务端维护已用 nonce 窗口拒重复。
- **A2A 的现实边界**:截至 2026-07 的 **A2A 1.0.0** 支持把 Agent Card 作为可选 JWS 签名对象,并规定生产传输使用加密通道;客户端若以该卡建立信任,应验证至少一枚来自可信密钥/安全渠道的签名、拒绝撤销或过期 key。未签名卡、错误的 `jku` 信任链、以及把「卡声明的认证方案」当成运行时授权事实,仍会造成冒充或降级风险。保护 Card 端点、带外配置受信任密钥、对每次请求独立做 TLS + 授权检查,且绝不把 secret 塞进 JSON-RPC payload。

```python
# agent 收到 peer 消息:零信任校验四件套(身份/完整性/新鲜性/授权)
def on_peer_message(msg):
    peer = verified_mtls_subject(msg.conn)           # ① 身份:SVID 双向认证
    if msg.sender != peer or not verify_sig(msg.body, msg.sig, trusted_key(peer)):
        return drop("integrity_fail")                # ② 不从未验证 msg.from 选公钥
    if msg.nonce in seen_nonces or expired(msg.ts):
        return drop("replay_or_stale")               # ③ 新鲜性:nonce+时间戳防重放
    if not policy.allow(peer, msg.intent):
        return drop("unauthorized")                  # ④ 授权:按策略校验,默认不信
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
记录最小且分级的 action trace(认证主体、from/to、intent、签名/授权校验结果、载荷哈希、trace_id、状态),用异常基线告警定位级联起点。原始载荷、凭证与可能含隐私的数据应脱敏、隔离存放、限时保留并受访问控制;可回放的是经过治理的证据,不是无差别地复制所有消息正文(链 [[25 监控、可观测与事件响应|监控]])。

## 面试高频

**Q1:单 agent 失陷和多 agent 级联失败,危害量级差在哪?**
标准答:单 LLM 时代一次失陷通常止于较小边界;多 agent 里消息能继续驱动下游,一个节点可沿信任链扩散到跨团队/跨租户。若拓扑持续分叉且没有去重、授权拒绝或熔断,才会出现 $d^k$ 级玩具上界;面试中应说明这是前提很强的最坏模型,不能把「一变三、三变九」当普适实测。ASI07 是「点」病灶(单条链路被伪造/篡改/重放),ASI08 是「面」效应(沿网络扩散放大)。
- 追问「为什么不能只防一头」:只堵通信边界(ASI07)挡不住已失陷节点的合法扩散;只堵扩散(ASI08)挡不住源头被伪造。必须同时设闸。

**Q2:ASI07 有哪些攻击手法?对应防御?**
标准答:伪造 agent(→mTLS 身份认证)、篡改消息(→消息签名)、重放委派令牌(→nonce+时间戳)、协议降级(→禁止降级、强制 TLS≥1.2)、毒化路由信誉信号(→信誉信号也要签名+校验)。根本前提:**多 agent 框架默认 peer 间无条件互信,这正是要被关掉的默认。**

**Q3:circuit breaker / bulkhead / fan-out cap 分别防什么?**
标准答:熔断器——检测异常立即断开停止外发,防级联沿单条路径蔓延;隔离舱——资源/故障分舱,一舱崩不拖垮全局,防横向外溢;fan-out 上限——限制单 agent 扇出量,既防级联放大又防成本型 DoS。一句话记忆:**失陷必须被「断」住,而不是被「转发」。**
- 陷阱:有人把这三者混为一谈——熔断是「时间维度」(失败累积就断路径),隔离舱是「空间维度」(资源分区互不影响),fan-out cap 是「数量维度」(限扇出度数)。

**Q4(追问)为什么级联失败常演变成成本型 DoS?**
标准答:失控 agent 互相触发,每次触发都是真实 API 调用 + 算力消耗,指数扇出会把 token/算力预算烧穿(链 #12)。所以 fan-out cap + 预算配额是「一石二鸟」的防御。

**Q5:A2A Agent Card 签名后还要做什么?**
标准答:A2A 1.0.0 的 Card 签名是可选 JWS;当用它建立信任时,从可信密钥库或安全渠道取 key、按规范验签并检查 key 未过期/撤销,再把 Card 只当「能力描述与身份线索」。每个请求仍要有 TLS、调用方认证、每操作授权范围和幂等/重放防护。**陷阱**:「有 JWS 就不用授权」是错的——完整性和来源不等于当前调用可访问该资源。

## 知识拓展
- **框架定位**:OWASP **ASI07 Insecure Inter-Agent Communication / ASI08 Cascading Failures**(Top 10 for Agentic Applications,2025-12-09);MITRE ATLAS 2025-10 新增的 agent 技术含跨 agent 传播/协同。
- **协议**:**A2A(Agent2Agent) 1.0.0**(截至 2026-07 的最新正式规范,Agent Card 可选 JWS)、[[17 MCP 模型上下文协议|MCP]];A2A 安全研究见 arXiv:2504.16902《Building A Secure Agentic AI Application Leveraging A2A》。
- **复用的成熟工程**:零信任(NIST SP 800-207)、service mesh mTLS(Istio/Linkerd + SPIFFE)、弹性模式(circuit breaker / bulkhead,源自 Hystrix / Polly / Resilience4j)。
- **对抗演化**:攻击从「伪造单条消息」→「劫持信任链做级联」→「跨租户外溢」→「协同串谋(见 [[19 人机信任利用与 Rogue Agents|流氓 agent]] 的 collusion)」;防御从「加 mTLS」→「零信任默认不信 peer」→「弹性兜底让失陷可被断住」→「跨 agent 可观测+可回放定位级联起点」。

## 相邻
- 上游:[[16 Agent 身份与权限滥用(非人类身份 NHI)|身份滥用]]、[[17 Memory 与 Context Poisoning|记忆投毒]]
- 下游:[[19 人机信任利用与 Rogue Agents|人机信任与流氓 agent]]、[[20 Agentic 供应链与 MCP 安全|MCP 安全]]
- 跨域:[[22 多智能体系统|多智能体系统]]、[[26 Sub-agents 与 Agent Teams|Sub-agents]]
- 框架定位:[[26 安全框架与治理地图|安全框架地图]]、[[01 AI 安全总览与三层栈|三层栈]]
