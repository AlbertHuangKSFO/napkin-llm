[[16 Agent 身份与权限滥用(非人类身份 NHI)|Agent 身份]]的核心认知:agent 不是人,而是一类新的**非人类身份(Non-Human Identity, NHI)**——它持有凭证、访问系统、跨服务调用,数量远超人类身份却常无治理。当 [[14 Excessive Agency 与 Goal Hijack|过度代理]] 落地为越权访问、token 被窃取重放、或 [[15 Tool Misuse 与意外代码执行|confused deputy]] 越权,根因往往在身份与权限管理。它对应 OWASP ASI03 身份与权限滥用。

## 为什么 NHI 是新问题
- 人类身份有 SSO、MFA、生命周期管理;NHI 长期靠**写死的长效 API key**,泄露即被冒用,无到期、无溯源、无逐 agent 审计。
- agent 数量爆炸式增长(一个工作流可拉起多个 sub-agent),共享凭证让「哪个 agent、哪次任务」无从区分。
- **over-privileged agent 凭证**:为省事给一把 admin key,任务只需读却拥有全部写权——一旦被劫持,后果无上限。

## 反模式 vs 正模式
![[sec-非人类身份NHI.png]]

| 维度 | ✗ 静态长效密钥 | ✓ 可验证工作负载身份 |
|---|---|---|
| 凭证形态 | 写死的长效 API key | 短时 SVID / JIT token,自动轮换 |
| 身份 vs 凭证 | 二者混同 | 身份唯一、凭证短命可换 |
| 泄露后果 | 长期可重放 | 窗口极短,重放价值趋零 |
| 越权 | 复用高权,confused deputy | 窄 scope + 身份透传 |
| 审计 | 共享凭证难溯源 | 逐 agent 可审计、可撤销 |

## Agent 身份方案
- **OAuth(client credentials / on-behalf-of)**:给 agent 颁发受限 scope 的 access token,人类委派时用 OBO 流让 agent 按用户身份动作,而非复用自身高权——这是堵 confused deputy 的关键。
- **SPIFFE / SPIRE 工作负载身份**:每个 agent 拿一个 **SPIFFE ID**,由 SPIRE 基于运行环境的**证明(attestation)**签发短时 **SVID**(X.509 证书或 JWT),自动轮换,**不依赖长效密钥**。身份基于「你是什么工作负载」而非「你持有哪把密钥」。
- **JIT / 短时凭证**:只有在 agent 有「已批准的意图 + 明确任务 + 限定执行窗口」时才发凭证,过期即废,把泄露窗口压到最小。

## 防御
- **最小权限**:每个 agent 窄 scope,按任务降权;删除/支付类只读或受控。呼应 [[24 沙箱、最小权限与人审闸门|最小权限]]。
- **短命化凭证**:长效 key → SVID/JIT,自动轮换,token 重放价值趋零。
- **身份透传**:工具与下游按真实调用者身份鉴权,不复用 agent 高权,堵 confused deputy。
- **逐 agent 审计与撤销**:唯一身份让每次动作可归因、可即时吊销。
- agent 间调用同样要带身份、校验完整性:见 [[18 Agent 间通信安全与级联失败|agent 间通信]]。

## 关键事实(含出处)
- OWASP **ASI03 Identity and Privilege Abuse**:agent 滥用凭证、token 或继承的权限,访问超出预期范围的系统或数据(OWASP Top 10 for Agentic Applications 2026, genai.owasp.org)。
- **SPIFFE**(Secure Production Identity Framework For Everyone)为工作负载定义安全身份标准;**SPIRE** 是其运行时,基于环境证明签发**短时、自动轮换的 SVID(SPIFFE Verifiable Identity Document)**,替代静态 API key / 长效密钥(hashicorp.com;nhimg.org)。
- **JIT for agents**:身份平台仅在 agent 有「approved intent + defined task + bounded execution window」时签发短时凭证(nhimg.org)。
- NHI(非人类身份)数量远超人类身份,治理缺口被 CSA 等列为 agentic AI 的核心风险面(labs.cloudsecurityalliance.org)。

## 工业界实践
真实的 NHI 治理不是「换个密钥管理工具」,而是把 agent 当一等公民纳入身份生命周期(发现 → 颁发 → 轮换 → 撤销 → 审计)。

**1. NHI 发现与盘点(治理第一步)**
你不能保护看不见的东西。专做 NHI 的厂商(**Astrix Security、Token Security、Oasis Security、Entro、GitGuardian**)先扫遍 IdP、云、SaaS、代码仓库,把所有 service account / API key / OAuth app / agent 凭证编目,标注 owner、最后使用时间、权限范围、是否过度授权,优先清理「僵尸 NHI」(无主、长期不用却仍持高权)。Gartner 已把 NHI 管理列为 2025 战略趋势。

**2. 短时凭证替代长效密钥(SPIFFE/SPIRE 实战)**
SPIRE Server 作签发权威,SPIRE Agent 跑在每个节点上,通过 **Workload API** 给本地工作负载发 SVID。颁发前先做**多因子证明(attestation)**——同时核验节点身份(k8s node、EC2 instance)+ 进程选择器(uid、k8s service account、容器镜像哈希),只有全部命中注册表里登记的条件才发身份。SVID 默认 **~1 小时轮换**,泄露窗口极短。规模佐证:Uber 每天签发**超过 10 亿** SPIFFE 凭证。HashiCorp Vault Enterprise 已原生支持 SPIFFE 认证 NHI 工作负载(含 AI agent)。

```yaml
# SPIRE 注册:把 SPIFFE ID 绑死到「k8s sa + 镜像」组合,而非密钥
# 只有同时满足这些选择器的工作负载才能拿到这个身份
spiffe ID: spiffe://corp.com/ns/agents/sa/research-agent
parent ID: spiffe://corp.com/spire/agent/k8s_psat/cluster-a/...
selectors:
  - k8s:ns:agents
  - k8s:sa:research-agent
  - k8s:container-image:sha256:abc123...   # 镜像被换 → 身份发不出来
ttl: 3600   # SVID 1 小时过期,自动轮换,无长效 key
```

**3. JIT + 最小权限的工程落法**
不预先给 agent 常驻高权,而是任务触发时按「approved intent + defined task + bounded window」临时提权(类似云上的 **PAM / privilege elevation**:AWS STS `AssumeRole` 拿短时 session token、GCP service account impersonation、Vault dynamic secrets 现签现用)。下面是「按任务降权」的伪代码:

```python
# 反模式:agent 全程持 admin key —— 一旦被劫持后果无上限
# agent = Agent(creds=ADMIN_KEY)  # ✗

# 正模式:每个任务现申请窄 scope 的短时令牌(JIT)
def run_task(task):
    assert task.intent in APPROVED_INTENTS          # 必须是已批准意图
    creds = sts.assume_role(                          # AWS STS:拿临时凭证
        role=ROLE_FOR[task.intent],                   # 按意图选最小角色
        duration_seconds=900,                         # 15 分钟即过期
        session_tags={"agent_id": agent.id,           # 透传真实调用者身份
                      "task_id": task.id})             # → 可逐任务审计
    return execute(task, creds)                       # 用窄 scope 短命凭证执行
```

**4. confused deputy 的工业堵法:身份透传(token exchange)**
人委派 agent 调下游时,用 **OAuth 2.0 Token Exchange(RFC 8693)/ On-Behalf-Of 流**把「真实用户身份」透传到每一跳,下游按用户而非按 agent 的高权鉴权。OpenID Foundation 2025 推进的 **Identity Assertion Authorization Grant** 正是为 agent 跨服务带原始用户身份设计。

**5. 检测与响应**
- 行为基线:某 agent 凭证突然访问从没碰过的 S3 bucket / 在非常规时段批量调用 → 告警。
- 凭证泄露即撤销:唯一身份让 kill 单个 agent 不影响其他 agent(对比共享 key「一撤全断」)。
- 全量审计:每次动作可归因到具体 agent + task + 原始用户。

## 面试高频

**Q1:为什么 agent 的身份问题比传统 service account 更棘手?**
标准答:① 数量爆炸——一个工作流拉起多个 sub-agent,NHI 数量远超人类身份;② 凭证常是写死的长效 key,无 MFA、无生命周期、泄露即长期可重放;③ agent 行为非确定、会被 prompt 劫持,持高权时破坏面无上限;④ 共享凭证让「哪个 agent、哪次任务」无法溯源。
- 追问「具体怎么解」:身份与凭证解耦(SPIFFE ID 是身份,SVID 是短命凭证)+ JIT 提权 + 身份透传 + 逐 agent 审计。

**Q2:什么是 confused deputy?在 agent 里怎么发生、怎么堵?**
标准答:被授权的「代理」(agent/工具)被诱导用**自己的高权**去执行**攻击者**的意图——权限属于 deputy,意图属于攻击者。agent 场景:用户 A 让 agent 读自己文档,prompt injection 诱导 agent 用其 admin 权限去读用户 B 的文档。堵法:**身份透传**——下游按真实调用者(用户 A)鉴权,而非复用 agent 高权;窄 scope;每跳重新授权。
- 陷阱:很多人答「加 guardrail 过滤恶意输入」——这是缓解不是根治;根因是**权限模型**(deputy 不该持有超出本次请求所需的权限)。

**Q3:SPIFFE/SPIRE 凭什么比 API key 安全?它解决了什么、没解决什么?**
标准答:解决——基于**工作负载是什么**(节点+进程证明)而非**持有什么密钥**来发身份;SVID 短命自动轮换,泄露价值趋零;无需到处散布长效 secret。没解决——拿到 SVID 之后 agent **在其 scope 内**仍可被 prompt 劫持去干坏事(身份对了不代表意图对),所以还要叠最小权限、人审闸门、行为审计。
- 陷阱:别把「身份」当「授权」——SPIFFE 解决 authn(你是谁),authz(你能干啥)仍需独立的最小权限策略。

**Q4(追问)为什么不彻底?**
短时凭证缩小的是「泄露后可重放的时间窗」,不是「合法窗口内被滥用」;JIT 也挡不住 agent 在已批准任务内被注入诱导做越权动作。身份层只是纵深防御一环,必须和 [[15 Tool Misuse 与意外代码执行|工具层鉴权]]、[[24 沙箱、最小权限与人审闸门|沙箱/人审]] 叠加。

## 知识拓展
- **框架定位**:OWASP **ASI03 Identity & Privilege Abuse**(Top 10 for Agentic Applications,2025-12 发布);MITRE ATLAS 在 2025-10 联合 Zenity Labs 新增多条 agent 专项技术,凭证滥用/权限提升落在其中;CSA(云安全联盟)把 NHI 治理缺口列为 agentic AI 核心风险面。
- **标准**:SPIFFE/SPIRE(CNCF 毕业项目);OAuth 2.0 Token Exchange(RFC 8693);OpenID Foundation 2025 的 **Identity Assertion Authorization Grant**(为 agent 跨服务透传用户身份)。
- **生态**:NHI 安全赛道(Astrix、Token Security、Oasis、Entro、GitGuardian)与传统 PAM(CyberArk)、密钥管理(HashiCorp Vault)正在融合出「agent identity」品类。
- **对抗演化**:防御从「藏好长效 key」→「让 key 短命到不值得偷」→「连 key 都不要、用可证明的工作负载身份」;攻击则从偷 key → 偷活动 session/SVID → 在合法身份的合法窗口内做 confused deputy。下一战场是**意图层授权**(不只验「你是谁」,还验「这个动作符不符合本次被授权的意图」)。

## 兄弟链
- [[15 Tool Misuse 与意外代码执行|工具滥用]] — confused deputy 的工具层表现
- [[18 Agent 间通信安全与级联失败|agent 间通信]] — 多 agent 场景下身份的延伸
- [[24 沙箱、最小权限与人审闸门|最小权限]] — 权限收窄的总篇
