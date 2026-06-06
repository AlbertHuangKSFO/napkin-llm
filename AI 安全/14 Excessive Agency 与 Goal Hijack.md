[[14 Excessive Agency 与 Goal Hijack|过度代理]]是「攻击如何落地」的总开关:[[05 Prompt Injection 提示注入|提示注入]]在模型层只是一段不安全文本,但当 agent 持有权限、自主性和功能时,这段文本就被规划进一次**真实 API 调用**——目标被悄悄替换(Goal Hijack),后果变成现实。它对应 OWASP LLM06 过度代理 + ASI01 目标劫持,是整个 Agentic 威胁链的「放大器」。

## 两个框架,一条因果链
- **LLM06 过度代理(Excessive Agency)**讲的是「条件」:系统给了 agent 太多自由度。
- **ASI01 目标劫持(Agent Goal Hijack)**讲的是「事件」:攻击者操纵 agent 的目标、指令或决策路径,让它去追求非预期目标。
- 前者是干柴,后者是火星。没有过度代理,目标劫持也烧不起来——这正是 [[02 Workflow 与 Agent 的边界|Workflow vs Agent 边界]] 里「最小自由度」的安全含义:**给 agent 的每一分自主性都是攻击面**,能用 Workflow(固定路径)解决就别给 Agent(自主决策)放权。

## 过度代理的三个维度
OWASP LLM06 把「过度」拆成三轴,缓解就是逐轴收窄:

| 维度 | 含义 | 危险示例 | 收窄手段 |
|---|---|---|---|
| 过度功能(functionality) | 挂了用不到的工具 | 只需读邮件却给了删除工具 | 工具白名单,只留最小必要 |
| 过度权限(permissions) | scope 比任务宽 | 用 admin 凭证跑只读任务 | 窄 scope、按需降权 |
| 过度自主(autonomy) | 不可逆动作不经人审 | 自动转账/部署/删库 | 人审闸门拦在动作前 |

## 目标劫持链:文本 → 规划 → 行动
![[sec-目标劫持链.svg]]

链条是:① 注入源(网页、邮件、文档、工具返回值里藏「忽略原任务,改去…」)→ ② Agent 规划阶段把注入文本当指令,目标被替换 → ③ 工具调用执行 send/delete/purchase/deploy 等不可逆动作 → ④ 现实后果(资金外流、数据删除、越权部署)。**任一环收窄都能斩断这条链**,这是分层防御的着力点。

## 防御
- **最小权限范围**:工具白名单 + 窄 scope,删除/支付/部署给只读或受限凭证,斩断「规划→行动」。详见 [[24 沙箱、最小权限与人审闸门|最小权限]]。
- **人审闸门(HITL)**:不可逆、高影响动作前停下,人确认才放行,斩断「行动→现实后果」。
- **能力上限**:限步数、限金额、限调用频次,即便被劫持也限定后果幅度。
- **从源头治理注入**:见 [[05 Prompt Injection 提示注入|提示注入]];把不可信内容与指令隔离、标注来源。
- 工具本身的设计也是防线:见 [[16 工具设计与工具层|工具层]]——窄接口、强类型参数、最小副作用的工具天然更难被滥用。

## 关键事实(含出处)
- OWASP **LLM06:2025 Excessive Agency** 官方将其定义为「LLM 被授予过多 functionality / permissions / autonomy,从而能执行非预期或有害动作」,三维度为业界共识(OWASP Gen AI Security Project, genai.owasp.org)。
- OWASP **Top 10 for Agentic Applications 2026** 的 **ASI01 = Agent Goal Hijack**:攻击者操纵 agent 的目标、指令或决策路径使其追求非预期动作(genai.owasp.org,2025-12-09 发布,100+ 专家评审)。
- 业界共识:过度代理是「提示注入变成可操作后果」的节点——模型不只是答错,而是能 send/delete/purchase/deploy/expose 真实资产。

## 工业界实践
过度代理在工业界是**最贵的一类 bug**:它不是模型答错,而是 agent 拿着真实凭证去 send/delete/purchase/deploy。防御的核心思路是把"agent 能做什么"从"模型决定"挪回"工程约束"。

**纵深防御架构(逐轴收窄 + 拦在动作前)**:
- **过度功能 → 工具白名单**:只挂任务必需的工具。只读任务不给写/删工具;按 agent 角色发不同工具集(读邮件 agent ≠ 发邮件 agent)。
- **过度权限 → 窄 scope + 按需降权**:删除/支付/部署用受限或只读凭证;OAuth scope 精确到操作;凭证短时效(见 [[16 Agent 身份与权限滥用(非人类身份 NHI)|NHI 身份]])。
- **过度自主 → 人审闸门(HITL)**:不可逆、高影响动作前停下,人确认才放行。这是斩断"行动→现实后果"的最后一道闸。
- **能力上限**:限步数、限金额、限调用频次,即便被劫持也限定后果幅度。

**真实案例:EchoLeak(CVE-2025-32711,2025-06,Aim Security 披露)**——首个生产 LLM 系统的**零点击间接提示注入**。攻击者只发一封特制邮件给受害者,Microsoft 365 Copilot 自动检索到它就被劫持目标:绕过 Microsoft 的 XPIA 注入分类器、用 reference-style Markdown 规避链接脱敏、靠自动抓取图片 + Teams 代理(被 CSP 放行)把公司机密**外传**——全程受害者零交互。它完美演示了"过度代理 + 目标劫持":Copilot 有检索 + 渲染 + 外联的功能/权限/自主,注入文本把这些能力规划成一次数据外泄。

**检测与护栏工具**:
- **输入/输出 guardrail**:NVIDIA **NeMo Guardrails**、**Lakera Guard**、**Llama Guard / Prompt Guard**、Protect AI、Robust Intelligence,在 agent 规划前后扫注入与越权意图。
- **agent 框架内建**:LangChain/LangGraph 的 human approval node、AutoGen 的 `human_input_mode`、OpenAI Agents SDK 的 guardrails + tool 审批——把 HITL 做成图里的显式节点。
- **Enkrypt AI / 各 Agent Gateway**:分层 guardrail + 风险分类(agent 授权劫持、checker-out-of-the-loop、关键系统交互、目标/指令操纵)。

**误报/延迟与"HITL 疲劳"权衡**:HITL 是最强防线但有致命弱点——**攻击者可用海量复杂交互淹没人审**,把人逼成橡皮图章(rubber-stamping),OWASP Agentic 明确把"用复杂度压垮 HITL"列为新风险。工业界做法:**只对不可逆/高金额/跨信任边界动作触发人审**(可逆动作放行 + 事后审计),配合金额/频次阈值自动分流,避免每步都问人导致疲劳。

```python
# 动作前闸门:按"可逆性 + 影响"决定放行 / 人审 / 拒绝(伪代码)
IRREVERSIBLE = {"transfer_funds", "delete_resource", "deploy", "send_external_email"}
def gate(action, args, ctx):
    if action not in TOOL_ALLOWLIST[ctx.agent_role]:   # 过度功能:不在白名单直接拒
        raise PermissionError("tool not allowed for role")
    if action in IRREVERSIBLE or args.get("amount", 0) > ctx.budget_cap:
        return require_human_approval(action, args)    # 过度自主:不可逆/超额 → 人审
    if ctx.step_count > ctx.max_steps:                 # 能力上限:防失控循环
        raise RuntimeError("step budget exceeded")
    return EXECUTE
```

## 面试高频
**Q1:"Excessive Agency 和 Prompt Injection 是什么关系?"**(高频送分但要答出因果)提示注入在模型层只是一段不安全文本;**过度代理是把这段文本变成可操作后果的节点**——agent 持有功能/权限/自主时,注入文本被规划进真实 API 调用,目标被悄悄替换(Goal Hijack)。一句话:注入是火星,过度代理是干柴。
- 陷阱:面试官问"那我把注入防住不就行了?"→ 注入防御不可能 100%(EchoLeak 就绕过了分类器),所以必须假设注入会成功,在**动作侧**用最小权限 + HITL 兜底,这才是分层防御。

**Q2:"OWASP 把过度代理拆成哪三轴?对应怎么收窄?"** 过度功能(挂了用不到的工具 → 工具白名单)、过度权限(scope 比任务宽 → 窄 scope/降权)、过度自主(不可逆动作不经人审 → HITL 闸门)。逐轴收窄就是缓解。

**Q3:"人审闸门(HITL)是银弹吗?为什么不彻底?"** 不是。① **HITL 疲劳**:攻击者用海量复杂交互淹没人审,人变橡皮图章;② 覆盖不全:可逆动作放行后仍可能被链式利用;③ 延迟/可用性:每步都问人则 agent 没意义。所以 HITL 必须配最小权限 + 能力上限 + 注入治理,且只拦不可逆/高影响动作。
- 追问"Workflow 和 Agent 哪个更安全?"→ Workflow(固定路径)更安全,因为没有自主决策面;**给 agent 的每一分自主性都是攻击面**,能用 Workflow 解决就别放权给 Agent(最小自由度原则)。

## 知识拓展
- **框架对照**:OWASP **LLM06:2025 Excessive Agency**(functionality/permissions/autonomy 三轴);OWASP **Top 10 for Agentic Applications 2026** 的 **ASI01 Agent Goal Hijack**(2025-12-09 发布,100+ 专家评审)。MITRE **ATLAS**(v5.1.0,2025-11 扩到 16 战术 / 84 技术 / 32 缓解,2026-02 续加 agentic 技术)对应缓解:**AML.M0026 最小权限**、**AML.M0029 关键动作人审**、**AML.M0030 限制不可信数据触发工具调用**。NIST **AI RMF** 的 Govern/Manage 强调自主系统的人类监督与可逆性。
- **真实漏洞**:EchoLeak(CVE-2025-32711,M365 Copilot 零点击注入→数据外泄,2025-06)。
- **工具生态**:NeMo Guardrails、Lakera Guard、Llama Guard / Prompt Guard、Enkrypt AI、Agent Gateway;LangGraph/AutoGen/OpenAI Agents SDK 的 HITL 节点。
- **对抗演化**:从"显式越权指令"→"间接注入(从检索内容劫持)"→"零点击(EchoLeak,受害者无交互)"→"压垮 HITL(用复杂度淹没人审)",防御从内容过滤升级到动作侧最小权限 + 分级人审 + 能力上限。

## 兄弟链
- [[15 Tool Misuse 与意外代码执行|Tool Misuse]] — 劫持后落地的具体工具滥用形态
- [[16 Agent 身份与权限滥用(非人类身份 NHI)|Agent 身份]] — 「最小权限」在身份层的落地
- [[17 Memory 与 Context Poisoning|记忆投毒]] — 注入的另一持久化入口
- [[24 沙箱、最小权限与人审闸门|沙箱与人审]] — 缓解措施的集中篇
