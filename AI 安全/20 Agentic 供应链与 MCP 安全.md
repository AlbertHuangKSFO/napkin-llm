[[20 Agentic 供应链与 MCP 安全|本篇]]把 **ASI04 不安全的 agent 供应链**落到一个活跃攻击面——**MCP**。[[17 MCP 模型上下文协议|MCP 协议]]让 agent 即插即用第三方工具,等于把「随便装个浏览器扩展」的供应链风险搬进了自主系统:工具来源不可信、描述可藏指令、结果可回流注入、宿主环境可被穿透。OWASP 的 **MCP Top 10 v0.1** 将 MCP01–MCP10:2025 作为 beta living document 发布。

## MCP 四处攻击面
MCP host(agent)的工作流是:连接 server → 读工具描述 → 调工具 → 收结果回流。**这三步每一步都能被武器化**:

- **① 工具投毒(Tool Poisoning,MCP03)**:把隐藏指令藏进**工具描述**里。agent 一连上 server,描述就会进入 context window——**工具甚至无需被真正调用**。具体攻击成功率必须随基准、模型、host 的审批策略、工具集合、judge 和日期一起报告,不能迁移为通用风险数字。
- **② Rug-pull 工具**:首次审核良性、获得信任后,服务端**事后改行为**。不 pin 版本就会无声变恶意,本质是供应链的「拉地毯」。
- **③ 间接提示注入(经工具结果回流)**:恶意指令藏在**工具返回的结果**里(抓来的网页、文件、DB 记录),随结果流回 agent 上下文,劫持其后续行为(链 [[05 Prompt Injection 提示注入|提示注入]])。
- **④ SSRF → 云元数据窃取**:server 不校验 URL,可能被引向 `http://169.254.169.254` 等链路本地元数据端点;若运行环境、网络 egress 与实例角色配置都允许,风险可能扩大为临时凭证泄露。它是工具/宿主的 SSRF 与网络边界问题,**不应贴成 OWASP MCP04**;MCP04 专指软件供应链攻击与依赖篡改。分类要按真实成因和控制点做,而不是为每个漏洞硬找一个 Top 10 标签。

![[sec-MCP攻击面.png]]

## 关键事实(含出处)
- **OWASP MCP Top 10 v0.1** 是 OWASP 的 MCP 专项 Top 10,条目为 **MCP01–MCP10:2025**;截至 **2026-07-18** 仍在 Phase 3 beta/试点,不是最终稳定标准。MCP03 为工具投毒,MCP04 为软件供应链与依赖篡改,MCP07 为认证/授权不足,MCP08 为审计/遥测缺失。来源:[OWASP MCP Top 10 v0.1](https://owasp.org/www-project-mcp-top-10/)。
- **SSRF 风险卡**:判断一个 URL 型 MCP 工具能否触及云元数据,至少要说明产品/版本、URL 解析与 DNS 重绑定防护、可达网段、是否强制 IMDSv2、实例角色权限和测试日期。没有这些方法卡的「扫描到多少 server」「成功率多少」不能外推为生态结论。AWS 对 IMDSv2 的 token 请求与 hop limit 机制见 [AWS EC2 User Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-service.html)(核验:2026-07-18)。

**防御预算手算(而非虚构生态统计)。** 设网关只允许 12 个已审核 server,每个 server 每分钟最多 20 次工具调用,高危写操作另需人工批准。则一个网关身份的自动化调用上界为 $12\times20=240$ 次/分钟;如果将高危写操作路由为 100% 人审,其自动执行上界为 $0$。这是可配置、可观测、可复测的本地控制量,比把不同扫描方法的外部百分比相乘更能指导容量和事故演练。

## 防御
把每个 MCP server 当作**不可信的外部依赖**对待:
- **来源校验 + pin 版本**:只接入可信来源,用签名/哈希锁定版本,防 rug-pull(对应 MCP04)。
- **沙箱 + 最小权限**:限制 server 的网络出口与可达资源;在适用云环境强制 IMDSv2、屏蔽链路本地与元数据网段 `169.254.0.0/16`,并把 egress 收敛到 allowlist。这是 SSRF 的纵深防御,不能替代 URL 解析、重定向/DNS 重绑定防护与最小权限实例角色(链 [[24 沙箱、最小权限与人审闸门|沙箱、最小权限]])。
- **工具结果当不可信输入**:回流进上下文前过 guardrail 过滤(链 [[21 Guardrails 与输入输出防护|Guardrails]])。
- **工具描述审计**:扫描描述里的隐藏指令;高危工具需人审才挂载。
- **治理面**:禁用未审批的 **shadow MCP server**(MCP09);**凭证不进提示与日志**(MCP01);整体对齐 OWASP **MCP Top 10** + **ASI04** 供应链(注意它与[[10 供应链安全(静态)|静态供应链]]互补——后者管模型/数据/包,本篇管运行时工具协议)。

## 工业界实践
核心心智:**把每个 MCP server 当不可信的外部依赖**——像对待第三方浏览器扩展/npm 包那样审、锁、隔离、监控。

**1. MCP 网关:统一的进出闸**
不让 agent 直连各 server,而是走一个 **MCP 网关/代理**集中管控——认证、授权、扫描、日志、速率限制都在网关做。代表:**Invariant Labs `mcp-scan`**(2025 早期就披露了 tool poisoning,提供 `scan` 静态扫描工具描述 + `proxy` 运行时拦截 MCP 流量做 guardrail)、各云厂商的 MCP gateway。网关是落地零信任 MCP 的关键支点。

```python
# MCP 网关拦截:挂载前扫描 + 运行时过滤(把 server 当不可信依赖)
def on_mount_server(server):
    desc = fetch_tool_descriptions(server)
    if scanner.has_hidden_instructions(desc):       # ① 工具描述审计(MCP03)
        block(server, "tool_poisoning_suspected")   #    扫"忽略以上指令/隐藏字符"
    pin = lockfile.get(server.id)
    if hash(desc) != pin.hash:                       # ② pin 版本防 rug-pull(MCP04)
        require_human_review(server)                 #    描述变了 → 人审才放行

def on_tool_result(result, server):
    if injection_filter.is_suspicious(result):      # ③ 结果回流当不可信输入
        result = sanitize(result)                    #    防间接注入(链 #5)
    redact_secrets(result)                           # ④ 凭证不进上下文/日志(MCP01)
    return result
```

**2. 堵 SSRF→云元数据(最响的真实案例)**
- **强制 IMDSv2**:IMDSv1 不需要 session token;IMDSv2 要求先取得会话 token,并允许配置 hop limit。它降低某些 SSRF 路径的可利用性,但不应被表述为单点万能修复;仍需 egress 控制、URL 校验和最小权限。
- **网络出口管控**:在 server 沙箱里**屏蔽链路本地与元数据网段 `169.254.0.0/16`**(含 `169.254.169.254`),egress allowlist 只放行业务必需域名。
- **URL 校验**:接受 URL 参数的工具做 SSRF 防护(解析后校验目标 IP 不落私网/元数据段,防 DNS rebinding)。

**3. 供应链锁定**
- **来源校验 + pin 版本**:只接可信来源,用签名/哈希锁版本,防 rug-pull(MCP04);MCP server 也纳入 SBOM 与依赖扫描。
- **禁用 shadow MCP server(MCP09)**:盘点并阻断未审批、影子接入的 server——这是 agent 版的「影子 IT」。
- **凭证治理(MCP01)**:server 的 token/key 走 secret manager,绝不进 prompt、不落日志;按 [[16 Agent 身份与权限滥用(非人类身份 NHI)|短时凭证]] 发放。

**4. 检测响应**
- 网关全量记录 MCP 调用(server / tool / 参数 / 结果哈希),异常基线告警(某 server 突然要读 `~/.ssh`、访问元数据 IP)。
- 扫描/网关工具只能提供一个检测层;采购或上线前应对自己的 server 集合做可复现的来源、版本、规则、误报/漏报和日期记录。

## 面试高频

**Q1:MCP 的四处攻击面是什么?各自怎么防?**
标准答:① **工具投毒(MCP03)**——隐藏指令藏在工具描述里,agent 一连上就被注入,**工具无需被真正调用**;防:扫描描述、人审挂载。② **rug-pull/依赖篡改**——先良性获信任、事后改恶意;防:pin 版本、来源校验、签名/SBOM 与持续监控。③ **间接提示注入(结果回流)**——恶意指令藏在工具返回结果里随上下文回流;防:结果当不可信输入过 guardrail。④ **SSRF→云元数据**——URL 工具被诱导访问内部/元数据地址;防:URL/DNS/重定向校验、egress allowlist、IMDSv2 与最小权限。**分类陷阱**:SSRF 不是 MCP04;MCP04 是软件供应链/依赖篡改。
- 陷阱:工具投毒最反直觉的点是「**不调用也中招**」——只要连上 server,描述就进了 context window。

**Q2:讲一个 MCP SSRF 取 IAM 凭证的完整链路。**
标准答:server 跑在云环境、接受 URL 参数、且未校验 URL/重定向/DNS 重绑定 → 攻击者诱导它访问链路本地元数据地址 → 若网络和角色权限允许,可能取到短期凭证并扩大权限。堵法:URL/DNS/重定向校验、egress allowlist、屏蔽 `169.254.0.0/16`、IMDSv2 与最小权限实例角色。
- 追问「为什么 IMDSv2 有帮助」:它引入 token 请求与 hop-limit 等额外条件,使部分简单 SSRF 路径无法直接读取元数据;但仍须实测当前代理、网络与重定向配置,不能把它当唯一防线。

**Q3:为什么说 MCP 安全是「运行时供应链」,和静态供应链什么关系?**
标准答:[[10 供应链安全(静态)|静态供应链]]管模型/数据/包(交付前),MCP 安全管**运行时即插即用的工具协议**(运行中动态接入第三方 server)。二者互补:静态管「装进来的东西干不干净」,MCP 管「运行时连上的工具能不能信、会不会事后变坏(rug-pull)」。对应 OWASP **ASI04 不安全的 agent 供应链**。

**Q4(追问)为什么不彻底?**
标准答:① 工具描述是自然语言、要喂给模型才有用,扫描器与对抗改写永远在拉锯;② rug-pull 是「事后变恶意」,首次审核通过不代表永远安全,只能靠 pin + 持续监控缩短暴露;③ 结果回流注入本质是 [[05 Prompt Injection 提示注入|提示注入]],而提示注入至今无完美解。所以 MCP 安全靠网关 + 沙箱 + 最小权限 + 监控的纵深,而非单点。

## 知识拓展
- **框架定位**:OWASP **MCP Top 10 v0.1(MCP01–MCP10:2025,截至 2026-07 仍 beta)**——上层对齐 **ASI04 不安全 agent 供应链**(OWASP Top 10 for Agentic Applications,2025-12)。代表条目:MCP01 令牌/凭证暴露、MCP03 工具投毒、MCP04 软件供应链/依赖篡改、MCP05 命令注入、MCP09 影子 MCP server。
- **基准与数据**:MCPTox、MCP 扫描报告和单个 SSRF PoC 都只能作为方法明确的风险信号。引用时至少附模型/host 审批策略、工具集、版本、指标定义、样本采样方式和日期;没有这些信息时改为定性结论。
- **工具生态**:Invariant Labs `mcp-scan`(静态扫描 + 运行时 proxy guardrail)、Snyk Labs、Akto、Lasso 等 MCP 安全扫描/网关赛道。
- **对抗演化**:攻击从「直连 server 投毒」→「rug-pull 事后变恶意」→「结果回流间接注入」→「SSRF 穿透宿主取云凭证」;防御从「人工审 server」→「自动化扫描描述」→「网关统一管控 + 运行时拦截」→「把 MCP server 纳入 SBOM/零信任/最小权限的标准供应链治理」。

## 相邻
- 上游:[[18 Agent 间通信安全与级联失败|级联失败]]、[[19 人机信任利用与 Rogue Agents|流氓 agent]]
- 同源:[[10 供应链安全(静态)|静态供应链]]、[[05 Prompt Injection 提示注入|提示注入]]
- 防御侧:[[21 Guardrails 与输入输出防护|Guardrails]]、[[24 沙箱、最小权限与人审闸门|沙箱]]
- 跨域:[[17 MCP 模型上下文协议|MCP 协议]]、[[16 工具设计与工具层|工具设计与工具层]]
- 框架定位:[[26 安全框架与治理地图|安全框架地图]]、[[01 AI 安全总览与三层栈|三层栈]]
