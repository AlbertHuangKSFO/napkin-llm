[[20 Agentic 供应链与 MCP 安全|本篇]]把 **ASI04 不安全的 agent 供应链**落到 2026 最热的攻击面——**MCP**。[[17 MCP 模型上下文协议|MCP 协议]]让 agent 即插即用第三方工具,等于把「随便装个浏览器扩展」的供应链风险搬进了自主系统:工具来源不可信、描述可藏指令、结果可回流注入、宿主环境可被穿透。OWASP 专门为它出了一套 **MCP Top 10(MCP01–10,2025 beta)**。

## MCP 三处攻击面
MCP host(agent)的工作流是:连接 server → 读工具描述 → 调工具 → 收结果回流。**这三步每一步都能被武器化**:

- **① 工具投毒(Tool Poisoning,MCP03)**:把隐藏指令藏进**工具描述**里。agent 一连上 server,描述就被注入 context window——**工具甚至无需被真正调用**。MCPTox 基准在开启 auto-approval 时,工具投毒成功率高达 **84.2%**。
- **② Rug-pull 工具**:首次审核良性、获得信任后,服务端**事后改行为**。不 pin 版本就会无声变恶意,本质是供应链的「拉地毯」。
- **③ 间接提示注入(经工具结果回流)**:恶意指令藏在**工具返回的结果**里(抓来的网页、文件、DB 记录),随结果流回 agent 上下文,劫持其后续行为(链 [[05 Prompt Injection 提示注入|提示注入]])。
- **④ SSRF → 云元数据窃取**:server 不校验 URL,被引向 `http://169.254.169.254`(IMDSv1)即可取出 **AWS IAM 临时凭证**——属于 MCP04 供应链/依赖与宿主环境穿透。

![[sec-MCP攻击面.png]]

## 关键事实(含出处)
- **BlueRock** 分析 **7000+ MCP server**,发现约 **36.7% 疑似存在 SSRF** 暴露。来源:[BlueRock — MCP fURI / Markitdown SSRF](https://www.bluerock.io/post/mcp-furi-microsoft-markitdown-vulnerabilities)。

**攻击面手算(把三个统计串起来)**。这几个数字单看是孤立的,串起来才是一条放大链:① BlueRock 的 7000 个 server × 36.7% ≈ **2570 个潜在 SSRF 入口**——每一个都可能被引向 `169.254.169.254` 取云凭证;② 这 2570 里只要混进 Trend Micro 报的那类**零认证、互联网直接暴露**的 server(实测 492 个),就是无需任何门槛、扫到即打的高危目标;③ 而一旦攻击者能把一条恶意工具描述塞进任意一个 server,MCPTox 测得在 auto-approval 下 **84.2%** 的概率直接得手。换算成期望:攻击者随机命中一个暴露 server 投毒,$0.842$ 的成功率意味着平均试 $1/0.842\approx1.19$ 次就中——攻击面广(2570)× 单点命中率高(84.2%),这才是 MCP 在 2026 成为最热攻击面的真实账。
- **PoC 实证**:BlueRock 在**微软 MarkItDown MCP server** 上证实,因不校验 URL,把元数据 IP `169.254.169.254` 喂进去并追加 `/latest/meta-data/iam/security-credentials`,即可拿到实例角色 → access key / secret key / session token,再用 AWS CLI 接管账号。任何运行在 EC2、接受 URL 参数、且允许查询 IMDSv1 的 MCP server 都可能有同类潜在风险。
- **OWASP MCP Top 10** 确为 OWASP 首个 MCP 专项 Top 10,条目 **MCP01–MCP10(MCP01:2025–MCP10:2025)**,2026 仍处 **beta/草案**,负责人 Vandana Verma Sehgal。来源:[OWASP MCP Top 10](https://owasp.org/www-project-mcp-top-10/)。代表条目:MCP01 令牌管理不当与凭证暴露、MCP03 工具投毒、MCP04 软件供应链与依赖篡改、MCP05 命令注入、MCP09 影子 MCP server。
- 旁证:Trend Micro 发现 492 个零认证、互联网暴露的 MCP server;2026 年 1–2 月披露 30+ 个 MCP 生态 CVE。来源:[Trend Micro — Exposed MCP Servers](https://www.trendmicro.com/vinfo/us/security/news/vulnerabilities-and-exploits/update-on-exposed-mcp-servers-the-threat-widens-to-the-cloud)。

## 防御
把每个 MCP server 当作**不可信的外部依赖**对待:
- **来源校验 + pin 版本**:只接入可信来源,用签名/哈希锁定版本,防 rug-pull(对应 MCP04)。
- **沙箱 + 最小权限**:限制 server 的网络出口与可达资源;**强制 IMDSv2**、屏蔽链路本地与元数据网段 `169.254.0.0/16`,堵死 SSRF 取凭证(链 [[24 沙箱、最小权限与人审闸门|沙箱、最小权限]])。
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
- **强制 IMDSv2**:IMDSv1 无需 token 即可读元数据,是 SSRF 取 IAM 凭证的根因;IMDSv2 要求带会话 token 的 PUT 预请求 + 限制 hop-limit,SSRF 难以拿到。
- **网络出口管控**:在 server 沙箱里**屏蔽链路本地与元数据网段 `169.254.0.0/16`**(含 `169.254.169.254`),egress allowlist 只放行业务必需域名。
- **URL 校验**:接受 URL 参数的工具做 SSRF 防护(解析后校验目标 IP 不落私网/元数据段,防 DNS rebinding)。

**3. 供应链锁定**
- **来源校验 + pin 版本**:只接可信来源,用签名/哈希锁版本,防 rug-pull(MCP04);MCP server 也纳入 SBOM 与依赖扫描。
- **禁用 shadow MCP server(MCP09)**:盘点并阻断未审批、影子接入的 server——这是 agent 版的「影子 IT」。
- **凭证治理(MCP01)**:server 的 token/key 走 secret manager,绝不进 prompt、不落日志;按 [[16 Agent 身份与权限滥用(非人类身份 NHI)|短时凭证]] 发放。

**4. 检测响应**
- 网关全量记录 MCP 调用(server / tool / 参数 / 结果哈希),异常基线告警(某 server 突然要读 `~/.ssh`、访问元数据 IP)。
- 扫描赛道厂商:**Invariant Labs(mcp-scan)、Snyk Labs、Akto、Lasso**;Trend Micro / BlueRock 持续披露暴露面与 CVE。

## 面试高频

**Q1:MCP 的三处攻击面是什么?各自怎么防?**
标准答:① **工具投毒(MCP03)**——隐藏指令藏在工具描述里,agent 一连上就被注入,**工具无需被真正调用**(MCPTox 在 auto-approval 下成功率 84.2%);防:扫描描述、人审挂载。② **rug-pull**——先良性获信任、事后改恶意;防:pin 版本 + 来源校验。③ **间接提示注入(结果回流)**——恶意指令藏在工具返回结果里随上下文回流;防:结果当不可信输入过 guardrail。外加 **SSRF→云元数据**穿透宿主(MCP04)。
- 陷阱:工具投毒最反直觉的点是「**不调用也中招**」——只要连上 server,描述就进了 context window。

**Q2:讲一个 MCP SSRF 取 IAM 凭证的完整链路。**
标准答:server 跑在 EC2、接受 URL 参数、且未校验 URL → 攻击者喂入 `http://169.254.169.254/latest/meta-data/iam/security-credentials/<role>` → 拿到实例角色的临时 access key/secret/session token → 用 AWS CLI 接管账号。BlueRock 在**微软 MarkItDown MCP server** 上做了 PoC 实证;分析 7000+ server 约 **36.7% 疑似 SSRF**。堵法:强制 IMDSv2 + 屏蔽 `169.254.0.0/16` + URL 校验 + 最小权限实例角色。
- 追问「为什么 IMDSv2 能挡」:它要求带 token 的 PUT 预请求且限制 hop-limit,简单 SSRF(只能发 GET)拿不到 token 就读不了元数据。

**Q3:为什么说 MCP 安全是「运行时供应链」,和静态供应链什么关系?**
标准答:[[10 供应链安全(静态)|静态供应链]]管模型/数据/包(交付前),MCP 安全管**运行时即插即用的工具协议**(运行中动态接入第三方 server)。二者互补:静态管「装进来的东西干不干净」,MCP 管「运行时连上的工具能不能信、会不会事后变坏(rug-pull)」。对应 OWASP **ASI04 不安全的 agent 供应链**。

**Q4(追问)为什么不彻底?**
标准答:① 工具描述是自然语言、要喂给模型才有用,扫描器与对抗改写永远在拉锯;② rug-pull 是「事后变恶意」,首次审核通过不代表永远安全,只能靠 pin + 持续监控缩短暴露;③ 结果回流注入本质是 [[05 Prompt Injection 提示注入|提示注入]],而提示注入至今无完美解。所以 MCP 安全靠网关 + 沙箱 + 最小权限 + 监控的纵深,而非单点。

## 知识拓展
- **框架定位**:OWASP **MCP Top 10(MCP01–MCP10:2025,2026 仍 beta/草案**,负责人 Vandana Verma Sehgal)——OWASP 首个 MCP 专项 Top 10;上层对齐 **ASI04 不安全 agent 供应链**(OWASP Top 10 for Agentic Applications,2025-12)。代表条目:MCP01 令牌/凭证暴露、MCP03 工具投毒、MCP04 软件供应链/依赖篡改、MCP05 命令注入、MCP09 影子 MCP server。
- **基准与数据**:**MCPTox**(工具投毒基准,auto-approval 下成功率 84.2%);**BlueRock** 7000+ server 约 36.7% 疑似 SSRF + MarkItDown PoC;**Trend Micro** 发现 492 个零认证、互联网暴露的 server,2026 年 1–2 月披露 30+ MCP 生态 CVE。
- **工具生态**:Invariant Labs `mcp-scan`(静态扫描 + 运行时 proxy guardrail)、Snyk Labs、Akto、Lasso 等 MCP 安全扫描/网关赛道。
- **对抗演化**:攻击从「直连 server 投毒」→「rug-pull 事后变恶意」→「结果回流间接注入」→「SSRF 穿透宿主取云凭证」;防御从「人工审 server」→「自动化扫描描述」→「网关统一管控 + 运行时拦截」→「把 MCP server 纳入 SBOM/零信任/最小权限的标准供应链治理」。

## 相邻
- 上游:[[18 Agent 间通信安全与级联失败|级联失败]]、[[19 人机信任利用与 Rogue Agents|流氓 agent]]
- 同源:[[10 供应链安全(静态)|静态供应链]]、[[05 Prompt Injection 提示注入|提示注入]]
- 防御侧:[[21 Guardrails 与输入输出防护|Guardrails]]、[[24 沙箱、最小权限与人审闸门|沙箱]]
- 跨域:[[17 MCP 模型上下文协议|MCP 协议]]
- 框架定位:[[26 安全框架与治理地图|安全框架地图]]、[[01 AI 安全总览与三层栈|三层栈]]
