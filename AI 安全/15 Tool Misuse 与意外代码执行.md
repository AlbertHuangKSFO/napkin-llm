[[15 Tool Misuse 与意外代码执行|工具滥用]]是 [[14 Excessive Agency 与 Goal Hijack|过度代理]] 落地的具体形态:工具是 agent 伸向真实世界的手,一旦被滥用,合法授权就被掰成攻击动作。它对应 OWASP ASI02 工具滥用 + ASI05 意外代码执行——同一次工具调用,既能是正常 [[15 Function Calling 工具调用|Function Calling]],也能是 RCE、SSRF 或 confused deputy。

## 三条滥用路径
![[sec-工具滥用与RCE.png]]

- **code-exec 工具 → RCE**:工具能跑 `eval()`/`exec()`/shell,模型生成的代码直接执行。被注入/被诱导后,任意命令落地——这正是 ASI05 意外代码执行(agent 生成、修改或运行代码/命令,产生安全或运维风险)。
- **fetch 工具 → SSRF**:URL 由模型或外部内容控制,agent 被引去打内网或云元数据端点(如 `169.254.169.254`)。SSRF 本质就是 confused deputy 的经典形态——应用被当成代理,替攻击者做它自身无权做的事。
- **confused deputy(混淆代理)**:低权方诱使高权 agent 替它越权动作。权限随对象(URL、文件句柄、任务)在程序间传递而**悄悄漂移**,双方都没显式改授权,却造成越权(CWE-441)。

## 脆弱代码 vs 缓解代码

脆弱:无沙箱直接 exec,URL 不校验——

```python
# ✗ 危险:模型输出直接进 exec / 无出口控制
def run_code(code: str):
    return exec(code)              # ASI05:任意 RCE

def fetch(url: str):
    return requests.get(url).text  # ASI02:任意 SSRF,可打内网/元数据
```

缓解:沙箱执行 + 出口白名单 + 参数校验——

```python
# ✓ 隔离 + 收窄出口 + 锁定身份
def run_code(code: str):
    # 在无网络、只读 FS 的容器/微 VM 里跑,资源与时间配额
    return sandbox.execute(code, network=False, fs="readonly",
                           cpu_quota="500m", timeout=5)

ALLOW_HOSTS = {"api.internal.example.com"}
DENY_NETS   = ["169.254.0.0/16", "10.0.0.0/8", "127.0.0.0/8"]

def fetch(url: str):
    host = urlparse(url).hostname
    ip = resolve(host)                          # 解析后再判,防 DNS rebinding
    if host not in ALLOW_HOSTS or in_nets(ip, DENY_NETS):
        raise PermissionError("blocked egress")
    # 工具按调用者身份执行,不复用 agent 的高权 → 堵 confused deputy
    return http_get(url, identity=caller_identity, allow_redirects=False)
```

## 防御
- **沙箱执行**:容器 / gVisor / 微 VM,无网络、只读 FS、资源配额,堵 RCE 横向移动。详见 [[24 沙箱、最小权限与人审闸门|沙箱与最小权限]]。
- **出口白名单**:域名/IP 白名单 + 禁内网网段与元数据端点,解析后再判防 DNS rebinding,堵 SSRF。
- **参数校验 + 用户身份透传**:工具按真实调用者权限执行,不复用 agent 高权,堵 confused deputy。
- **工具设计层面收窄**:见 [[16 工具设计与工具层|工具设计]]——窄接口、强类型参数、最小副作用,从源头降低可滥用面。
- 输出再用前清洗:见 [[08 不安全输出处理|不安全输出处理]]。

## 关键事实(含出处)
- OWASP **ASI02 Tool Misuse and Exploitation**:agent 以不安全方式使用工具,或攻击者利用工具接口取得访问/造成危害;**ASI05 Unexpected Code Execution**:agent 生成、修改或运行代码/命令产生安全或运维风险(OWASP Top 10 for Agentic Applications 2026, genai.owasp.org)。
- **Confused deputy(CWE-441,Unintended Proxy / Confused Deputy)**:低权程序诱使高权程序滥用其权限;**SSRF 本质即一种 confused deputy 攻击**——应用被当作代理替攻击者发起其本身无权的请求(en.wikipedia.org/wiki/Confused_deputy_problem;cwe.mitre.org)。

## 工业界实践
工具滥用在工业界的反直觉点:**这些大多是经典 AppSec 漏洞(RCE/SSRF/路径穿越/命令注入)长在 AI 基础设施上**——不是什么新攻击,而是 MCP/工具层把老漏洞重新暴露了一遍,且现在由 LLM 的不可信输出来触发。

**真实案例(2025-2026,密集爆发)**:
- **MCP 供应链 RCE 链**:Anthropic 官方 **mcp-server-git** 三连漏洞——CVE-2025-68145(逃逸配置仓库路径)、CVE-2025-68143(把任意目录变 git 仓库)、CVE-2025-68144(用户可控参数传给 GitPython → 任意文件写),链上 Filesystem MCP 后经恶意 `.git/config` 达成 **RCE**。
- **CVE-2026-27825(mcp-atlassian,严重)**:Confluence 附件下载工具缺少目录限定 + 路径穿越校验不足 → 远程未认证写任意路径 → 提权/RCE;同时中间件无校验地信任 `X-Atlassian-*` 头 → **SSRF** 到任意目的地。
- **CurXecute / MCP STDIO 命令注入**:Cursor、Claude Code、Windsurf、Gemini-CLI、GitHub Copilot 等可经 MCP JSON 配置被命令注入——**用户 prompt 直接影响 MCP 配置且无需交互**(公开 prompt 变本地 shell)。
- 体量:Vulnerable MCP Project 截至 2026-04 追踪 50+ 已知 MCP 漏洞(13 个严重);研究发现 **82% 的 2,614 个 MCP 实现用了易路径穿越的文件操作**——路径穿越是 MCP 最常见单一漏洞。

**纵深防御(三条滥用路径逐条堵)**:
- **code-exec → RCE**:**沙箱执行**——容器 / gVisor / 微 VM(Firecracker / E2B / Modal),无网络、只读 FS、CPU/内存/时间配额,堵 RCE 横向移动。永远不要在主进程 `eval`/`exec`/`subprocess(shell=True)` 跑模型输出。
- **fetch → SSRF**:**出口白名单**——域名/IP 白名单 + 禁内网网段与云元数据端点(`169.254.169.254`、`fd00::/8` 等);**DNS 解析后再判 IP**(防 DNS rebinding);禁跟随重定向(防 302 跳内网)。
- **confused deputy**:**用户身份透传**——工具按真实调用者权限执行,不复用 agent 高权;权限不随对象(URL/句柄/任务)在程序间漂移(CWE-441)。
- **工具设计收窄**:窄接口、强类型参数、最小副作用——从源头降低可滥用面。
- **MCP 供应链**:对 MCP server 做 AppSec(SAST/依赖扫描/签名校验),pin 版本,审计第三方 server,见 [[20 Agentic 供应链与 MCP 安全|MCP 安全]]。

**检测响应与误报权衡**:出口白名单越严越安全但会**误杀正常外联**(合法 API、CDN),工业界做法是默认拒绝 + 显式 allowlist + 人工加白队列;沙箱有冷启动延迟(微 VM 数十至数百 ms),用预热池缓解。监控侧对"工具尝试访问内网/元数据端点""code-exec 触发网络系统调用""路径含 `../`"打高优告警,接入 [[25 监控、可观测与事件响应|事件响应]]。

```python
# SSRF 防御要点:解析后判 IP + 禁重定向 + 身份透传(对照本篇缓解代码补强)
import ipaddress, socket
from urllib.parse import urlparse
DENY_NETS = [ipaddress.ip_network(n) for n in
             ["169.254.0.0/16","10.0.0.0/8","172.16.0.0/12","192.168.0.0/16","127.0.0.0/8","::1/128","fd00::/8"]]
def safe_fetch(url, caller_identity):
    host = urlparse(url).hostname
    if host not in ALLOW_HOSTS:                       # 默认拒绝
        raise PermissionError("host not allowlisted")
    for fam, *_ , sa in socket.getaddrinfo(host, None):   # 解析所有 A/AAAA 记录
        ip = ipaddress.ip_address(sa[0])
        if any(ip in net for net in DENY_NETS):       # 任一解析结果命中内网即拒(防 rebinding)
            raise PermissionError("blocked egress to internal IP")
    return http_get(url, identity=caller_identity, allow_redirects=False)  # 禁重定向 + 身份透传
```

## 面试高频
**Q1:"agent 的 code-exec 工具有什么风险?怎么安全地让 LLM 跑代码?"** 风险是 **RCE**——模型生成的代码若直接进 `eval`/`exec`/shell,被注入后任意命令落地(ASI05 意外代码执行)。安全做法:在**无网络、只读 FS、资源/时间配额的沙箱**(容器/gVisor/微 VM)里跑,绝不在主进程执行。
- 陷阱:面试官问"我做了输入过滤(禁 `import os`)够吗?"→ **不够**,黑名单过滤几乎总能绕过(`__import__`、`getattr`、字节码、编码混淆),正道是隔离执行环境,不是过滤代码字符串。

**Q2:"什么是 SSRF?为什么说它是 confused deputy?agent 里怎么防?"** SSRF = 攻击者控制 URL 让服务端替它发请求,打内网或云元数据端点(`169.254.169.254` 拿临时凭证)。它是 **confused deputy** 的经典形态——应用被当代理替攻击者做它自身无权做的事。防御:出口白名单 + 禁内网网段/元数据端点 + **解析后判 IP**(防 DNS rebinding)+ 禁重定向。
- 追问"为什么要解析后再判,不能直接看域名?"→ DNS rebinding:域名首次解析到公网通过校验,实际请求时再解析到内网;必须对实际连接的 IP 判定。

**Q3:"confused deputy 的根因是什么?光做沙箱够吗?"** 根因是**权限随对象在程序间传递时悄悄漂移**(CWE-441),双方都没显式改授权却造成越权;低权方诱使高权 agent 替它越权。沙箱堵的是 RCE,不堵 confused deputy——后者要靠**身份透传**(工具按真实调用者权限执行,不复用 agent 高权),这是身份层问题,见 [[16 Agent 身份与权限滥用(非人类身份 NHI)|NHI 身份]]。

## 知识拓展
- **框架对照**:OWASP **Top 10 for Agentic Applications 2026** 的 **ASI02 Tool Misuse and Exploitation** + **ASI05 Unexpected Code Execution**。底层 CWE:**CWE-441**(Confused Deputy / Unintended Proxy)、**CWE-918**(SSRF)、**CWE-78**(OS 命令注入)、**CWE-22**(路径穿越)、**CWE-94**(代码注入)。MITRE ATLAS 有 agent 工具滥用 / 代码执行相关技术。
- **真实 CVE**:mcp-server-git 链(CVE-2025-68143/68144/68145)、mcp-atlassian(CVE-2026-27825,RCE+SSRF)、CurXecute(Cursor MCP 自启 RCE)、MCP STDIO 命令注入。数据库:Vulnerable MCP Project、Endor Labs / Ox Security / Imperva 的 MCP AppSec 研究。
- **沙箱/隔离技术**:gVisor、Firecracker 微 VM、E2B、Modal、Daytona——LLM code-exec 的隔离基座。
- **对抗演化**:从"直接 eval 模型输出"→"过滤被绕过"→"经检索/MCP 间接注入触发工具"→"MCP 供应链 RCE(老 AppSec 漏洞 + AI 触发面)",防御从代码过滤转向隔离执行 + 出口控制 + 身份透传 + 对工具供应链做 AppSec。

## 兄弟链
- [[14 Excessive Agency 与 Goal Hijack|过度代理]] — 工具滥用的上游「为什么有权滥用」
- [[16 Agent 身份与权限滥用(非人类身份 NHI)|Agent 身份]] — confused deputy 的身份层根因与解法
- [[20 Agentic 供应链与 MCP 安全|MCP 安全]] — 工具来源侧的供应链风险
- [[24 沙箱、最小权限与人审闸门|沙箱与人审]] — 缓解集中篇
