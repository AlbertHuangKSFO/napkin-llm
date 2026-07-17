[[17 MCP 模型上下文协议|MCP 模型上下文协议]]（Model Context Protocol）是 host／client 与 server 间交换上下文能力的开放协议：server 可以暴露 Tools、Resources、Prompts，client 在初始化后发现并消费它们。MCP 解决的是「能力如何标准暴露和接入」；[[15 Function Calling 工具调用|Function Calling]] 解决的是模型如何请求一次具体工具。

## 直觉／生活类比

MCP 像通用插座标准，而不是替你决定谁有钥匙的门禁系统。插座让 IDE、桌面应用或 agent 能以同一方式接文件、数据库和 SaaS；但接上插座不自动获得财务系统的权限。远端公开 server 必须仍在服务端验证来访者，决定每个 `tools/call` 是否可执行。

MCP server 身份、工具定义与依赖来源的供应链风险，见 [[AI 安全/20 Agentic 供应链与 MCP 安全]]。

三类原语可按控制权理解：Tool 是可调用动作，Resource 是可读上下文，Prompt 是可复用提示模板。协议允许 host 选择怎样呈现它们；不要把「列出了一个 Tool」误读为「任何模型、任何用户都获准调用」。 [MCP Tools 2025-11-25（检索于 2026-07-17）](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

## 小数字手算

若有 $N=3$ 个 host 应用、$M=4$ 个后端能力：

$$
N\times M=3\times4=12
$$

表示两两定制时的适配关系数。双方都实现同一个协议后，粗略变成：

$$
N+M=3+4=7,\qquad 12-7=5
$$

这只是集成接口的规模模型；认证、数据治理和各后端的业务逻辑仍不能「自动相加消失」。

## 公式推导

一次远端工具执行的安全条件不是「MCP 已连接」，而是：

$$
\operatorname{Allow}(r)=T(r)\land I(r)\land P(r)\land R(r)\land B(r)
$$

其中 $T$ 是受保护传输，$I$ 是 token／会话真实性，$P$ 是 scope 或权限，$R$ 是 token 的目标资源／audience，$B$ 是业务级资源归属和状态检查。MCP 的 HTTP 授权规范可帮助实现其中一部分；它不能替代 $B$，也不能让 model 生成的参数成为可信身份。

## 手绘图

![[MCP 模型上下文协议-NxM 集成.png]]

![[MCP 模型上下文协议.png]]

## 可运行代码：❌ 把可发现当可访问 vs ✅ policy 与本地 MCP server 分层

```python
# ❌ 能运行，但“能发现 server”就默认放行，公开远端部署不可接受。
def naive_allow(_headers: dict, _tool_name: str) -> bool:
    return True

print(naive_allow({}, "refund_invoice"))  # True：没有身份、scope、audience 或业务校验。
```

```python
# ✅ 纯标准库，可直接运行：这是 HTTP 网关／server 执行前的策略边界示例，
# 不是 OAuth 或 MCP transport 的替代实现。
from dataclasses import dataclass

@dataclass(frozen=True)
class Principal:
    subject: str
    scopes: frozenset[str]
    audience: str

TOKENS = {
    "demo-token": Principal("user-1", frozenset({"orders:read"}), "https://orders.example/mcp"),
}

def authorize(headers: dict[str, str], required_scope: str, expected_resource: str) -> Principal:
    raw = headers.get("Authorization", "")
    if not raw.startswith("Bearer "):
        raise PermissionError("missing bearer token")
    principal = TOKENS.get(raw.removeprefix("Bearer "))
    if principal is None:
        raise PermissionError("unknown token")
    if principal.audience != expected_resource:
        raise PermissionError("token audience is not this MCP server")
    if required_scope not in principal.scopes:
        raise PermissionError("insufficient scope")
    return principal

principal = authorize(
    {"Authorization": "Bearer demo-token"}, "orders:read", "https://orders.example/mcp"
)
assert principal.subject == "user-1"
try:
    authorize({"Authorization": "Bearer demo-token"}, "orders:write", "https://orders.example/mcp")
except PermissionError as exc:
    print(f"blocked: {exc}")
```

当前 MCP Python SDK 的 stable v1 示例使用 `FastMCP`；为避免把尚在演进的主线 API 当成生产接口，下面固定到 v1：`pip install "mcp[cli]>=1.27,<2"`。 [MCP Python SDK v1 README（检索于 2026-07-17）](https://github.com/modelcontextprotocol/python-sdk/tree/v1.x)

```python
# 一个完整的本地 stdio MCP server；运行前：pip install "mcp[cli]>=1.27,<2"
# 本地 stdio 凭证通常来自受控环境；上面的远端策略应置于 HTTP 路径。
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("orders")

@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """查询订单状态。仅返回演示数据；实际部署必须在 transport 层取得可信主体后再做资源授权。"""
    if order_id != "ORD-7":
        return {"ok": False, "error": "订单不存在"}
    return {"ok": True, "order_id": "ORD-7", "status": "shipped"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

授权边界要分清：MCP 协议的授权功能是**可选**的；这不等于公开远端 server 可以无授权运行。本笔记的生产基线是：公开 HTTP server 在执行任何 `tools/call` 前必须完成身份与权限验证。若选择实现 MCP 的 HTTP 授权规范，client/server 应按规范使用 OAuth、PKCE、Protected Resource Metadata，并以 RFC 8707 `resource` 参数将 token 绑定到目标 server；server 必须验证 token 的目标资源。stdio 则不走这一 HTTP 授权流程，应从受控环境取得凭证。 [MCP Authorization 2025-06-18（检索于 2026-07-17）](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：MCP 与 Function Calling 是一回事吗？** 不是。MCP 定义 server 如何发现、列举与调用能力；Function Calling 定义模型一次如何产生调用请求、harness 如何回灌结果。没有 MCP 也可使用 function calling。

**Q：MCP 的发现流程？** client 初始化并协商能力；对 tools 发 `tools/list`（可分页），随后以 `tools/call` 执行。server 可用 `notifications/tools/list_changed` 提醒工具清单变化。发现不是授权。 [MCP Tools 2025-11-25（检索于 2026-07-17）](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

**Q：远端 MCP 一定要 OAuth 吗？** 协议授权是 optional；但公开生产 server 应有服务端访问控制。实现 MCP HTTP 授权 profile 时才遵循相应 OAuth／PKCE／resource-bound 规则，且仍需业务授权。

## 关键事实

- MCP 标准 transport 是 stdio 和 Streamable HTTP，消息以 JSON-RPC 编码；Streamable HTTP 取代了早期的 HTTP+SSE transport。 [MCP Transports 2025-11-25（检索于 2026-07-17）](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
- `tools/list` 是协议级能力发现，支持分页；`tools/call` 才是具体执行请求。工具名、输入 schema、可选输出 schema 都由 server 描述。 [MCP Tools 2025-11-25（检索于 2026-07-17）](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- MCP 的授权机制是 optional；在支持 HTTP 授权时，规范定义 MCP server 为 OAuth 资源服务器、client 使用 Protected Resource Metadata 和 Resource Indicators，且 client 必须实现 PKCE。 [MCP Authorization 2025-06-18（检索于 2026-07-17）](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
- Resource-bound token 不是业务授权的替身：server 仍须验证 token 面向自己，并在每次工具执行前按主体、scope、目标资源和业务状态做决定。 [MCP Authorization 2025-06-18（检索于 2026-07-17）](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)
- Tools、Resources、Prompts 是 MCP server 向 host 提供的能力类别；client 对不可信 server 给出的 annotation 不能作安全信任依据。 [MCP Server Overview 2025-06-18（检索于 2026-07-17）](https://modelcontextprotocol.io/specification/2025-06-18/server/index)
