[[15 Function Calling 工具调用|Function Calling 工具调用]] 是把模型的「下一步意图」编码为带参数的函数调用；**模型请求，runtime 决定是否执行**，并把结果再交回模型继续推理。它让 [[03 Agent 核心循环|Agent 核心循环]] 能接触外部数据和动作，但不把权限交给模型。

## 直觉／生活类比

把模型看成餐厅的点单员：它能把「客人要取消 ORD-7」填进标准点菜单，却不能自己进后厨退款。runtime 才是后厨门禁：它用登录态确认是谁、检查订单归属和确认状态、记录审计日志，然后才执行或返回可纠正的错误。

`strict: true` 只保证点菜单的**栏位与格式**符合 schema；它不会证明点单员认识客人，也不会让「取消订单」在业务上合法。语义授权、限额、幂等和人工确认必须在 runtime 重做。

对高风险调用的最小权限、沙箱隔离与人工确认，可继续参照 [[AI 安全/24 沙箱、最小权限与人审闸门]]。

## 小数字手算

同一轮有两个互不依赖的读取调用，分别耗时 $800\text{ ms}$ 与 $500\text{ ms}$。

- 串行：$800+500=1300\text{ ms}$。
- 并行：$\max(800,500)=800\text{ ms}$。
- 节省：$1300-800=500\text{ ms}$，相对串行 $500/1300\approx38.5\%$。

这是没有排队、网络拥塞和依赖关系时的上界；若第二个参数依赖第一个结果，就必须分两轮，不能为了省时而猜参数。

## 公式推导

把一次调用能否被接受分成三层：格式、身份／权限、业务状态。

$$
\begin{aligned}
\operatorname{Execute}(c) &= S(c) \land A(c,\text{session}) \land B(c,\text{state}) \\
S(c) &= \text{参数是否符合 JSON Schema} \\
A(c,\text{session}) &= \text{会话主体是否有权对目标操作} \\
B(c,\text{state}) &= \text{订单是否存在、可取消、已确认且未重复执行}
\end{aligned}
$$

`strict: true` 针对的是 $S(c)$：OpenAI 文档说明它让 function call 可靠遵循函数 schema，并要求各 object 禁止额外字段、所有 properties 都列入 `required`（可用 `null` 表示可选）。它**不蕴含** $A$ 或 $B$；即使 $S(c)=1$，陌生用户仍可能给出格式完全正确的 `order_id`。 [OpenAI Function calling（检索于 2026-07-17）](https://developers.openai.com/api/docs/guides/function-calling)

## 手绘图

![[Function Calling 工具调用.png]]

![[Function Calling 工具调用-并行调用.png]]

## 可运行代码：❌ 把 schema 当授权 vs ✅ Responses 回灌循环

先安装并配置真实示例所需依赖：`pip install openai`，再设置 `OPENAI_API_KEY`。下列改善版采用当前 Responses 形态：从 `response.output` 读取 `function_call`，把执行结果以同一 `call_id` 的 `function_call_output` 加回输入；这是 API 文档的调用／回灌约定，而不是 SDK 的猜测。 [OpenAI Function calling（检索于 2026-07-17）](https://developers.openai.com/api/docs/guides/function-calling)

```python
# ❌ 能运行，但错误地把“字段合法”当成“允许取消”
def naive_cancel(arguments: dict) -> dict:
    if set(arguments) == {"order_id"} and isinstance(arguments["order_id"], str):
        return {"ok": True, "cancelled": arguments["order_id"]}
    return {"ok": False, "error": "bad shape"}

print(naive_cancel({"order_id": "ORD-7"}))
# 任何人只要填对形状都“成功”；这里完全没有用户、归属或业务状态。
```

```python
# ✅ 可直接运行；需要 pip install openai 和 OPENAI_API_KEY。
from __future__ import annotations

import json
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-5.6"  # 以运行时可用模型替换；API 字段见文末来源。

TOOLS = [{
    "type": "function",
    "name": "cancel_order",
    "description": "取消指定订单。仅当用户明确要求取消订单时调用；runtime 会独立检查身份、归属、确认和订单状态。",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号，例如 ORD-7"},
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
}]

ORDERS = {"ORD-7": {"owner": "user-1", "status": "paid"}}

def cancel_order(arguments: dict, session: dict) -> dict:
    # schema 合规仍要在 runtime 防御性校验；身份来自会话，不来自模型参数。
    if set(arguments) != {"order_id"} or not isinstance(arguments["order_id"], str):
        return {"ok": False, "error": "order_id 必须是唯一的字符串字段"}
    order = ORDERS.get(arguments["order_id"])
    if order is None:
        return {"ok": False, "error": "订单不存在；请核对 order_id"}
    if order["owner"] != session["user_id"]:
        return {"ok": False, "error": "无权取消该订单"}
    if not session["confirmed"]:
        return {"ok": False, "error": "需要用户确认后才能取消"}
    if order["status"] != "paid":
        return {"ok": False, "error": f"当前状态为 {order['status']}，不可取消"}
    order["status"] = "cancelled"  # 真实系统还应使用事务与幂等键。
    return {"ok": True, "order_id": arguments["order_id"], "status": "cancelled"}

def dispatch(name: str, arguments: dict, session: dict) -> dict:
    if name != "cancel_order":
        return {"ok": False, "error": f"未知或未允许的工具：{name}"}
    return cancel_order(arguments, session)

def run_agent(user_text: str, session: dict, max_steps: int = 4) -> str:
    history: list[object] = [{"role": "user", "content": user_text}]
    for _ in range(max_steps):
        response = client.responses.create(model=MODEL, input=history, tools=TOOLS)
        history.extend(response.output)  # 保留模型先前的 function_call 项。
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text

        for call in calls:
            try:
                arguments = json.loads(call.arguments)
                result = dispatch(call.name, arguments, session)
            except (json.JSONDecodeError, TypeError) as exc:
                result = {"ok": False, "error": f"参数无法解析：{exc}"}
            history.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            })
    raise RuntimeError("工具调用超过 max_steps，已停止以避免无限循环")

if __name__ == "__main__":
    print(run_agent("请取消我的订单 ORD-7", {"user_id": "user-1", "confirmed": True}))
```

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：Function Calling 中模型到底执行了什么？** 它产出工具名和 JSON 编码参数的调用意图；应用解析、验证、执行，并把带匹配 call ID 的结果回灌。模型并不持有执行权。

**Q：`strict: true` 能防越权吗？** 不能。它解决 $S(c)$ 的 schema adherence；身份、资源归属、状态转换、限额与确认属于 $A$、$B$，必须由 runtime 执行。

**Q：什么时候可以并行？** 只有读取或其他无依赖调用可以并行。参数依赖前一步结果、同一资源的写入，或业务顺序有要求时必须串行，并给每次写入幂等键。

## 关键事实

- 工具调用是模型对外部能力的请求；工具输出可以是 JSON 或文本，并应关联具体调用的 `call_id`。 [OpenAI Function calling（检索于 2026-07-17）](https://developers.openai.com/api/docs/guides/function-calling)
- 当前 Responses 风格把调用放在 `response.output` 的 `function_call` 项中；应用回传 `function_call_output`，其 `call_id` 必须对应调用。 [OpenAI Function calling（检索于 2026-07-17）](https://developers.openai.com/api/docs/guides/function-calling)
- `strict: true` 是 API 的 schema adherence 特性；它要求严格 schema 约束，并不携带用户身份或业务授权语义。 [OpenAI Function calling（检索于 2026-07-17）](https://developers.openai.com/api/docs/guides/function-calling)
- 一次响应可能含多个 function call；是否并行是 harness 的依赖分析与执行策略，API 也允许以 `parallel_tool_calls: false` 限制为至多一个调用。 [OpenAI Function calling（检索于 2026-07-17）](https://developers.openai.com/api/docs/guides/function-calling)
