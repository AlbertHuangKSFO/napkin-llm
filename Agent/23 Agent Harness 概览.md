[[23 Agent Harness 概览|Agent Harness]] 的本质：模型本身只生成下一条消息；**harness（或 scaffold）**把输入、上下文、工具调用、权限、重试、状态记录和结果交付组织成一个可运行的 agent 系统。

## 直觉：驾驶员、行车记录仪与受控试验场

把模型想成驾驶员：它决定下一步方向；harness 是调度与交通规则，负责把方向转换为受限的工具调用；session 是行车记录仪，保留发生过什么；sandbox 是隔离试验场，限定代码和文件操作的爆炸半径。把三者混为「一个容器」会让故障恢复、审计和权限边界都变得含糊。

更精确地说：

- **session**：可持久化的事件记录，如消息、工具调用、产物引用、检查结果；不是模型上下文窗口本身。
- **harness**：agent loop 与编排层，负责调用模型、路由工具、执行策略和写事件。
- **sandbox**：实际执行命令、编辑文件或运行浏览器的受限环境；它可以被当作一种工具，而非 agent 的全部状态。

这三个可替换边界来自 [Anthropic Managed Agents 的设计说明（2026-04-08）](https://www.anthropic.com/engineering/managed-agents)。该文把托管的长程 agent runtime 定义为稳定接口上的受管服务；不要把某个产品宣传名当作行业标准术语或必需层级。自建 agent SDK、开源框架和受管 runtime 都可以实现 harness。

## 小数字手算：可靠性来自可恢复状态

设一次任务有 4 次模型决策，每次工具调用耗时 $(3,5,4,2)$ 秒，harness 的调度开销每次 $0.5$ 秒。忽略模型推理时间时：

$$
T_{\text{run}}=\sum_i T_{\text{tool},i}+4\times0.5
=3+5+4+2+2=16\text{ s}
$$

若第三次工具失败，而 session 在第二次后已经持久化，则恢复只需重放/继续余下的 $(4+2)+2\times0.5=7$ 秒工作量，而非重新执行 16 秒。真实恢复时间还受副作用是否幂等、sandbox 是否可重建、外部工具是否可重试影响；这个计算只说明为什么事件边界有价值。

## 推导：把 agent loop 写成可观测状态机

令 $s_t$ 是 session 中已持久化事件，$c_t$ 是被挑选进入模型的上下文。harness 每轮做：

$$
c_t=\operatorname{select}(s_t),\quad
a_t=\operatorname{model}(c_t),\quad
o_t=\operatorname{execute}_{\text{policy}}(a_t),\quad
s_{t+1}=\operatorname{append}(s_t,a_t,o_t)
$$

其中 `execute_policy` 必须在动作发生前验证：工具是否允许、参数是否合规、是否需要用户批准、是否超预算。只有把 $a_t$、$o_t$、错误和 artifact 引用持续写进 $s$，harness 崩溃后才可能由新的进程从已知状态恢复。

[Anthropic 对 agent/eval harness 的定义（2026-01-09）](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)很直接：agent harness 让模型作为 agent 运行，处理输入、编排工具调用并返回结果；评估的是「模型 + harness」的组合，不能把质量全部归因于模型。

![[Agent Harness 概览.png]]

![[Agent Harness 概览-结果分叉.png]]

## 可运行代码：把 session、harness、sandbox 分开

❌ 把事件日志和执行环境隐含在一个全局对象里；重启后既不知道做过什么，也无法区分失败发生在哪层。

```python
state = []
def run(action):
    state.append(action)
    return eval(action)  # 同时混淆执行、权限和审计；绝不可用于生产
```

✅ 此最小示例显式记录 session，harness 只路由允许动作，sandbox 负责受限执行。示例仅返回字符串，不执行任意 shell。

```python
from dataclasses import dataclass, field

@dataclass
class Session:
    events: list[dict] = field(default_factory=list)
    def emit(self, kind: str, value: str) -> None:
        self.events.append({"kind": kind, "value": value})

class Sandbox:
    def execute(self, command: str) -> str:
        if command != "healthcheck":
            raise PermissionError("command is not allowed")
        return "ok"

def harness_step(session: Session, sandbox: Sandbox, action: str) -> str:
    session.emit("requested", action)
    result = sandbox.execute(action)
    session.emit("result", result)
    return result

s = Session()
assert harness_step(s, Sandbox(), "healthcheck") == "ok"
assert [event["kind"] for event in s.events] == ["requested", "result"]
print(s.events)
```

生产版还应把 session 放到持久存储，给每次工具调用加 trace id、超时、幂等键、预算和最小权限策略；不能因为有 sandbox 就跳过审批和输出验证。

## 设计清单：SDK、受管 runtime 与应用 harness

- **agent SDK**：提供构建 agent loop 的代码原语；你拥有状态存储、部署和权限方案。
- **应用自建 harness**：为某个任务域定制工具、提示、artifact 和验收；简单优先，模型提升后定期删掉不再必要的脚手架。
- **managed runtime**：供应商托管长程执行、session 恢复或 sandbox 调度，并暴露稳定接口；它不等于模型，也不保证业务流程正确。

⚠️ **沙箱不是安全的同义词。** 它限制执行环境，却不能自动判断业务授权、提示注入、数据分级或外部 API 的破坏性。高风险工具仍需最小权限、参数校验、人审闸门和可撤销的操作设计，关联 [[16 工具设计与工具层|工具设计与工具层]] 与 [[38 Agent 评估与可观测性|评估与可观测性]]。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**问：harness 与模型、工具有什么边界？**

答：模型提出下一步；工具/沙箱执行受限动作；harness 维护循环、上下文、策略、记录和恢复。换模型或换工具不应迫使 session 格式一起变化。

**问：为什么 session 不能只保存在容器内存？**

答：容器失败会同时丢失恢复点和审计证据。把 append-only 事件记录放到容器外，harness 才能重启后从最后一个可信事件继续；有副作用的调用还需幂等键或补偿逻辑。

**问：何时选受管 runtime？**

答：需要长程执行、弹性 sandbox、故障恢复和运行运维能力，又接受供应商接口与数据边界时可考虑；若任务短、基础设施要求特殊或需完全控制执行面，自建 SDK/harness 往往更合适。先做威胁建模与成本测量，不按流行术语选型。

## 关键事实

- [Anthropic，2026-01-09](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)：将 agent harness/scaffold 定义为处理输入、编排工具调用、返回结果的系统，并指出评估对象是 harness 与模型的组合。
- [Anthropic，2026-04-08](https://www.anthropic.com/engineering/managed-agents)：把 session（append-only log）、harness（模型/工具循环）和 sandbox（执行环境）解耦为可替换接口；这是其 Managed Agents 的架构，不是所有产品都必须照搬。
- [Anthropic，2025-11-26](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)：长程编码 agent 需要清晰的进度 artifact 与端到端验证，单靠循环与 compaction 不足以保证交付质量。
