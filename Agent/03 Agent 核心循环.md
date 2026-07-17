[[03 Agent 核心循环|Agent 核心循环]] 是 harness 驱动的 `observe → decide → act → observe` 回路：模型提出下一步，运行时受控执行并回灌结果，直到返回一个可被上游处理的结构化终态。

它是 [[01 什么是 AI Agent|AI Agent]] 的工程骨架，也是 [[02 Workflow 与 Agent 的边界|Agent]] 一侧的运行机制。注意：**没有 tool call 不是成功的充分条件**。模型可能在等用户数据、等待批准、被工具阻塞，或只是没有按协议输出；只有 `completed` 且验收条件满足，才可报告任务完成。

## 本质：一个带状态机的 while 循环

去掉框架名，循环包含三类工作：

- **observe（观察）**：读取目标、历史、最新工具结果、剩余预算与权限状态。
- **decide（决策）**：模型选择工具调用、请求输入/批准，或提出终态与证据。
- **act（行动）**：[[23 Agent Harness 概览|harness]] 校验并执行工具，将结果截断、格式化为 observation。

一轮的状态更新是「模型给候选行动 → harness 检查 → 环境返回结果 → 写回状态」。单次模型调用只依据被提供的输入；会话连续性由 harness 或平台保存、压缩并重新提供状态。

![[Agent 核心循环.png]]

## 直觉：流水线上的受控回合

模型像提出下一张工单的调度员；harness 像值班工程师。调度员说「执行测试」并不直接触发命令：值班工程师要先核对工具、参数、权限与预算，再把日志摘要回传。若工单要求付款，值班工程师返回「等待批准」；若缺少收件人，返回「需要输入」。这些都是正确结束，而不是把「没再发工单」误判为成功。

## 小数字手算：预算如何产生终态

设本次运行 token 预算为 $B=3500$。前三次模型调用分别消耗 $900$、$1200$、$1600$ token：

$$
T_1=900,\qquad T_2=900+1200=2100,\qquad T_3=2100+1600=3700
$$

第 3 轮后 $T_3>B$，因此本轮应返回 `budget_exhausted`（`budget_type="tokens"`），而不是继续执行第 3 轮模型建议的工具。若另有 `max_steps=4`，第 4 个回合结束后仍未得到可验证终态，也应以 `budget_exhausted`（`budget_type="steps"`）结束。预算是终态协议的一部分，不只是日志指标。

## 公式推导：回路、约束与完成判定

令状态为 $s_i$，模型提出候选输出 $r_i$，策略为 $\pi$，harness 的约束/执行器为 $H$：

$$
r_i=\pi(s_i),\qquad (a_i,o_i)=H(r_i,s_i),\qquad s_{i+1}=U(s_i,a_i,o_i)
$$

当模型提出 `completed` 时，系统还需验证验收谓词 $V$：

$$
\text{成功} \iff \text{status}=\texttt{completed}\ \land\ V(\text{evidence},\text{goal})=\text{true}
$$

因此「无工具调用」至多表示 $r_i$ 没有要求行动，不能推出 $V=true$。例如修复代码时，`V` 可以检查测试输出、变更范围和人工评审要求。

## 机制：一轮里到底发生什么

1. **harness → 模型**：发送目标、相关历史、最新 observation、工具定义、预算和权限上下文。
2. **模型 → harness**：返回一个或多个结构化 tool call，或符合协议的终态/请求。
3. **harness 校验**：检查工具名、参数 schema、速率、策略和是否需要审批。
4. **harness → 工具**：执行允许的工具；失败也应产生可控的错误 observation。
5. **工具 → harness**：返回结果、错误或外部阻塞信号。
6. **harness 回灌**：对结果做脱敏、截断或摘要后更新状态，进入下一轮，或把终态交给上游 workflow。

![[Agent 核心循环-时序.png]]

## 终态协议：把「停下」说清楚

| 状态 | 含义 | 上游的典型动作 |
|---|---|---|
| `completed` | 验收条件已满足，附可核查证据 | 交付结果 |
| `needs_input` | 缺少用户提供的数据、目标或选择 | 请求补充输入后续跑 |
| `needs_approval` | 下一动作有副作用且需授权 | 展示动作与影响，等待批准 |
| `blocked` | 外部前提、工具可用性或策略限制阻断且无安全替代 | 升级、修复依赖或改目标 |
| `failed` | 已允许的路径在重试/降级后仍无法完成 | 返回错误与诊断，必要时人工接手 |
| `budget_exhausted` | token、费用、时间或步数预算耗尽 | 终止或由上游分配新预算 |

人工审批和用户输入通常是**可恢复暂停**，但对当前 `run` 仍是明确终态。`completed` 应包含结果、证据与验收版本；其余状态应包含原因、可恢复方式和已消耗预算。

![[Agent核心循环-停机状态机.png]]

## 最小可运行例子：类型化终态不把「停止」当成功

**❌ 朴素写法：**把模型给出的终态字符串原样交付，`"completed"` 即使没有证据也会被误判为成功。

**✅ 改进写法：**用枚举、字面量、`TerminalState` 与验收谓词约束终态；无证据的 `completed` 会转为 `failed`。

```python
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal


# ❌ 朴素写法：终止字符串被直接当成完成，无法表达证据或预算原因。
def naive_finalize(status: str) -> str:
    return status


# ✅ 改进写法：终态有受限的状态名、证据和预算维度。
class TerminalStatus(str, Enum):
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


TerminalName = Literal[
    "completed", "needs_input", "needs_approval",
    "blocked", "failed", "budget_exhausted",
]
BudgetType = Literal["tokens", "steps", "time", "cost"]


@dataclass(frozen=True)
class TerminalState:
    status: TerminalStatus
    detail: str
    evidence: str | None = None
    used_tokens: int = 0
    budget_type: BudgetType | None = None


def terminal(
    status: TerminalName,
    detail: str,
    evidence: str | None = None,
    used_tokens: int = 0,
    budget_type: BudgetType | None = None,
) -> TerminalState:
    return TerminalState(TerminalStatus(status), detail, evidence, used_tokens, budget_type)


def is_success(state: TerminalState, accepts: Callable[[str], bool]) -> bool:
    return (
        state.status is TerminalStatus.COMPLETED
        and state.evidence is not None
        and accepts(state.evidence)
    )


def finalize(state: TerminalState, accepts: Callable[[str], bool]) -> TerminalState:
    if state.status is TerminalStatus.COMPLETED and not is_success(state, accepts):
        return terminal(
            "failed", "completed 缺少可验收证据", state.evidence,
            state.used_tokens, state.budget_type,
        )
    return state


if __name__ == "__main__":
    accepts = lambda evidence: evidence == "tests=green"
    naive = naive_finalize("completed")
    missing_evidence = finalize(terminal("completed", "模型说已完成"), accepts)
    verified = finalize(terminal("completed", "测试已通过", "tests=green"), accepts)
    waiting = finalize(terminal("needs_input", "缺少仓库路径"), accepts)
    exhausted = finalize(
        terminal("budget_exhausted", "达到 token 上限", used_tokens=100_000, budget_type="tokens"),
        accepts,
    )

    assert naive == "completed"  # 这正是朴素写法的问题：没有证据仍显示完成。
    assert missing_evidence.status is TerminalStatus.FAILED
    assert verified.status is TerminalStatus.COMPLETED and is_success(verified, accepts)
    assert waiting.status is TerminalStatus.NEEDS_INPUT and not is_success(waiting, accepts)
    assert exhausted.status is TerminalStatus.BUDGET_EXHAUSTED
    assert exhausted.budget_type == "tokens" and exhausted.used_tokens == 100_000
    print(f"naive: {naive}（无证据仍被当作完成）")
    print(f"{missing_evidence.status.value}: {missing_evidence.detail}")
    print(f"{verified.status.value}: {verified.evidence}")
    print(f"{waiting.status.value}: {waiting.detail}")
    print(f"{exhausted.status.value}: {exhausted.budget_type}={exhausted.used_tokens}")
```

这里的 `TerminalName` 和 `BudgetType` 限制允许的字符串字面量，`TerminalStatus` 提供运行时枚举，`TerminalState` 是上游可消费的类型化终态。`finalize` 明确将「声称 completed 但证据不通过」转成 `failed`；真实循环只需在每次模型/工具回合后构造并消费这个终态。

## 与 ReAct 的关系

[[09 ReAct|ReAct]] 是这类回路的重要 prompt 范式：论文让模型交错生成 reasoning trace 与 action，再从外部环境取得 observation。现代 function calling 可把 action 表达成结构化调用，模型的推理也未必暴露给用户；但两者都遵循「行动结果影响下一步」的闭环。

- 核心循环是实现无关的运行时骨架。
- ReAct 是让模型以 `Thought → Action → Observation` 形式组织回合的一种具体方法。
- [[10 Plan-and-Execute|Plan-and-Execute]]、[[13 Reflection 与 Reflexion|Reflection]] 等是在同一骨架上增加规划或评估步骤，而不是自动取消终态、预算和审批要求。

## 何时用、坑与反模式

✅ 任务需要多轮工具交互、步骤不可可靠预定，且每轮有可用反馈与可验证验收时，才需要这个循环。

- **把无 tool call 当完成**：会吞掉缺输入、缺批准、未验证结果和协议异常；用终态 schema 与验收谓词替代。
- **错误直接抛出**：可恢复的工具错误应作为 observation 回灌；连续失败再以 `failed` 结束。
- **observation 不处理**：超长日志、机密数据或无关输出会污染上下文；回灌前做裁剪、摘要和脱敏，见 [[20 上下文工程|上下文工程]]。
- **只增不减地累积状态**：长任务需压缩或外置记忆，见 [[19 Agent 记忆系统|Agent 记忆系统]]、[[21 上下文压缩与卸载|上下文压缩]]。
- **用循环硬扛固定任务**：路径若可预列，应回到 [[02 Workflow 与 Agent 的边界|workflow]]。

## 工业界实践：谁来跑循环

- **OpenAI Agents SDK**：官方文档说明 SDK runner 可执行工具循环、handoff，并在 run 完成或为审批暂停时停止；若需要自定义循环与分支，可使用 Responses API 自行拥有循环。
- **Claude Agent SDK**：官方文档提供与 Claude Code 相同的工具、agent loop 与上下文管理，并暴露权限、会话与可观测能力。
- **LangGraph**：文档将其定位为支持持久执行、持久化和 human-in-the-loop 的编排运行时，适合把状态、暂停和恢复显式建模。

无论使用哪个 SDK，应用仍需定义自己的成功谓词、工具权限、预算维度和终态消费逻辑；框架只能帮你运行循环，不能替你判定业务完成。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q1：描述 agent 的核心循环。**

标准答：harness 组装状态和工具定义 → 模型提出 tool call 或结构化终态 → harness 校验、执行、处理 observation → 回灌进入下一轮。循环由预算、权限和终态协议约束。

**Q2：为什么「无 tool call」不等于成功？**

标准答：它只说明这一轮没有请求工具，可能是模型在等输入/批准、协议格式错误或没有验证结果。只有 `status=completed` 且验收谓词对证据为真，才算成功。

**Q3：六种终态怎样区分？**

标准答：`completed` 是验收通过；`needs_input` 和 `needs_approval` 是可恢复暂停；`blocked` 是外部前提或策略阻断；`failed` 是允许路径重试后仍失败；`budget_exhausted` 是资源限制触发。上游据此决定续跑、升级还是终止。

**Q4：harness 与模型如何分工？**

标准答：模型做受约束的候选决策；harness 管状态、参数校验、执行、权限、预算、脱敏、持久化和终态交接。不要让模型文本绕过这些控制面。

**Q5：ReAct 与核心循环的关系？**

标准答：ReAct 是交错 reasoning 与 action 的一种 prompt 范式；核心循环是更上位的运行时抽象。现代结构化工具调用可以采用同一闭环而不公开详细推理。

## 关键事实

- ReAct 研究将 reasoning trace 与 task-specific action 交错生成，并通过外部环境获得信息：Yao 等，*ReAct: Synergizing Reasoning and Acting in Language Models*，https://arxiv.org/abs/2210.03629 ，2022。
- Anthropic 建议 agent 在执行中利用环境的 ground truth、在检查点或阻塞时回到人，并设置最大迭代等停止条件：*Building effective agents*，https://www.anthropic.com/engineering/building-effective-agents ，2024。
- OpenAI 当前文档说明 Agents SDK runner 管理工具循环、handoff、审批暂停与运行生命周期：*Agents SDK*，https://developers.openai.com/api/docs/guides/agents ，2026（访问于 2026-07）。
- Claude Agent SDK 当前提供工具、agent loop 与上下文管理：Anthropic，*Agent SDK overview*，https://code.claude.com/docs/en/agent-sdk/overview ，2026（访问于 2026-07）。
- LangGraph 当前将 durable execution、persistence 与 human-in-the-loop 列为编排运行时能力：LangChain，*LangGraph overview*，https://docs.langchain.com/oss/python/langgraph/overview ，2026（访问于 2026-07）。
