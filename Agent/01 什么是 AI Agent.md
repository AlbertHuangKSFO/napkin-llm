[[01 什么是 AI Agent|AI Agent]] 是为了达成目标而让模型根据环境反馈选择下一步行动的系统；本文聚焦以 LLM 为决策器、能调用工具的 LLM Agent。

这里的 **agent** 是系统角色，不等于某个模型或一段 prompt：LLM 负责提出下一步，[[23 Agent Harness 概览|Agent Harness]] 负责保存状态、执行工具、施加权限与预算。更宽泛的「AI agent」还可用规则或规划器实现；本域讨论的边界是 LLM 是否在运行时主导流程与工具使用。它与 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 的区别，归根到底是「下一步由谁决定」。

## 本质：一个感知—决策—行动的闭环

一个 LLM Agent 通常由四个部件和一条回路组成：

- **决策器（LLM）**：基于当前上下文提出行动、询问用户，或给出结构化终态。
- **工具（Tools）**：让系统搜索、读写文件、执行代码或访问受控 API，见 [[15 Function Calling 工具调用|Function Calling 工具调用]]、[[16 工具设计与工具层|工具设计]]。
- **状态与记忆（State/Memory）**：短期状态是本次运行的消息、观察和预算；跨会话知识可放在外部存储，见 [[19 Agent 记忆系统|Agent 记忆系统]]。
- **环境（Environment）**：文件系统、浏览器、数据库或用户；行动的可验证反馈从这里来。

闭环是：**观察环境 → 决定下一步 → 执行动作 → 回灌反馈 → 再观察**。模型并不直接「做事」；它提出候选行动，harness 进行校验、执行与记录。核心运行机制见 [[03 Agent 核心循环|Agent 核心循环]]。

![[什么是 AI Agent.png]]

## 直觉：像一位有权限边界的实习生

把 agent 想成实习生而不是「会自动完成任务的模型」：你给目标、工具清单和权限边界；它先看日志，再选择读文件、跑测试或向你提问。每次动作的结果都会改变下一步。若缺少账号、验收标准或付款批准，正确行为是停下来报告 `needs_input` 或 `needs_approval`，不是猜测后继续。

## 小数字手算：一次修测试的回路成本

设 agent 为「修复一个测试失败」运行三轮：第 1 轮搜索报错，调用 1 个工具；第 2 轮读取 2 个文件，调用 2 个工具；第 3 轮运行测试，调用 1 个工具，随后提交终态。

$$
N_{tool}=1+2+1=4,\qquad N_{model}=3+1=4
$$

其中最后一次模型调用用于根据测试结果报告终态，而不是调用工具。这个数字不是性能基准；它说明 agent 的模型调用和工具调用数由轨迹决定，不能仅从用户目标预先精确写死。

## 公式推导：为什么「反馈」使它成为回路

令第 $i$ 轮进入模型的状态为 $s_i$，模型策略为 $\pi$，harness 的执行器为 $E$：

$$
a_i=\pi(s_i),\qquad o_i=E(a_i),\qquad s_{i+1}=U(s_i,a_i,o_i)
$$

若 $a_i$ 是工具调用，$o_i$ 是工具结果；若 $a_i$ 是结构化终态，则运行结束。关键在 $U$：下一轮状态包含真实环境反馈，因此下一步能随结果改变。若所有 $a_i$ 与分支都由代码预先给定，便是 [[02 Workflow 与 Agent 的边界|workflow]]，即使其中调用了多次 LLM。

## 机制：一次完整运行的生命周期

1. **接收目标与约束**：明确成功条件、可用工具、权限、预算与截止时间。
2. **观察**：读取当前状态，例如报错日志、文件树或用户补充的信息。
3. **决策**：模型选择工具、给出下一步计划，或声明需要输入/批准。
4. **受控执行**：harness 校验参数、权限和频率，再执行工具。
5. **回灌与评估**：将截断后的结果写入状态，检查是否满足验收条件。
6. **返回终态**：完成、需要用户输入、需要批准、被阻塞、失败或预算耗尽；「不再调用工具」本身不足以证明完成。

与单次问答的差别不在于「用了几个 prompt」，而在于系统是否基于环境反馈动态选择后续行动与终态。

## 来源：现代 LLM Agent 的工程切分

- 经典 AI 中，agent 是能感知环境并采取行动的实体。
- Anthropic 将 agentic system 区分为两类：**workflow** 由预定义代码路径编排 LLM 与工具；**agent** 由 LLM 动态主导流程和工具使用。
- [[09 ReAct|ReAct]]（Yao 等，2022）提出将推理轨迹与行动交错生成，是理解「观察—行动—再观察」的一条重要研究脉络，而非所有实现都必须采用的格式。

## 最小可运行例子：消费结构化终态

```python
from dataclasses import dataclass
from enum import Enum


class TerminalStatus(str, Enum):
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class RunResult:
    status: TerminalStatus
    detail: str
    evidence: str | None = None


HANDLERS = {
    TerminalStatus.COMPLETED: lambda r: f"交付：{r.evidence or r.detail}",
    TerminalStatus.NEEDS_INPUT: lambda r: f"请求输入：{r.detail}",
    TerminalStatus.NEEDS_APPROVAL: lambda r: f"请求批准：{r.detail}",
    TerminalStatus.BLOCKED: lambda r: f"升级阻塞：{r.detail}",
    TerminalStatus.FAILED: lambda r: f"记录失败：{r.detail}",
    TerminalStatus.BUDGET_EXHAUSTED: lambda r: f"停止并重分配预算：{r.detail}",
}


def dispatch_terminal(result: RunResult) -> str:
    if set(HANDLERS) != set(TerminalStatus):
        raise RuntimeError("每种终态都必须有上游处理器")
    return HANDLERS[result.status](result)


if __name__ == "__main__":
    cases = [
        RunResult(TerminalStatus.COMPLETED, "测试已通过", "tests=green"),
        RunResult(TerminalStatus.NEEDS_INPUT, "缺少仓库路径"),
        RunResult(TerminalStatus.NEEDS_APPROVAL, "准备提交付款"),
        RunResult(TerminalStatus.BLOCKED, "测试服务不可用"),
        RunResult(TerminalStatus.FAILED, "连续三次工具错误"),
        RunResult(TerminalStatus.BUDGET_EXHAUSTED, "token=100000"),
    ]
    for case in cases:
        print(f"{case.status.value}: {dispatch_terminal(case)}")
```

这个离线例子完整覆盖六种终态，并把结构化 `RunResult` 交给上游。真实 `run_loop` 还要回灌工具结果、执行验收与权限检查，见 [[03 Agent 核心循环|Agent 核心循环]]。

## 对比：Agent、chatbot 与 RPA

| 维度 | 对话式 chatbot | RPA | LLM Agent |
|---|---|---|---|
| 下一步控制权 | 对话逻辑或产品代码 | 预设规则/脚本 | 模型在允许范围内选择 |
| 环境反馈 | 可有，但未必驱动行动 | 按固定规则处理 | 反馈参与下一步决策 |
| 工具使用 | 可选 | 固定动作序列 | 动态选择受控工具 |
| 典型适用 | 问答、信息呈现 | 稳定重复流程 | 开放任务、多轮试错 |

![[什么是 AI Agent-自主性谱系.png]]

这不是能力高低排序。带工具的 chatbot 也可能只是 workflow；RPA 也可非常可靠。判定时只问：运行时的流程与工具选择，主要由预设代码还是模型决定？

## 何时真的需要 agent

✅ 任务开放、子步骤难以预先列全，且每轮能从环境得到可用反馈时，例如排查代码问题、跨资料调研或受控浏览器操作。

❌ 步骤稳定可枚举、需要严格可复现或预算/延迟很紧时，先用单次调用或 [[02 Workflow 与 Agent 的边界|workflow]]。多轮 agent 往往增加调用次数和轨迹不确定性；是否值得，应由评测与验收结果决定。

## 坑与反模式

- **为 agent 而 agent**：分类、抽取等固定任务硬套循环，只增加成本与不确定性。
- **把模型文本当作完成证明**：模型不调工具或说「完成」时，仍应由 harness 对照验收条件检查证据。
- **没有边界**：缺少最大步数、token/费用/时间预算、权限与人工批准，会让失败难以收敛。
- **工具接口含糊**：不清楚的参数和噪声过大的结果会降低决策质量，见 [[16 工具设计与工具层|工具设计]]。
- **上下文只增不减**：长任务要截断、摘要或外置状态，见 [[20 上下文工程|上下文工程]]、[[21 上下文压缩与卸载|上下文压缩]]。

## 工业界实践：框架只是运行时选择

- **OpenAI Agents SDK**：官方将其定位为管理 agent loop、反复工具调用、审批暂停、状态与追踪的 SDK；若要完全自定义循环，则使用 Responses API 自己执行函数调用回合。
- **Claude Agent SDK**：提供驱动 Claude Code 的工具、agent loop 与上下文管理，适合把同类 harness 能力嵌入应用。
- **LangGraph**：文档定位为编排运行时，强调持久化、持久执行和 human-in-the-loop；图中的边是否由代码或模型决定，才决定它在本题里更像 workflow 还是 agent。

选择框架前先写清：目标的验收条件、可用工具、哪些动作需批准、预算如何计量、每个终态如何交给上游处理。框架不能替代这些产品与安全决策。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q1：一句话定义 AI Agent；它和 chatbot / workflow 的本质区别？**

标准答：本文的 LLM Agent 是让模型在工具与权限边界内，根据环境反馈动态决定下一步的系统。chatbot、RPA 或多步 LLM 系统不自动等于 agent；判据是「编排权归谁」，详见 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]]。

**Q2：一个 agent 的最小组成是什么？**

标准答：LLM 决策器、工具、外部状态/记忆、环境，以及运行这些部件的 harness。模型决定候选下一步；harness 负责状态、执行、权限、预算和终态处理。

**Q3：什么时候该上 agent？**

标准答：当任务步骤无法可靠预列、需要多轮环境反馈且能设置验证与护栏时才上。先验证单次调用或 workflow 是否足够；agent 是用可预测性、成本和调试难度交换灵活性。

**Q4：agent 为什么会跑飞，如何防？**

标准答：无明确终态、重复失败、预算无限或把模型输出误认完成都会跑飞。应设结构化终态、验收检查、最大步数/费用/时间预算、失败阈值、审批和沙箱，见 [[03 Agent 核心循环|Agent 核心循环]]。

## 关键事实

- Anthropic 对 workflow 与 agent 的工程切分，以及「先找最简单可用方案」的建议：*Building effective agents*，https://www.anthropic.com/engineering/building-effective-agents ，2024。
- ReAct 交错生成 reasoning trace 与 action，并从环境取回信息：Yao 等，*ReAct: Synergizing Reasoning and Acting in Language Models*，https://arxiv.org/abs/2210.03629 ，2022。
- OpenAI 当前文档说明：Responses API 适合自行拥有循环；Agents SDK 可管理循环、追踪、护栏与可恢复审批：*Agents SDK*，https://developers.openai.com/api/docs/guides/agents ，2026（访问于 2026-07）。
- 对应关系：概念总览在本篇；控制权边界见 [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]]；实现级状态机见 [[03 Agent 核心循环|Agent 核心循环]]；多智能体扩展见 [[22 多智能体系统|多智能体系统]]。
