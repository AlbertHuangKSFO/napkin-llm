[[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]] 的核心判据是：预设代码路径在决定下一步时是 **workflow**；LLM 在运行时选择流程与工具时是 **agent**。

两者都可以调用 LLM、工具和数据库，也都可以多步运行；这些都不是分类标准。应从控制权、验收方式和生命周期边界做设计，并与 [[01 什么是 AI Agent|AI Agent]]、[[03 Agent 核心循环|Agent 核心循环]] 一起理解。

## 本质：谁掌握「下一步做什么」的控制权

- **Workflow**：工程师在代码或图中预先定义顺序、分支、重试和终态。LLM 可以是某个节点的「智能算子」，但不能任意改变主路径。
- **Agent**：系统给模型目标、工具和边界；模型在每轮根据状态选择工具、子任务或终态。harness 仍拥有权限、预算和最终执行权。

所以它不是「有没有循环」的区别：`for` 循环、条件边、甚至多个 LLM 调用都可能是 workflow。关键是运行时对下一步的**主要编排权**在代码还是在模型。

![[Workflow 与 Agent 的边界.png]]

## 直觉：菜谱与受监督的厨师

workflow 像菜谱：配料、顺序和火候分支预先写好，厨师只在一个步骤内发挥。agent 像受监督的厨师：给目标与食材后，它可决定先备菜还是先试味；但燃气、采购和上菜仍由厨房规则控制。两者都能做菜，区别在谁决定下一步以及哪些动作必须经过审批。

## 小数字手算：可估算性来自预设路径

有 10 封邮件：6 封无金额，4 封有金额。若 workflow 固定为「翻译；有金额再换算并润色」，无金额邮件各走 1 次模型调用，有金额邮件各走 2 次：

$$
N_{workflow}=6\times1+4\times2=14
$$

若受限 agent 每封邮件允许 1 到 4 轮模型调用，则同一批数据的调用次数落在

$$
10\times1\le N_{agent}\le10\times4\quad\Rightarrow\quad10\le N_{agent}\le40
$$

这不是二者质量比较，也不是实际成本报价；它只说明 workflow 的路径成本可由已知分支计算，agent 的成本还取决于运行时轨迹。

## 公式推导：控制权如何写成程序

令输入为 $x$、状态为 $s_i$、下一步动作为 $a_i$。

workflow 的转移函数由工程师固定：

$$
a_i=f_{code}(s_i,x),\qquad s_{i+1}=U(s_i,a_i,o_i)
$$

agent 则把候选行动交给模型策略，harness 再施加约束 $G$：

$$
a_i=G\bigl(\pi_{LLM}(s_i,x)\bigr),\qquad s_{i+1}=U(s_i,a_i,o_i)
$$

两者都有状态更新 $U$ 和观察 $o_i$。差异是产生 $a_i$ 的控制器：前者是 $f_{code}$，后者主要是 $\pi_{LLM}$；$G$ 始终可以拒绝、暂停或终止行动。

## 机制：从同一任务走两条路

任务：「将英文邮件翻成中文；若有金额，按给定汇率标注人民币；最后生成摘要。」

**可运行对照：workflow 固定路径 vs 模型选择动作**

**❌ 朴素写法：**把一个步骤未知的任务也写成固定步骤，后续需求一变就要改流程代码。

**✅ 改进写法：**保留最小工具集与预算，但每轮由 `policy(state)` 在最新 observation 上选择下一动作。

```python
import re
from collections.abc import Callable
from dataclasses import dataclass, field

WORKFLOW_STEPS = ("translate", "convert_if_needed", "summarize")
ALLOWED_ACTIONS = frozenset({"translate", "convert", "summarize"})
MONEY = re.compile(r"(\d+) USD")
ZH = {"pay": "支付"}


def translate(text: str) -> str:
    return " ".join(ZH.get(token, token) for token in text.split())


def convert_amount(text: str, usd_cny: int) -> str:
    return MONEY.sub(lambda m: f"{int(m.group(1)) * usd_cny} CNY", text)


def summarize(text: str) -> str:
    return f"摘要：{text}"


# ❌ 朴素写法：所有步骤和分支由工程师预先指定。
def deterministic_workflow(email: str, usd_cny: int) -> str:
    text = email
    for step in WORKFLOW_STEPS:  # 每一步和分支都由代码预设。
        if step == "translate":
            text = translate(text)
        elif step == "convert_if_needed" and MONEY.search(text):
            text = convert_amount(text, usd_cny)
        elif step == "summarize":
            text = summarize(text)
    return text


# ✅ 改进写法：每轮策略都读取前一轮写回的 observation。
@dataclass
class AgentState:
    text: str
    observations: list[str] = field(default_factory=list)


Policy = Callable[[AgentState], str]


def decide_next_action(state: AgentState) -> str:
    """离线策略 stub：真实系统在这里调用模型并校验其结构化动作。"""
    if not state.observations:
        return "translate"
    if state.observations[-1] == "translated":
        return "convert" if MONEY.search(state.text) else "summarize"
    if state.observations[-1] == "converted":
        return "summarize"
    raise RuntimeError(f"未知 observation：{state.observations[-1]}")


def model_directed_loop(
    email: str, usd_cny: int, policy: Policy = decide_next_action, max_steps: int = 4
) -> str:
    state = AgentState(email)
    for _ in range(max_steps):
        action = policy(state)  # 每轮从上一次 observation 更新后的 state 决策。
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"未允许的动作：{action}")
        if action == "translate":
            state.text = translate(state.text)
            state.observations.append("translated")
        elif action == "convert" and MONEY.search(state.text):
            state.text = convert_amount(state.text, usd_cny)
            state.observations.append("converted")
        elif action == "summarize":
            return summarize(state.text)  # 策略选择结构化终止动作。
        else:
            raise RuntimeError("策略请求的动作不适用于当前 state")
    raise TimeoutError("超过 max_steps，未得到 summarize 终止动作")


if __name__ == "__main__":
    email, rate = "pay 5 USD", 7
    fixed = deterministic_workflow(email, rate)
    seen_observations: list[tuple[str, ...]] = []

    def tracing_policy(state: AgentState) -> str:
        seen_observations.append(tuple(state.observations))
        return decide_next_action(state)

    dynamic = model_directed_loop(email, rate, policy=tracing_policy)
    assert fixed == dynamic == "摘要：支付 35 CNY"
    assert seen_observations == [(), ("translated",), ("translated", "converted")]
    print(f"workflow={fixed}")
    print(f"agent={dynamic}")
    print(f"policy_observations={seen_observations}")
```

`decide_next_action(state)` 在每次动作把 observation 写回 state 后，才会被下一轮调用；这正是 agent 与预先传入动作序列的差别。真实系统可用模型的结构化输出替换这个 stub，但仍须保留 `ALLOWED_ACTIONS`、最大步数和参数校验。

第二段并非「更高级」。只要金额规则、数据源和输出格式已稳定，第一段往往更容易验证。只有当邮件结构、所需资料或处理步骤难以预列时，第二段的运行时决策才有价值。

## 取舍：一张账

| 维度 | Workflow | Agent |
|---|---|---|
| 下一步控制权 | 代码/图定义 | 模型在护栏内选择 |
| 路径与成本 | 可按分支估算 | 随轨迹波动 |
| 测试重点 | 节点、边和固定输出 | 轨迹、工具选择、终态与护栏 |
| 适用 | 步骤稳定、规则明确 | 步骤开放、需反馈探索 |
| 失败处理 | 预设重试/补偿/升级 | 还要防重复尝试与错误工具选择 |

![[Workflow 与 Agent 的边界-决策树.png]]

## 生命周期边界：混合系统怎样不失控

真实系统常是混合体，重点是把不确定性限制在可观察、可中断的边界内：

1. **workflow 主干**接收请求、身份校验、路由、预算分配与最终交付。
2. **agent 子节点**只处理开放子问题，并拥有最小工具集、独立的步数/费用/时间预算。
3. agent 以结构化终态返回：`completed`、`needs_input`、`needs_approval`、`blocked`、`failed` 或 `budget_exhausted`。
4. **主干而非模型**决定如何把终态变成重试、人工升级、补充信息或对外响应。

实现时可将上例的 `deterministic_workflow` 或 `model_directed_loop` 作为 workflow 节点：主干先做固定 Routing（见 [[05 Routing|Routing]]），仅把开放子问题交给受限 agent，并由主干消费其结构化终态。

这既允许 agent 探索未知，也避免它把一次局部失败扩大为整条业务流程失控。

## 判定清单：先选最小可行控制器

1. **单次 LLM 调用加检索/示例是否够用？** 够用则不加编排。
2. **步骤、分支和补偿动作能否可靠预列？** 能则用 workflow。
3. **是否必须根据每轮环境反馈决定后续步骤？** 是则考虑受限 agent。
4. **验收、权限、预算和终态交接是否明确？** 未明确时先补这些设计，再扩大自主性。

Anthropic 的建议是从最简单可用方案开始，在有明确收益时再增加复杂度；这是设计顺序，不是「workflow 永远优于 agent」的结论。

## 坑与反模式

- **把多步误认为 agent**：[[04 Prompt Chaining|Prompt Chaining]]、[[05 Routing|Routing]] 可以多步，但若路径预设，仍是 workflow。
- **workflow 硬扛开放问题**：`if-else` 为未知步骤不断膨胀，说明应将该局部改为受限 agent。
- **把终态交给自然语言**：模型写「完成」并不等于验收通过；上游应消费结构化状态与证据。
- **agent 子节点无独立预算或超时**：局部重试会拖垮整个请求的延迟与费用。
- **框架名称代替架构判断**：图框架、SDK 或多智能体库可以实现两边；判据仍是控制权归谁。

## 工业界实践：框架如何映射到边界

- **LangGraph** 文档将其描述为面向 agent 编排的运行时，支持持久化、持久执行和 human-in-the-loop；用固定边写出的图仍是 workflow，用模型决策的条件边才更接近 agent。
- **OpenAI Agents SDK** 可管理循环、工具调用、handoff、追踪与可恢复审批；其官方文档也明确保留由 Responses API 自己拥有循环的选择。
- **Claude Agent SDK** 提供工具、agent loop、上下文管理、权限与会话能力；是否让它自主运行，仍应由系统的验收和权限边界决定。

生产选型应先决定控制面与生命周期，再决定框架：可复现的固定子流程留在 workflow；仅把真正需要探索的子问题交给 agent，并保留可追踪的输入、工具结果与终态。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q1：Workflow 和 Agent 的唯一核心区别是什么？**

标准答：运行时「下一步做什么」的主要控制权在预设代码还是在模型。多步、LLM、工具调用都不是充分判据。

**Q2：为什么不默认使用 agent？**

标准答：若流程能稳定预列，workflow 的路径更易测试、审计和估算。只有需要基于环境反馈探索未知步骤时，才用 agent 的灵活性交换额外不确定性。

**Q3：如何设计 workflow 里嵌 agent？**

标准答：workflow 主干负责入口、权限、预算和终态路由；agent 仅处理开放子问题，使用最小工具集和独立预算，并返回结构化状态而非只返回一段自然语言。

**Q4：Orchestrator-Workers 是 workflow 还是 agent？**

标准答：它可能处在边界上。若整体骨架、停止条件与 worker 调度规则预设，按 workflow 设计与验证；若模型在运行时主导分解与工具选择，则 agent 成分增加。必须说明具体控制权在哪一层。

## 关键事实

- workflow 与 agent 的定义、适用性以及「从最简单方案开始」的建议：Anthropic，*Building effective agents*，https://www.anthropic.com/engineering/building-effective-agents ，2024。
- LangGraph 目前将自身定位为提供持久执行、持久化与 human-in-the-loop 的编排运行时：LangChain，*LangGraph overview*，https://docs.langchain.com/oss/python/langgraph/overview ，2026（访问于 2026-07）。
- OpenAI 当前文档区分「由 Responses API 自己拥有循环」与「由 Agents SDK 管理循环」：OpenAI，*Agents SDK*，https://developers.openai.com/api/docs/guides/agents ，2026（访问于 2026-07）。
- 延伸阅读：固定编排模式见 [[04 Prompt Chaining|Prompt Chaining]] 至 [[08 Evaluator-Optimizer|Evaluator-Optimizer]]；执行级终态与预算见 [[03 Agent 核心循环|Agent 核心循环]]。
