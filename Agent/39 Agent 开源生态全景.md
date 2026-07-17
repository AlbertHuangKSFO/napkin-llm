[[39 Agent 开源生态全景|Agent 开源生态全景]] 的本质是：所谓“Agent 生态”不是一排互相替代的框架，而是一组可组合的职责层——运行时、模型/工具适配、知识与状态、协议、安全、评测、可观测和部署。选型应先把每项职责落到一个可验证的 owner，再决定哪些开源组件、托管服务或自建代码承担它。

本页是**截至 2026-07-17**的来源快照，不含 GitHub 星标、市场份额或“第一/最佳”结论。软件版本快速变化，表内版本只说明本文核验的发布/文档边界；部署应重新核验许可证、维护状态、API 兼容性、数据处理和锁定版本。

## 直觉：工具箱不是一把万能瑞士军刀

造一间厨房需要炉灶、冰箱、排烟、食品安全与账本；把它们都称为“厨房框架”会导致错误比较。Agent 系统也一样：[[37 Agent 框架对比|运行时框架]] 管状态与控制流，[[17 MCP 模型上下文协议|MCP]] 一类协议管工具边界，检索组件管知识，[[38 Agent 评估与可观测性|eval/trace]] 管质量证据。一个项目可只需要其中两层，也可把不同层替换而不重写业务。

因此，生态选型的第一问不是“用哪个名字”，而是“这个责任是否已经有明确接口、数据所有者和验收方式”。

## 小数字手算：少一个未受控接口，常比多一个功能更值钱

比较两套满足同一需求的最小栈。A 的基础接入成本为 $4$ 人日，另有三个自定义 adapter，各 $2$ 人日；B 的基础接入为 $6$ 人日，只有一个 adapter，成本 $1$ 人日。短期接入账为：

$$
E_A=4+3\times2=10\ \text{人日},\qquad E_B=6+1=7\ \text{人日}
$$

若每个 adapter 的半年维护风险期望为 $0.5$ 人日，则：

$$
E_A^{6m}=10+3\times0.5=11.5,\qquad E_B^{6m}=7+1\times0.5=7.5
$$

这不是某产品的客观报价，而是一个提醒：接口数量、版本漂移和所有权会进入总成本。也不能为了“组件更少”而接受不满足安全、恢复或审计门槛的栈。

## 公式推导：生态选择是职责覆盖与接口风险的平衡

令必须职责集合为 $R$，候选栈为 $S$，组件 $c$ 覆盖的职责为 $cover(c)$。先满足：

$$
R\subseteq\bigcup_{c\in S}cover(c)
$$

在满足硬门槛后，比较可量化总成本：

$$
T(S)=C_{adopt}(S)+C_{operate}(S)+\sum_{e\in I(S)}P(\text{break}_e)\cdot C(\text{repair}_e)
$$

其中 $I(S)$ 是组件之间的接口集合。`C_operate` 要包括凭据轮换、数据留存、升级、on-call、审计和 eval，不只是 `pip install`。若一个接口跨越信任边界，还要把最小权限、人审和 terminal-state verification 视为硬约束而不是可加分功能。

## 手绘图

![[Agent 开源生态全景.png]]

![[Agent 开源生态全景-选型.png]]

![[Agent-生态需求矩阵.png]]

## 可运行代码：❌ 按名气堆栈 vs ✅ 按职责与硬约束组合

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Component:
    name: str
    roles: frozenset[str]
    hard: frozenset[str]
    adapters: int

REQUIRED_ROLES = {"runtime", "tool_boundary", "trace", "eval"}
REQUIRED_HARD = {"approval", "exportable_trace"}

def viable(stack: list[Component]) -> bool:
    roles = set().union(*(c.roles for c in stack))
    guarantees = set().union(*(c.hard for c in stack))
    return REQUIRED_ROLES <= roles and REQUIRED_HARD <= guarantees

def integration_score(stack: list[Component]) -> int:
    # 越少自定义 adapter 越容易维护；实际还需 POC 与安全审查。
    return sum(c.adapters for c in stack)

if __name__ == "__main__":
    stack = [
        Component("state-runtime", frozenset({"runtime"}), frozenset({"approval"}), 1),
        Component("mcp-adapter", frozenset({"tool_boundary"}), frozenset(), 1),
        Component("otel-pipeline", frozenset({"trace"}), frozenset({"exportable_trace"}), 0),
        Component("eval-harness", frozenset({"eval"}), frozenset(), 1),
    ]
    print("可部署候选：", viable(stack))
    print("需维护 adapter 数：", integration_score(stack))
```

❌ “全都装上”不会自动覆盖职责：例如 trace 平台不等于 eval harness，工具协议不等于权限系统。✅ 每项 `hard` 均须在锁定版本的 POC 中有配置、测试与运行证据，不能由营销页推断。

## 六层地图：责任先于产品

| 层 | 必须回答的问题 | 可考察的开源/开放接口例子 | 验收证据 |
|---|---|---|---|
| 1. 运行时/编排 | 状态、分支、重试、恢复如何表达？ | LangGraph、Google ADK、PydanticAI、CrewAI、LlamaIndex Workflows；AutoGen（仅既有维护/迁移）；Microsoft Agent Framework（新项目候选） | 中断恢复与重复写入测试 |
| 2. 模型与适配 | provider、模型版本、结构化输出怎样隔离？ | provider SDK、内部 adapter | 换一 provider 的契约测试 |
| 3. 工具/协议 | 工具 schema、认证、授权、审计在哪一层？ | [MCP](https://modelcontextprotocol.io/specification/2025-06-18) 与显式 function tools | 越权、注入、幂等与审批测试 |
| 4. 知识与状态 | 文档、向量/关键词索引、会话和业务状态由谁拥有？ | LlamaIndex 等索引/workflow 组件，业务数据库 | 来源、权限、时效与恢复测试 |
| 5. 评测/可观测 | 何为正确，如何复现与归因？ | OpenTelemetry GenAI 约定、内部 eval harness | task/trial/outcome 与 trace 导出 |
| 6. 部署/治理 | 凭据、预算、队列、回滚、数据保留谁负责？ | CI/CD、密钥系统、队列、策略引擎 | 最小权限、告警、回滚演练 |

注意第 4 层的“知识”不等于第 1 层的“状态”：检索到的文档证据与可恢复的业务事务是不同对象。把两者混在模型上下文里，会同时损害新鲜度、权限和恢复能力。

## 版本与来源快照：用作候选清单，不用作排名

| 组件/规范 | 本文核验的版本或文档边界 | 来源与可验证定位 |
|---|---|---|
| LangGraph | 1.2.9 | [官方发布页](https://github.com/langchain-ai/langgraph/releases)：显式 runtime 候选 |
| Google ADK | 2.4.0 | [官方发布页](https://github.com/google/adk-python/releases)、[文档](https://adk.dev/)：多语言 agent SDK 候选 |
| AutoGen | Python v0.7.5；maintenance mode | [官方仓库快照](https://github.com/microsoft/autogen)：community-managed，适合既有项目维护/迁移；新项目不应仅因既有示例直接选用 |
| Microsoft Agent Framework | [官方发布页](https://github.com/microsoft/agent-framework/releases) 核验：2026-07-17 | 新项目应作为运行时候选，按同一 POC 验证恢复、治理、观测与 provider 边界；部署时锁定实际测试版本 |
| PydanticAI | 2.12.0 | [官方发布页](https://github.com/pydantic/pydantic-ai/releases)、[文档](https://pydantic.dev/docs/ai/overview/)：Python 类型化候选 |
| CrewAI | 发布 1.15.4；文档显示 v1.14.6 | [发布页](https://github.com/crewAIInc/crewAI/releases)、[文档](https://docs.crewai.com/)：版本不一致时先做兼容 POC |
| LlamaIndex | 0.14.23 | [官方发布页](https://github.com/run-llama/llama_index/releases)、[Workflow 文档](https://developers.llamaindex.ai/python/llamaagents/workflows/)：检索/事件工作流候选 |
| MCP | 规范 2025-06-18 | [规范页](https://modelcontextprotocol.io/specification/2025-06-18)：工具/上下文协议，不是权限替代品 |
| OpenTelemetry GenAI | 演进中的语义约定 | [官方规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：trace 语义互操作候选 |

版本号仅是此次核验锚点。选择某行后，应把精确依赖、镜像 digest、配置和迁移说明写进部署仓库，并在升级 PR 里重跑 [[38 Agent 评估与可观测性|评测]]。

## 组合原则与反模式

- **先薄后厚**：从一个显式 runtime、少数工具和一个 eval harness 开始；当真实需求证明需要时再拆子 agent 或引入更多层。
- **协议与权限分开**：MCP 说明客户端/服务器交互，不自动做用户授权、业务审批、输入校验或数据分级。
- **避免双重状态真相**：业务系统是副作用的权威来源；agent memory/向量库可辅助，不应伪装成事务账本。
- **不要把 SaaS 与开源混为一类**：开源仓库、托管控制面、企业版与自托管部署的许可、数据路径和运维责任可能不同，采购前逐项验证。
- **观测不等于质量**：能看到 trace 不表示任务正确；必须在 harness 中定义 terminal verifier、重复试次和 release 门槛。

## 面试高频

**Q：为什么生态图要按层画？**  因为框架、协议、索引、trace 和部署解决不同问题。按层能避免把“某库支持工具”误解为“已解决授权、状态或评测”，也让替换一个组件的成本可估算。

**Q：开源组件是否天然更可控？**  不一定。源码可见不等于部署、依赖供应链、数据留存、模型 provider 和操作权限可控；控制面与数据面都要在目标环境验证。

**Q：何时应自建而不是引入框架？**  当工作流确定、接口少、团队已有状态/观测基础且框架会遮蔽关键语义时，自建一层小而可测的 adapter 可能更低风险。反之，恢复、分支、人审和多工具协作已成为核心需求时，成熟 runtime 候选值得 POC。

## 关键事实

- 生态组件应按责任组合，而非凭项目热度比较。运行时选型展开见 [[37 Agent 框架对比|Agent 框架对比]]；成本与接口总账见 [[35 Agent 成本与延迟优化|成本与延迟优化]]。
- [MCP 2025-06-18 规范](https://modelcontextprotocol.io/specification/2025-06-18) 是工具/上下文交互协议；[OpenTelemetry GenAI 规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) 是可观测语义约定。两者均不替代业务授权与 terminal-state verification（核验：2026-07-17）。
- 上表版本来自各项目官方发布页/文档，已明确标注 CrewAI 的发布—文档版本差异，以及 AutoGen 的 maintenance/community-managed 边界；任何“适合”均限定在该日期、版本和 POC 验收条件下。根据 [AutoGen 官方仓库快照](https://github.com/microsoft/autogen)（核验：2026-07-17），新项目应把 [Microsoft Agent Framework 官方发布页](https://github.com/microsoft/agent-framework/releases) 纳入评估，AutoGen 则聚焦既有维护或迁移。
- Agentic RAG 是第 3/4 层与运行时闭环的组合，定义与边界见 [[36 Agentic RAG|Agentic RAG]]；它不要求全套生态组件。
