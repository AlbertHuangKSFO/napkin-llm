[[37 Agent 框架对比|Agent 框架对比]] 的本质是：选的不是“最流行的 agent 名字”，而是一组运行时语义——状态如何保存与恢复、工具怎样受控、分支/并发怎样表达、失败怎样重放、日志怎样导出。框架只能降低这些工程成本；它不能替代 [[16 工具设计与工具层|权限设计]]、[[38 Agent 评估与可观测性|评估]] 或业务的 terminal verifier。

本文所有框架信息都是**截至 2026-07-17 的官方文档/发布页快照**，不是市场份额、星标或通用排名。依赖落地时应将“候选版本 + provider + 工具协议 + 评测结果”锁进 `requirements`/lockfile，而不能仅凭这张表升级。

## 直觉：选施工规范，不是选魔法棒

同样是“送包裹”，有的业务只需一张收件单；有的必须记录中转站、可暂停、人工签收、失败重派。前者用一个小而透明的循环或 SDK 可能更好，后者需要显式状态图、持久化与审计。把后者塞进隐式 `while True`，或把前者强塞进多 agent 角色扮演，都会增加不必要的故障面。

先问**不可妥协的运行时约束**：是否需要断点续跑？是否存在有副作用的审批节点？是否必须自托管与 OTel 导出？是否已锁定某语言/云/模型 provider？答案先淘汰不合格候选，再谈开发体验。

## 小数字手算：硬门槛先行，评分只用于同类候选

某团队给四项“软偏好”权重：显式控制 $0.35$、持久化 $0.25$、类型/结构化输出 $0.20$、团队熟悉度 $0.20$。候选 A 的分数为 $(0.9,0.8,0.6,0.7)$：

$$
S_A=0.35\times0.9+0.25\times0.8+0.20\times0.6+0.20\times0.7
=0.775
$$

候选 B 为 $(0.5,0.9,0.9,0.5)$：

$$
S_B=0.35\times0.5+0.25\times0.9+0.20\times0.9+0.20\times0.5
=0.680
$$

似乎 A 更适合。但若 B 不支持项目要求的私网部署或可导出的审计 trace，B 应在**硬门槛**阶段直接淘汰；反过来，A 若不支持必须的恢复语义，也不能被 $0.775$ 掩盖。分数用于比较“都能交付”的候选，不是粉饰缺失能力。

## 公式推导：从需求到可验证选择

令必须项集合为 $H$，候选框架为 $f$，其能力谓词为 $cap(f,h)$。可行性为：

$$
\operatorname{feasible}(f)=\bigwedge_{h\in H}cap(f,h)
$$

对可行候选才计算加权效用：

$$
U(f)=\sum_{j=1}^{m}w_jr_j(f)-\lambda\,\operatorname{migration}(f),\qquad \sum_jw_j=1
$$

`migration` 要把既有工具封装、观测、状态迁移和团队学习写入；否则“快速原型”会被错误地当成总成本最低。最终决策还要通过一个纵切 POC：跑真实工具、注入失败、暂停恢复、输出 OTel trace，再由 [[38 Agent 评估与可观测性|eval harness]] 对任务结果与资源预算验收。

## 手绘图

![[Agent 框架对比.png]]

![[Agent 框架对比-代码风格对照.png]]

## 可运行代码：❌ 口号选型 vs ✅ 可重复的门槛与打分

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Candidate:
    name: str
    hard: dict[str, bool]
    soft: dict[str, float]
    migration_cost: float

REQUIRED = {"audit_trace", "approval", "resume"}
WEIGHTS = {"control": 0.35, "persistence": 0.25, "typing": 0.20, "familiarity": 0.20}

# ❌ “大家都在用 X”没有说明版本、约束或验收方式。

def choose(candidates: list[Candidate]) -> list[tuple[str, float]]:
    viable = [c for c in candidates if all(c.hard.get(k, False) for k in REQUIRED)]
    ranked = []
    for c in viable:
        score = sum(WEIGHTS[k] * c.soft[k] for k in WEIGHTS) - 0.10 * c.migration_cost
        ranked.append((c.name, round(score, 3)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)

if __name__ == "__main__":
    candidates = [
        Candidate("explicit-state-runtime", dict.fromkeys(REQUIRED, True),
                  {"control": .9, "persistence": .8, "typing": .6, "familiarity": .7}, .2),
        Candidate("typed-service-runtime", dict.fromkeys(REQUIRED, True),
                  {"control": .5, "persistence": .9, "typing": .9, "familiarity": .5}, .1),
        Candidate("prototype-only", {"audit_trace": True, "approval": False, "resume": False},
                  {"control": 1, "persistence": 1, "typing": 1, "familiarity": 1}, 0),
    ]
    print(choose(candidates))  # prototype-only 因硬门槛失败被排除
```

代码只说明决策过程，不宣称任何框架天然支持这些谓词。每个 `True` 都需要在锁定版本的 POC 中由测试、文档或运行 trace 证明。

## 截图式生态快照：按运行时能力选候选

| 候选 | 本文使用的版本/来源快照 | 适合先验证的场景 | 先验证的风险 |
|---|---|---|---|
| LangGraph | [1.2.9 发布页](https://github.com/langchain-ai/langgraph/releases) | 显式状态图、分支、检查点与长流程 | 状态 schema、恢复、部署与观测是否匹配团队 |
| Google ADK | [2.4.0 发布页](https://github.com/google/adk-python/releases)，[官方文档](https://adk.dev/) | 已在多语言/Google 工具生态内 | provider、会话服务与部署边界是否可移植 |
| AutoGen | [Python v0.7.5 发布页](https://github.com/microsoft/autogen/releases)，[官方仓库快照](https://github.com/microsoft/autogen) | **既有** AutoGen 项目的维护或迁移 POC | 官方仓库标记为 maintenance mode、后续由社区维护；新项目应同时评估 [Microsoft Agent Framework 官方发布页](https://github.com/microsoft/agent-framework/releases)（核验：2026-07-17） |
| PydanticAI | [2.12.0 发布页](https://github.com/pydantic/pydantic-ai/releases)，[官方文档](https://pydantic.dev/docs/ai/overview/) | Python 服务、强类型输入/输出、结构化测试 | 图、provider 与运行期持久化是否覆盖需求 |
| CrewAI | [1.15.4 发布页](https://github.com/crewAIInc/crewAI/releases)，[文档 v1.14.6](https://docs.crewai.com/) | 以 agent/crew/flow 快速组织业务自动化 | 文档与发布版本不一致时，不能推断功能，须锁定后验收 |
| LlamaIndex Workflows | [0.14.23 发布页](https://github.com/run-llama/llama_index/releases)，[Workflow 文档](https://developers.llamaindex.ai/python/llamaagents/workflows/) | 检索/知识工作流为中心的应用 | 事件模型、索引层与应用状态的边界 |

这不是横向冠军榜。一个候选的“适合”仅是基于该来源快照的能力定位；任何推荐都以你的语言、数据边界、版本和 POC 结果为条件。

## 最小 POC 验收清单

1. **结果**：用固定任务集跑多次，检查答案或 terminal state verifier，而不只看 demo。
2. **状态**：在工具调用前后强制中断，证明可恢复、不会重复写入。
3. **安全**：对写入型工具注入越权参数和提示注入，证明最小权限、审批与幂等键有效。
4. **运维**：导出 trace，至少含模型/版本、脱敏工具事件、状态迁移、token、延迟、错误与 trace ID。
5. **可迁移性**：替换一个模型 provider 或一个工具实现，量化 adapter 改动与回归结果。

若核心 POC 失败，回到更薄的 runtime 或重谈需求；不要为了保住框架选择而隐藏失败。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：何时不该上 agent 框架？**  单次、确定性、无恢复需求的任务用普通函数、队列或工作流往往更透明。只有需要模型决策、工具循环、可暂停状态或人工接管时，框架带来的语义才值得其复杂度。

**Q：多 agent 是选型理由吗？**  不是。先证明一个 agent 无法清晰地分解权限、上下文或并发边界；多角色名称不会自动带来可靠协作，反而增加状态关联与评测难度。

**Q：为什么版本要写进选型记录？**  agent SDK 的状态、持久化、工具与观测 API 变化会直接影响运行时语义。文档能说明方向，锁定版本的 POC 和回归集才说明你的部署真的可用。

## 关键事实

- [LangGraph 1.2.9 发布页](https://github.com/langchain-ai/langgraph/releases)、[Google ADK 2.4.0 发布页](https://github.com/google/adk-python/releases)、[PydanticAI 2.12.0 发布页](https://github.com/pydantic/pydantic-ai/releases) 是本篇版本快照的主要来源（均核验于 2026-07-17）；它们不是长期兼容承诺。
- [AutoGen 官方仓库快照](https://github.com/microsoft/autogen)（核验：2026-07-17）说明 AutoGen 已进入 maintenance mode、停止新增特性并转为 community-managed；因此它适合既有系统维护/迁移评估，而新项目应把 [Microsoft Agent Framework 官方发布页](https://github.com/microsoft/agent-framework/releases)（同日核验）作为单独候选做同一套 POC，并在部署仓库锁定实际测试版本。另有 [CrewAI 文档](https://docs.crewai.com/) 对 agents、crews、flows 的分层，以及 [LlamaIndex Workflow 文档](https://developers.llamaindex.ai/python/llamaagents/workflows/) 对事件与步骤类型检查的说明；这些来源支持“按运行时语义分层”，不支持全局排名。
- 框架必须与 [[17 MCP 模型上下文协议|MCP]]、工具权限、状态存储和 [[38 Agent 评估与可观测性|eval/trace]] 一起选；框架名不是安全或质量保证。
