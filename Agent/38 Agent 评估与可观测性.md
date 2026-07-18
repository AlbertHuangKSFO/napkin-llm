[[38 Agent 评估与可观测性|Agent 评估与可观测性]] 的本质是：agent 的正确性不是“最后一段文本像不像答案”，而是它在授权范围内是否把系统带到正确、可验证的终态。评估定义何为合格；可观测性提供安全、可审计的运行证据。两者共享同一份结构化事件数据，才能把一次失败归因到模型、工具、状态、权限、成本或策略版本。

## 直觉：验收一笔转账，要查账本而不是看聊天气泡

客服 agent 说“退款已完成”不等于退款真的入账；代码 agent 说“测试通过”不等于 CI、提交状态和部署策略都满足条件。最终文本是一个**声明**，terminal state 才是可核验事实。评估像验收单，可观测性像脱敏审计账本：记录谁在什么授权下调用了哪个工具、获得什么状态、为何停止，而不是收集模型的原始内心推理。

## 七个必须分开的定义

| 术语 | 精确定义 | 最常见误用 |
|---|---|---|
| **task（任务）** | 固定初始状态、输入、可用工具/权限、预算与成功谓词的一项工作 | 只写一句自然语言需求 |
| **trial（试次）** | 对同一个 task 的一次独立运行：重置或隔离状态，记录版本、随机种子/采样参数与 trace | 把同一会话连续重试当独立样本 |
| **outcome（结果）** | trial 结束后由成功谓词得到的通过/失败/未知，连同终态快照 | 把 agent 的自我宣称当结果 |
| **grader（评分器）** | 将 outcome、输出或安全轨迹映射为分数/标签的规则、程序 verifier、人工或已校准 LLM judge | 未校准 judge 的“感觉分” |
| **eval harness（评测执行器）** | 负责装载 task、隔离环境、运行试次、收集 trace、调用 grader、汇总与比较版本的系统 | 一段临时 notebook 循环 |
| **repetitions（重复）** | 同一 task 的多次独立 trial，用来测随机性、脆弱性与方差 | 单跑一次就称“稳定” |
| **terminal-state verification（终态验证）** | 在 agent 停止后直接读取权威系统状态，执行成功谓词 | 只验证工具返回 `200 OK` 或最终文本 |

这七项让“agent 做得好吗？”变成可复现的问题，而不是演示时的一次好运气。

## 小数字手算：一次通过不等于五次可靠

某个 task 重复运行 $5$ 次，其中终态 verifier 通过 $4$ 次，经验单次通过率是：

$$
\hat p=4/5=0.8
$$

若把每次 trial 近似看作独立，要求“连续 $k$ 次都成功”的可靠性为 $R_k=p^k$。即便单次通过率为 $0.95$，连续五次都成功也只有：

$$
R_5=0.95^5=0.7737809375\approx77.4\%
$$

这里的 $R_k$ 是“全部 $k$ 次都通过”，不是代码 benchmark 中常见的 `pass@k`（至少一次通过）。对有副作用的 agent，后者可能掩盖四次错误执行，因此要先明确使用哪一种可靠性语义。

## 公式推导：评的是任务分布上的终态，不是一次回复

令评测集为 $D=\{\tau_1,\ldots,\tau_n\}$，每个任务运行 $R$ 次。第 $r$ 次的终态为 $s_{i,r}^{final}$，成功谓词为 $V_i$，则硬 outcome 为：

$$
o_{i,r}=V_i(s_{i,r}^{final})\in\{0,1\}
$$

硬通过率和每任务稳定率分别为：

$$
\operatorname{Pass}=\frac{1}{nR}\sum_{i=1}^n\sum_{r=1}^R o_{i,r},\qquad
\operatorname{Stable}_i=\prod_{r=1}^{R}o_{i,r}
$$

对于没有唯一答案的软目标，grader $g$ 可以给 $[0,1]$ 分，但它应与人工样本校准，并和硬约束分开：

$$
\operatorname{release}=\bigl(\operatorname{Pass}_{hard}\ge q\bigr)\land
\bigl(\overline g\ge s\bigr)\land
\bigl(P95\_latency\le\ell\bigr)\land
\bigl(P95\_cost\le c\bigr)
$$

这使质量、资源和安全都成为发布条件；具体阈值来自业务风险，而不是框架默认值。

## 手绘图

![[Agent 评估与可观测性.png]]

![[Agent 评估与可观测性-评估维度分层.png]]

## 可运行代码：❌ 评分文本 vs ✅ 验证终态并记录安全轨迹

代码可直接运行，模拟“创建工单”。重点是 trial 从干净状态启动，结果来自数据库终态；trace 只保留脱敏的动作、参数摘要、状态和版本，不保存或展示原始 CoT。

```python
from dataclasses import dataclass
from hashlib import sha256
from copy import deepcopy

@dataclass(frozen=True)
class Task:
    owner: str
    title: str

def args_digest(value: object) -> str:
    return sha256(repr(value).encode()).hexdigest()[:12]

def create_ticket(db: dict, owner: str, title: str) -> str:
    ticket_id = f"T-{len(db) + 1}"
    db[ticket_id] = {"owner": owner, "title": title, "state": "open"}
    return ticket_id

def terminal_verifier(db: dict, ticket_id: str, task: Task) -> bool:
    # ✅ 读回权威状态；不是相信 create_ticket 的返回或 agent 的文字。
    row = db.get(ticket_id)
    return row == {"owner": task.owner, "title": task.title, "state": "open"}

def run_trial(task: Task, initial_db: dict, agent_version: str) -> tuple[bool, list[dict]]:
    db, trace = deepcopy(initial_db), []  # 每个 trial 隔离状态
    # ❌ 不要 trace.append({"raw_cot": model_hidden_reasoning})
    ticket_id = create_ticket(db, task.owner, task.title)
    trace.append({
        "event": "tool_call", "tool": "create_ticket", "args_hash": args_digest((task.owner, task.title)),
        "result": "ok", "state_ref": ticket_id, "agent_version": agent_version,
        "decision_summary": "创建满足用户请求的工单",
    })
    passed = terminal_verifier(db, ticket_id, task)
    trace.append({"event": "terminal_verify", "verifier": "ticket_row_equals_expected", "passed": passed})
    return passed, trace

if __name__ == "__main__":
    task = Task(owner="u-17", title="重置密码")
    outcomes = [run_trial(task, {}, "agent-2026-07-17")[0] for _ in range(3)]
    print(outcomes, sum(outcomes) / len(outcomes))  # [True, True, True] 1.0
```

真实 harness 还要隔离真实外部系统（sandbox、临时租户或可撤销夹具）、收集成本/延迟、固定工具与 prompt 版本，并把失败 trial 保存为可复现案例。

## 安全、可审计的 trajectory：记录什么，不记录什么

可观测性要帮助回答“哪个动作导致了什么状态”，不需要原始 chain-of-thought。建议每个 trace/span 记录：

- `trace_id`、task/trial ID、agent/模型/prompt/工具版本、开始结束时间；
- **安全决策摘要**、策略/审批结果、停止原因和预算状态；
- 工具名、权限范围、脱敏或哈希后的参数、返回码、错误类别、幂等键；
- 状态迁移前后摘要、证据/文档 ID、输出引用与 terminal verifier 结果；
- token、缓存、成本、关键路径延迟，以及数据保留/访问控制元数据。

不要默认记录原始提示、密钥、个人数据、完整工具输出或原始 CoT。对于高敏数据，优先最小字段、哈希引用、角色访问控制、保留期和按需取证；人工调试也应走审批。安全轨迹仍足以定位“选错工具、参数被拒、证据不足、状态没有落地还是预算耗尽”。

## 评测闭环：离线门禁 + 在线证据 + 人审校准

1. **离线**：版本化任务集覆盖常规、边界、对抗、工具故障与恢复；多次 trial；hard verifier 优先，LLM judge 仅用于软维度并与人工集校准。**只允许在 dev 集调 prompt、router、grader/rubric 与阈值；release 报告和发布门禁只使用从未参与调参的 held-out 集**，否则指标会发生评测泄漏。
2. **发布门禁**：比较候选与基线的 outcome、稳定率、成本和延迟；指标回退或安全失败即阻断，而非平均分掩盖。
3. **在线**：按风险采样真实流量，监控 terminal verifier、错误、预算、关键路径和审批拒绝；不把线上用户当无隔离的测试集。
4. **人审**：将低置信度、高影响或冲突 trace 送审，把标签回灌任务集和 grader 校准。评估集应随真实失败模式演进。

OpenTelemetry GenAI 语义约定可帮助统一 span 字段，但规范会演进；把内部领域字段（如 `terminal_verify`）与标准字段分层，升级时用兼容测试验证导出。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：为什么最终答案正确仍可能是失败？**  因为有副作用的任务要求权威终态满足谓词。agent 可能调用了错误账户、重复写入、越过审批，或只是猜对了文本；这些都必须由状态和审计轨迹发现。

**Q：LLM-as-judge 能替代 verifier 吗？**  不能。它适合可读性、帮助性等软指标，必须校准、版本化 rubric 并抽样人工复核；权限、金额、文件内容、测试结果等应由确定性或权威系统 verifier 判断。

**Q：为什么不保存原始 CoT 也能调试？**  事故归因通常需要动作、工具参数摘要、状态迁移、证据、错误、版本和终态验证。它们比原始推理更可审计，且减少泄露与不当依赖的风险。

## 关键事实

- OpenAI 的 [Evaluate agent workflows 文档](https://developers.openai.com/api/docs/guides/agent-evals) 明确将 traces、graders、datasets 与 eval runs 作为 agent 质量改进要素（核验：2026-07-17）。本文将它落实为 task、trial、outcome、grader 和 harness 的可复现契约。
- [OpenTelemetry Generative AI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) 提供跨后端语义约定；其状态会变化，因此生产需要固定已验证的规范/SDK 版本并测试导出（核验：2026-07-17）。
- OpenAI 在 [Learning to reason with LLMs（2024）](https://openai.com/index/learning-to-reason-with-llms/) 说明不向用户展示原始 CoT，并展示模型生成摘要的边界；这支持“可观测不是收集原始 CoT”的安全工程取向。
- 可靠性、资源和终态的联合门禁，直接约束 [[35 Agent 成本与延迟优化|成本与延迟]]；检索 agent 则应把证据 ID 和停止原因纳入 trajectory，见 [[36 Agentic RAG|Agentic RAG]]；RAG 专项的证据链、归因与终态验证见 [[RAG/24 RAG 评估：证据链、归因与终态验证|RAG 评估]]；通用检索质量指标与数据集评测见 [[RAG/18 RAG 评估|RAG 评估]]。
