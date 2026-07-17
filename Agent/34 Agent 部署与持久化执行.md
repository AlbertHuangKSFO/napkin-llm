[[34 Agent 部署与持久化执行|Agent 部署与持久化执行]]：把长跑 Agent 的运行状态、外部副作用标识和审批记录持久化，使 worker 崩溃后可由新 worker 恢复；可靠恢复依赖确定性编排、幂等边界与安全授权，而不是“把对话存起来”。

## 直觉：快递单、仓库账本与签收人缺一不可

普通 HTTP handler 像一次性口头交代：进程消失，正在做到哪一步也消失。持久化执行更像快递账本：每个包裹有稳定运单号、当前状态、已完成步骤与签收人；换了配送员也能继续。对 Agent，至少应分开保存三类状态：

- **编排状态**：`run_id`、计划版本、步骤输入/输出、重试与 checkpoint；
- **副作用状态**：稳定 `effect_id`、目标、幂等结果、补偿/失败状态；
- **治理状态**：权限范围、审批请求与审批人、审计时间线，以及预算上限、预留额度和已结算成本。

Temporal 的架构文档将 durable workflow 描述为基于追加历史重建状态，并要求 workflow 代码可确定性重放；Activity 位于副作用/重试边界，可能至少执行一次，因此应幂等或明确不可重试。[Temporal architecture，访问于 2026-07-17](https://github.com/temporalio/temporal/blob/main/docs/architecture/README.md) 已提交的 activity 完成结果可在 workflow history 中被重放逻辑**一次观察**（exactly-once observed），但这不等于外部动作或 Activity 本身只执行一次。这也解释了“有状态恢复”与“Agent 记忆”不同：`[[19 Agent 记忆系统|记忆]]` 帮模型理解任务，持久状态负责正确地恢复执行。

持久化 runtime 不会自动替任务设定或守住经济边界。启动昂贵步骤前，应把本轮的最大花费预留到持久账本；重放时复用同一预留，而不是在新 worker 中重新“批准”一次额度。预算超限、价格/配额不确定或授权已过期，都应是可恢复状态机的暂停条件。

## 小数字手算：为什么日志本身不能消灭重复发送

一次通知使用稳定键 $k=\texttt{notify-42}$。第一次尝试已经把邮件交给接收方，但在把“成功”写回本地日志前崩溃；重启后本地看到 $k$ 未完成，于是重试。

| 接收方能力 | 两次请求造成的可见邮件数 |
|---|---:|
| 不识别 $k$ | $1+1=2$ |
| 以 $k$ 去重 | $1$ |

因此“日志 + retry”最多说明本地何时该重试；**端到端一次效果**还要求接收方用同一幂等键去重，或使用具备等价原子保证的事务/outbox 设计。官方 durable-execution 指南同样明确：中断可能重跑步骤，幂等需要由步骤安全性保证。[AWS Idempotency 指南，访问于 2026-07-17](https://docs.aws.amazon.com/durable-execution/patterns/best-practices/idempotency/)

## 公式推导：可重放控制流与可去重副作用

令 $H_t$ 是已提交的历史，$S_t$ 是运行状态。可重放的编排逻辑应满足

$$
S_{t+1}=F(S_t,H_t),
$$

其中 $F$ 对相同输入与历史产生相同的下一步决定。时间、随机数、网络读取、文件系统和可变全局状态都不能裸放在 $F$ 中；它们要作为记录结果的 step/activity 边界。AWS 的官方说明把这些来源列为重放时的非确定性因素。[AWS Determinism 指南，访问于 2026-07-17](https://docs.aws.amazon.com/durable-execution/patterns/best-practices/determinism/)

对外动作可抽象为

$$
\operatorname{effect}(k,p)=
\begin{cases}
\operatorname{result}[k], & k\ \text{已被接收方或原子账本确认};\\
\operatorname{perform}(k,p), & \text{否则}.
\end{cases}
$$

没有任何 durable 平台或“持久化执行”类别能为任意外部动作笼统宣称“只执行一次”。历史中已提交完成结果的 replay 可做到 **exactly-once observed**，即 workflow 每次重放都读到同一份完成记录；但 Activity/worker 仍可能 at-least-once 执行。只有持久历史与重放、稳定幂等键、接收方/账本去重，以及明确的完成边界共同覆盖某个**具体操作**时，才可把外部副作用称为 **effectively-once**（一次效果）。

同样，预算也必须随状态持久化。若本轮预留为 $q_t$、实际结算为 $c_t\le q_t$、已花为 $C_t$、总上限为 $B$，则只有在 $C_t+q_t\le B$ 时启动，并在同一 `run_id` 下结算为 $C_{t+1}=C_t+c_t$；重放同一 `run_id` 必须读回既有预留，不能再扣一次额度。

## 图：恢复的是状态机，不是原来的进程

![[Agent 部署与持久化执行.png]]

![[Agent 部署与持久化执行-中断恢复架构.png]]

等待人审或 webhook 时，持久后端保存“等什么”和当前 checkpoint，worker 释放资源；恢复时必须再次检查授权、租约和输入版本，不能因为旧审批存在就无限期复用权限。

## 可运行代码：❌ 重试双发 vs ✅ 用稳定 effect_id 去重

```python
# Python 3 标准库；模拟“崩在确认写回前”的重复投递与可恢复预算预留。
delivered = []
ledger = {}
budget = {"limit": 5.0, "spent": 0.0, "reserved": {}}

def send_without_idempotency(recipient, body):
    delivered.append((recipient, body))  # ❌ 重试无法判断是否已发
    return "sent"

def send_once(effect_id, recipient, body):
    if effect_id in ledger:              # ✅ 恢复时复用已确认结果
        return ledger[effect_id]
    delivered.append((recipient, body))  # 接收方/原子账本也必须按 effect_id 去重
    ledger[effect_id] = "sent"
    return "sent"

def reserve_budget(run_id, maximum_cost):
    if run_id in budget["reserved"]:       # ✅ 重放复用原预留，不双扣预算
        return budget["reserved"][run_id]
    if budget["spent"] + sum(budget["reserved"].values()) + maximum_cost > budget["limit"]:
        raise RuntimeError("budget exceeded")
    budget["reserved"][run_id] = maximum_cost
    return maximum_cost

def settle_budget(run_id, actual_cost):
    maximum_cost = budget["reserved"].pop(run_id)
    assert 0.0 <= actual_cost <= maximum_cost
    budget["spent"] += actual_cost

send_without_idempotency("ops@example.test", "restart")
send_without_idempotency("ops@example.test", "restart")
assert len(delivered) == 2

delivered.clear()
reserve_budget("run-7", 2.0)
reserve_budget("run-7", 2.0)                # 模拟恢复后的重复预留
send_once("notify-42", "ops@example.test", "restart")
send_once("notify-42", "ops@example.test", "restart")  # 模拟恢复后的 retry
settle_budget("run-7", 1.5)
assert len(delivered) == 1
assert ledger["notify-42"] == "sent"
assert budget["spent"] == 1.5
print("deduplicated deliveries=", len(delivered), "spent=", budget["spent"])
```

这个最小例子只演示键的语义；真实系统还要处理“外部已成功、本地未落账”的窗口，因此要让外部 API 接受该键、或采用事务性 outbox/对账补偿，而不是只在进程内放一个字典。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**为什么持久化执行要求确定性？** 崩溃后会用已提交历史重放编排代码；若同样历史因 `now()`、随机数或直接 HTTP 调用走出新分支，就无法正确复原。将这些操作放在可记录的 activity/step 边界内。[AWS Determinism 指南，访问于 2026-07-17](https://docs.aws.amazon.com/durable-execution/patterns/best-practices/determinism/)

**重放、Activity 与外部动作各保证什么？** 已提交 completion 在 workflow history 中可被重放逻辑一次观察（exactly-once observed）；Activity/worker 仍可能至少执行一次；外部动作只有在具体边界上接好持久重放、稳定幂等键、接收方去重或原子事务及完成确认时，才可称 effectively-once。不能把三层语义都归因给平台类别。[AWS Idempotency 指南，访问于 2026-07-17](https://docs.aws.amazon.com/durable-execution/patterns/best-practices/idempotency/)

**为什么预算也要 durable？** 若额度只放在 worker 内存，崩溃/重放会丢失“已预留、未结算”的事实，造成重复扣额或越限。将预留和结算以 `run_id` 持久化，并在恢复时复用；超限、配额不明或授权过期时暂停给人处理。

**部署时最容易漏什么？** 把 checkpoint 当安全边界。还需要最小权限、密钥隔离、输入/工具审计、行动级审批、状态保留与删除策略；NIST AI RMF 1.0（2023）要求定义并记录人机协作中的角色、职责和监督流程。[NIST AI RMF 1.0, 2023](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf)

## 关键事实

- Durable execution 的价值是保存历史并在新 worker 上恢复进度；Temporal 文档以 event history 和确定性 workflow/activity 边界说明该机制。已提交 completion 是 workflow history 中的 exactly-once observed 结果；Activity 自身仍应按至少一次执行来设计。[Temporal architecture，访问于 2026-07-17](https://github.com/temporalio/temporal/blob/main/docs/architecture/README.md)
- 对外副作用只有在具体操作边界上才可能 effectively-once：持久/重放、幂等副作用、接收方去重与完成边界共同成立，不能归功于某个平台或类别本身；预算预留与结算也要与 `run_id` 一起持久化。
- 状态化恢复不是把所有上下文塞进 checkpoint：保存最小决策状态与工件引用，配合 `[[21 上下文压缩与卸载|上下文压缩]]`，并对敏感内容加密、最小留存与访问控制。
- `[[33 长程任务与自我改进(Ralph loop)|Ralph loop]]` 的 git 提交能保存工件级进度；生产长程任务若涉及外部副作用、等待审批或故障恢复，需要额外设计状态机、幂等边界和人类责任链。
