[[33 长程任务与自我改进(Ralph loop)|长程任务与自我改进（Ralph loop）]]：Ralph loop 是反复唤起编码 Agent、从持久工件读取状态并只推进一小步的实践性模式；它不是控制系统的“环路工程”，更不是保证收敛的训练算法。

## 直觉：交接班，而不是让一个人永不下班

把长任务交给连续不断的短班次：每班都重新阅读目标、待办、上轮测试结果和当前 diff，只做一件可验收的事，然后把证据写回工件。下一班不依赖上一班的对话记忆，而依赖可审查的状态。Huntley 的原始博文发表于 **2025-07-14**，其中把 Ralph 的最纯形态写成 Bash 重启循环；晚于该文的社区回顾将其描述为在 **2025 年后期**走红。这是技术传播史，不是效果或采用率证据。[Huntley, 2025-07-14](https://ghuntley.com/ralph/)；[Well Engineered Tech, 2026-06-01](https://wellengineered.tech/en/blog/ralph-wiggum-claude-code/)

这里的“自我改进”只指**下一轮基于失败证据修正工件或计划**，不表示模型在更新权重；后者属于 `[[32 Agentic RL 与训练|Agentic RL]]`。`[[13 Reflection 与 Reflexion|Reflection]]` 可以是单轮内的反馈，Ralph 把状态和反馈移到轮次之间。

**Ralph 不是 `[[40 Loop Engineering：可验证的 Agent 外循环|Loop Engineering]]` 的同义词。** 前者是一个刻意极简的重启/文件记忆/测试反压模式；后者是更宽的运行系统设计，包含验证器与收敛判据、调度、隔离、持久状态、预算和人工闸门。可以用 Loop Engineering 给 Ralph 加护栏，也可以在没有 Ralph 重启模式时设计外循环。

| 维度 | Ralph loop（最小模式） | Loop Engineering（更宽的设计范围） |
|---|---|---|
| 轮次 | 重启短会话，读仓库工件 | 可重启、常驻或事件驱动的任意外循环 |
| 状态 | 文件、TODO、git、测试报告 | 外加 durable state、租约、审计与恢复语义 |
| 反压 | 通常是测试/build/lint | 可组合 verifier、收敛与无进展判据 |
| 治理 | 原法并不自动提供 | 显式调度、沙箱、预算和人审责任 |

## 小数字手算：预算先于下一轮启动

总预算 $B=12$，已完成三轮的实际花费为 $[2.5,2.5,2.5]$，则

$$
C_3=2.5+2.5+2.5=7.5,\qquad B-C_3=4.5.
$$

若下一轮的预留上限是 $5.0$，则 $C_3+5.0=12.5>B$，**不能启动**。这比“先跑、超了再停”多一道实际约束。即使待办清空，也必须独立运行验收；`TODO.md` 是状态，不是完成证明。

## 公式推导：继续条件必须同时包含验证、预算与人工责任

令 $V_t$ 为独立验收是否通过，$C_t$ 为已花费预算，$\hat c_{t+1}$ 为下一轮预留，$N$ 为轮数上限，$s_t$ 为连续无进展轮数，$K$ 为其阈值，$P_t$ 为当前动作是否获授权。一个保守的继续条件是

$$
\operatorname{continue}_t=
\neg V_t\ \land\ (C_t+\hat c_{t+1}\le B)\ \land\ (t<N)\ \land\ (s_t<K)\ \land\ P_t.
$$

其中 $V_t$ 应来自测试、lint、可复现实验、独立评审或其组合，而非 Agent 的“已完成”文本；若动作涉及发布、删除、付费、权限或外部通信，$P_t$ 必须由明确的人审/策略授权决定。NIST AI RMF 1.0（2023）要求把人类监督的角色、责任与流程定义并记录，正好对应这里不可外包的 $P_t$。[NIST AI RMF 1.0, 2023](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf)

因此 Ralph 的状态转移是 $A_{t+1}=\operatorname{update}(A_t,\text{evidence}_t)$；没有可靠 evidence，重复只会放大错误，而不会形成“自动收敛”。

## 图：状态必须落在会话之外

![[长程任务与自我改进(Ralph loop).png]]

![[长程任务与自我改进(Ralph loop)-checkpoint恢复.png]]

图中的 checkpoint 可以是 git 提交与测试报告，但它不等价于生产级的副作用恢复；涉及外部动作时还要看 `[[34 Agent 部署与持久化执行|持久化执行]]` 的幂等与审计边界。

## 可运行代码：❌ 无限重试 vs ✅ 有验证、预算、停滞与人审的 harness

```python
# Python 3 标准库；以预先给定的“轮次结果”模拟运行，不调用外部 Agent。
rounds = [
    {"cost": 2.0, "progress": True,  "verified": False, "allowed": True},
    {"cost": 2.0, "progress": False, "verified": False, "allowed": True},
    {"cost": 2.0, "progress": False, "verified": False, "allowed": False},
]

def guarded_ralph(events, budget=8.0, max_rounds=5, stale_limit=2):
    spent = stale = 0
    for number, event in enumerate(events, start=1):
        # ❌ while True: run_agent()  # 没有停止、预算、验证或责任边界
        if number > max_rounds or spent + event["cost"] > budget:
            return "stopped: limit"
        if not event["allowed"]:             # ✅ 高风险/未授权动作转人工
            return "paused: human approval"
        spent += event["cost"]
        stale = 0 if event["progress"] else stale + 1
        if event["verified"]:                # ✅ 独立验证，而非自报完成
            return "completed: verified"
        if stale >= stale_limit:
            return "stopped: no progress"
    return "stopped: no more scheduled rounds"

result = guarded_ralph(rounds)
assert result == "paused: human approval"
print(result)
```

真正的 harness 还应把每轮目标、输入工件 hash、测试报告、成本、授权人和停止原因落入不可随意覆盖的审计记录；在专用分支/沙箱中运行，绝不把“自动重试”直接接到生产写操作。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Ralph loop 的核心是什么？** 新会话/新进程可以反复读取外部状态并推进小步；核心资产是验收证据与持久工件，不是无限 `while`。它是实践模式，不是已证明收敛的算法；Huntley 的原始博文日期为 2025-07-14。[Huntley, 2025-07-14](https://ghuntley.com/ralph/)

**和 Loop Engineering 有何区别？** Ralph 是最小的“重启 + 文件记忆 + 测试反压”模式；Loop Engineering 还要定义 verifier、收敛/停滞、调度隔离、durable state、预算和人工闸门。两者可组合但不等价，见 `[[40 Loop Engineering：可验证的 Agent 外循环|Loop Engineering]]`。

**什么时候可以无人值守？** 仅限动作可逆、权限受限、验收可自动化且预算/轮数/停滞阈值明确的范围。发布、删改数据、付款、凭据访问和对外沟通都应暂停给具名责任人，而不是由“循环继续”隐式授权。[NIST AI RMF 1.0, 2023](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf)

**为什么测试全绿仍不够？** 测试是一个证据源，可能覆盖不足或与业务验收不一致；需要把测试、变更审查和适用的人审等级组合为 $V_t$。

## 关键事实

- Ralph 是编码 Agent 社区中的实践性循环方法：Huntley 于 2025-07-14 把其最纯形态写作 Bash 循环；2025 年后期的“走红”是社区传播现象，不是可靠性或经济性结论。[Huntley, 2025-07-14](https://ghuntley.com/ralph/)；[Well Engineered Tech, 2026-06-01](https://wellengineered.tech/en/blog/ralph-wiggum-claude-code/)
- 停止、验证、预算和人审是四个不同责任：任何一项缺失都可能让“持续尝试”变成失控执行；AI RMF 1.0（2023）可作为记录角色与监督流程的治理参考。[NIST AI RMF 1.0, 2023](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf)
- git checkpoint 只保存代码/工件层进度；跨网络副作用、恢复语义和审批等待应交给 `[[34 Agent 部署与持久化执行|有状态的持久化执行]]` 设计处理。
