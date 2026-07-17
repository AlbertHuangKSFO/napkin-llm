[[40 Loop Engineering：可验证的 Agent 外循环|Loop Engineering]] 是一种**2026 年新兴的工程实践**：人不再逐轮提示 agent，而是设计一个有触发器、持久状态、独立验证器与明确边界的外循环，让 agent 反复推进，直到**可验证**地完成、暂停或安全停止。它不是既成的岗位头衔，也不取代 [[23 Agent Harness 概览|Agent Harness]] 或 prompt；它关心的是“谁在何时重启哪次工作、凭什么停止、出错后从哪里续跑”。

要先把四层分清：**Prompt** 是一次任务指令，通常由人决定下一轮；**Harness** 是一次 agent 运行里的工具、上下文、权限与日志运行时；[[33 长程任务与自我改进(Ralph loop)|Ralph loop]] 是“固定目标 + 外部状态 + 反复重启”的极简外层模式；**Loop Engineering** 则把触发、验收、恢复、审批和预算写成可执行的生命周期。裸 Ralph 可以只是 `while true`，不天然保证独立验证、停止条件或成本上限；成熟的 Loop Engineering 可以采用 Ralph 的重启思想，却不能只相信 agent 说“完成了”。

## 直觉：夜班组长不是替工人喊口号

把 agent 想成夜班维修工。Prompt 是你交给他的一张工单；Harness 是工具箱、门禁和工作台；Ralph 是每小时把同一张工单交给一位“刚上班”的维修工，并让他看交接本。Loop Engineering 则是夜班组长设计整套制度：有新故障才派工（触发），每位工人把现场、照片和未完成项写进交接本（持久状态），**另一位质检员**用仪表复测（独立验证），高风险修复等主管签字（人工审批），超过工时或预算就停机（成本边界）。

因此，关键不是“多跑几轮”，而是让每一轮都从真实证据得到下一步。对 [[03 Agent 核心循环|Agent 核心循环]] 而言，这是在其 observe → think → act 的**外面**再加一层生命周期控制；对 [[34 Agent 部署与持久化执行|持久化执行]] 而言，崩溃后读取检查点即可续跑，而不是依赖一段已经消失的对话历史。

## 小数字手算：12 个检查项怎样避免空转

设一个依赖升级任务有 $12$ 个验收检查，循环每轮最多花 $1.4$ 美元，总代价上限 $5.2$ 美元，连续 $2$ 轮检查通过数不增加就暂停。外层把每轮的 `passed`、失败日志、当前 git revision 与累计花费都写入状态文件；独立 CI 依次测得：

- 第 $1$ 轮：$4/12$ 项通过，累计 $\$1.4$；有进展，保存状态并继续。
- 第 $2$ 轮：$7/12$ 项通过，累计 $\$2.8$；有进展，继续。
- 第 $3$ 轮：$9/12$ 项通过，累计 $\$4.2$；有进展，但只余 $\$1.0$ 预算，下一轮必须更小心。
- 第 $4$ 轮：$12/12$ 项通过，累计 $\$5.6$，虽然测试绿了，却已超过 $\$5.2$ 的硬边界，结果应是 **STOP_BUDGET**，不是自动发布。

若把单轮费用降到 $1.2$ 美元，且在第 $3$ 轮达到 $12/12$，三轮总额为

$$
C_3=3\times\$1.2=\$3.6\le\$5.2,
$$

此时还需“CI 全绿 + 产物绑定到当前 revision + 人工批准发布”三者同时成立，才能进入 DONE。这个例子说明：**测试通过是证据的一部分，不能越过预算或审批边界。**

把低成本策略换成每轮 $\$1.2$ 后，演示循环在第 $3$ 轮得到 $12/12$、累计 $\$3.6$，先把 `verified_revision=r3` 与 `ready_for_release=true` 持久化为 `WAIT_APPROVAL`。随后的人审只把**这份已验证快照**从 `WAIT_APPROVAL` 转为 DONE：revision 仍是 `r3`、费用仍是 $\$3.6$，不会再唤起 agent、增加轮数或重新计费。若批准前工作区发生改变，它已是新候选，必须重新验证，不能沿用旧批准。

## 公式推导：把“完成”写成一个可判定状态

令第 $t$ 轮的持久状态为

$$
S_t=(W_t,E_t,C_t,R_t,H_t,Q_t),
$$

其中 $W_t$ 是工作区及其 revision，$E_t$ 是已保存的证据与失败日志，$C_t$ 是累计成本，$R_t$ 是连续无进展轮数，$H_t\in\{0,1\}$ 表示需要时是否得到人工批准，$Q_t$ 是 `ready_for_release` 与 `verified_revision` 组成的可发布快照。触发器 $\tau_t$ 可以是人工请求、定时器或外部事件；Harness 调用 agent 得到候选改动 $A_t$，但不把它的自然语言自评当作事实：

$$
(W_{t+1},E_{t+1})=\operatorname{persist}\bigl(S_t,\operatorname{agent}(\tau_t,S_t)\bigr),
$$

$$
v_{t+1}=\operatorname{verify}_{\perp}(W_{t+1},E_{t+1}).
$$

下标 $\perp$ 表示验证器与执行 agent 的“完成宣称”解耦：它可以是 CI、浏览器端到端测试、只读数据库核对，或带新上下文的 reviewer；至少要校验证据对应的 revision。若目标谓词为 $G$、总预算为 $B$、最大无进展轮数为 $K$，则

$$
\operatorname{DONE}_{t+1}=G(W_{t+1})\land(v_{t+1}=\mathrm{pass})\land(C_{t+1}\le B)\land H_{t+1}=1\land(W_{t+1}.\mathrm{revision}=Q_{t+1}.\mathrm{verified\_revision}).
$$

反之，$C_{t+1}>B$ 导致 `STOP_BUDGET`，$R_{t+1}\ge K$ 导致 `PAUSE_STALE`，需要发布但 $H_{t+1}=0$ 则 `WAIT_APPROVAL`。在 `WAIT_APPROVAL`，批准只能把**同一份** $Q_{t+1}$ 直接转成 DONE；若 revision 不再等于 `verified_revision`，状态应变为 `REVERIFY_REQUIRED`，然后把它当作新候选重走 agent + verifier。这样审批不会偷偷多跑一轮。这些非 DONE 终态也要落盘；恢复时从 $S_{t+1}$ 读取，而不是假装还记得上一段上下文。成本可直接记为 $C_n=\sum_{t=1}^{n}(c_t^{\text{agent}}+c_t^{\text{verify}})$，所以 [[35 Agent 成本与延迟优化|成本优化]] 的对象不只是模型，也包括重试次数和验证频率。

## 手绘图：外循环把“宣称完成”改成“证据过闸”

![[Loop Engineering-可验证外循环.png]]

图中的红色闸门刻意独立于 agent：即使模型输出“done”，它也只能形成候选结果；只有外部证据、费用、无进展探测与必要的人审共同允许，循环才会结束。每次失败或暂停都把证据写回状态，因此下一次触发能够恢复而非盲目重来。这也是 [[38 Agent 评估与可观测性|可观测性]] 从“事后看日志”升级为“运行时裁决”的位置。

## 可运行骨架：❌ 自报完成 vs ✅ 可恢复的验证外环

下面是只用 Python 标准库即可运行的教学骨架。`agent_step` 用确定性假数据模拟 agent；实际接入时替换为 Claude Code、Codex 或其他 CLI 调用，而 `independent_verify` 应替换为**独立** CI / 浏览器测试 / 只读核对。先运行 `python loop.py`，它会在验收通过后把 `r3` / $\$3.6$ 固化在审批闸门；检查这一版本后再运行 `APPROVE_RELEASE=1 python loop.py`，它只释放该快照，不会再调用 `agent_step` 或增加花费。

```python
# ❌ Prompt-only / Ralph-ish：agent 的“完成”宣称就是唯一停止条件。
def agent(prompt: str) -> str:
    return "done：我认为已经修好了"

while True:
    claim = agent("修好 12 个依赖升级检查项")
    if "done" in claim:       # 没有独立验证、状态、预算或审批
        break
```

```python
# ✅ loop.py：可直接运行的、证据闸门化外循环（Python 3.10+）。
from __future__ import annotations

import json
import os
from pathlib import Path

STATE = Path("loop-state.json")
TOTAL_CHECKS, BUDGET, MAX_STALE = 12, 5.2, 2

def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {
        "round": 0, "passed": 0, "spent": 0.0, "stale": 0,
        "revision": "r0", "evidence": [], "terminal": None,
        "ready_for_release": False, "verified_revision": None,
    }

def save(state: dict) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

def agent_step(state: dict) -> dict:
    """演示替身：生产中在沙箱内调用 agent，只返回候选改动而非 DONE。"""
    gain = [4, 3, 5][min(state["round"], 2)]
    return {"revision": f"r{state['round'] + 1}",
            "claimed_done": True, "candidate_passed": min(TOTAL_CHECKS, state["passed"] + gain),
            "cost": 1.2}

def independent_verify(candidate: dict) -> tuple[bool, str]:
    """演示独立闸门；真实实现应跑 CI/E2E 并绑定 candidate revision。"""
    passed = candidate["candidate_passed"]
    return passed == TOTAL_CHECKS, f"CI@{candidate['revision']}: {passed}/{TOTAL_CHECKS}"

def run() -> str:
    state = load()                              # 断点续跑：不依赖旧对话
    if state["terminal"] == "WAIT_APPROVAL":
        # 审批不是新一轮 agent 工作：只能释放已验证、未变更的那个 revision。
        if not state["ready_for_release"] or state["revision"] != state["verified_revision"]:
            state["terminal"] = "REVERIFY_REQUIRED"; save(state); return state["terminal"]
        if os.getenv("APPROVE_RELEASE") == "1":
            state["terminal"] = "DONE"; save(state); return state["terminal"]
        return "WAIT_APPROVAL"
    while state["round"] < 20:
        if state["spent"] >= BUDGET:
            state["terminal"] = "STOP_BUDGET"; save(state); return state["terminal"]
        candidate = agent_step(state)
        ok, evidence = independent_verify(candidate)  # 不信 claimed_done
        previous = state["passed"]
        state.update(round=state["round"] + 1, revision=candidate["revision"],
                     passed=candidate["candidate_passed"],
                     spent=round(state["spent"] + candidate["cost"], 2))
        state["evidence"].append(evidence)
        state["stale"] = state["stale"] + 1 if state["passed"] == previous else 0
        if state["stale"] >= MAX_STALE:
            state["terminal"] = "PAUSE_STALE"; save(state); return state["terminal"]
        if ok and state["spent"] <= BUDGET:
            # 先保存可发布快照；批准恢复时绝不再执行 agent_step / 计费。
            state.update(ready_for_release=True, verified_revision=state["revision"])
            if os.getenv("APPROVE_RELEASE") != "1":
                state["terminal"] = "WAIT_APPROVAL"; save(state); return state["terminal"]
            state["terminal"] = "DONE"; save(state); return state["terminal"]
        save(state)                             # 失败证据也必须持久化
    state["terminal"] = "STOP_ROUND_LIMIT"; save(state); return state["terminal"]

print(run())
```

`claimed_done=True` 在第 1 轮就出现，但独立验证器只在第 3 轮给出 `12/12`；这正是要防的“模型自评即验收”错误。此时状态已固化为 `revision=r3`、`spent=3.6`、`verified_revision=r3`、`ready_for_release=true`；第二次带批准的启动只把终态改为 DONE，四个值中的前三个不变。真实系统还应把 agent 的权限限定到隔离 worktree / 沙箱，release、外发消息、删除数据和付费动作一律由人审闸门控制。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：Loop Engineering 和 Prompt Engineering 的差别？**

A：Prompt Engineering 优化一轮中“对 agent 说什么”；Loop Engineering 优化多轮中“何时说、读什么状态、由谁验证、何时停或恢复”。前者仍然必要，后者不等于一个新职业；它是 2026 年围绕长程 agent 流程浮现的工程实践。

**Q：它和 Harness、Ralph loop 是包含关系吗？**

A：Harness 是单次运行的执行底座；Ralph 是反复启动 agent 的简洁外循环模式；Loop Engineering 是对外循环生命周期的设计要求。可以用 harness 执行每轮、用 Ralph 的“新会话 + 文件状态”技巧跨轮恢复，但还要补上独立 verifier、停止状态、成本和审批政策。

**Q：为什么“tests passed”还不足以自动上线？**

A：测试只证明已编码的检查项通过，不自动覆盖预算、权限、数据风险和业务批准。至少应绑定当前 revision、设轮次 / token / 美元上限，并在发布、支付、删除、生产写入等不可逆动作前等待人工批准。批准应只释放已验证的 revision；若审批前有任何改动，就把它视作新候选重新验证，不能在审批调用中顺手再跑 agent。

**Q：独立验证器必须是另一个模型吗？**

A：不是，确定性 CI、lint、E2E、指标阈值、只读数据库核对通常更强。若用 reviewer agent，应给它新上下文并把它的结论限制为证据输入，不能让同一执行轨迹的自评直接放行。

## 关键事实

- Anthropic Claude Code 团队在 **2026-06-30** 将 loop 定义为 agent 重复工作直到满足停止条件，并按触发方式、停止方式、所用 primitive 和任务类型分类；因此“循环”不是天然的无限重试。出处：[Anthropic, *Loop engineering: Getting started with loops* (2026-06-30)](https://claude.com/blog/getting-started-with-loops)。
- 该文的 goal-based loop 用“目标达成或达到最大轮数”停止，并指出确定性的测试数 / 分数阈值比主观“够好了”更适合作为退出条件；time-based 与 proactive loop 则分别由时间和事件 / 日程触发。出处同上。
- Loop Engineering 在此笔记中被严格标成**2026 年新兴工程实践**，而非成熟职业头衔或 Anthropic 的独立产品；它把 Prompt、Harness、[[33 长程任务与自我改进(Ralph loop)|Ralph loop]] 中已有的部件编排为可审计的外部生命周期。
- 最小合格规格：触发器、持久状态、与执行 agent 解耦的验证器、`verified_revision` 可发布快照、DONE / STOP / PAUSE / WAIT_APPROVAL 状态、恢复入口、权限与成本边界。审批恢复只能释放该快照；代码一变即 `REVERIFY_REQUIRED`。缺任何一项，都可能把“再跑一次”变成无证据的 token 炉子。
- 对长期任务，状态和证据要落在可恢复工件中（如受控状态文件、git revision、CI 报告与 trace），并由 [[38 Agent 评估与可观测性|评估与可观测性]] 记录；不能只存在于模型的会话上下文中。
