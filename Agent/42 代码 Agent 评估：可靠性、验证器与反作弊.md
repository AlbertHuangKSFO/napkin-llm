[[42 代码 Agent 评估：可靠性、验证器与反作弊|代码 Agent 评估]] 的本质是：把「模型说自己修好了」变成一个可重复的实验——对**未泄漏的任务**，在独立、干净的环境中多次运行，用验证器检查代码造成的**最终环境状态**，并记录足以复核的外部轨迹。它把 [[28 代码 Agent 与 SWE-bench|代码 Agent]] 的「测试变绿」升级为：修对了、没改坏、没钻评测空子，而且值得为这份可靠性付出成本。

## 直觉：驾照路考不是背路线

把代码 Agent 想成参加驾照路考的人。只让他在练车场绕一圈、并由他自己举起「PASS」牌，等于只评最终一句话；他甚至可以把计时器关掉或改评分表。可靠的路考会：

- 把**正式路线**留作 held-out（留出集），不让考生事先按题背答案；
- 每次从同一辆初始车辆出发，连续考多次，因为一次顺利可能只是运气；
- 用摄像头、里程表和事故记录核验**车最后停在哪里、有没有撞到护栏**，而非相信「我已安全到达」；
- 只允许正常驾驶，不许改交通灯、偷看下一轮录像、把考官的评分单改成满分。

对应到代码任务：`task` 是「给仓库与 issue、说明成功条件」；一次 `trial` 是 agent 从全新 worktree 出发的一次尝试；`outcome` 是 patch 后数据库、文件、服务与测试的终态；`grader` 是断言这些终态的程序；`eval harness` 则负责建环境、并发执行、收集证据、评分和聚合。评测的对象不是裸模型，而是「模型 + [[23 Agent Harness 概览|Agent harness]]」这个整体。

⚠️ **可审计轨迹不等于 raw CoT。**前者是外部可复核的输入、工具调用、命令、文件 diff、标准输出、测试日志、耗时与资源用量，可用来定位失败；原始思维链既不是验证正确性的必要证据，也不该被索取、存储或拿来当唯一评分依据。应审计可观察行为与结果，而不是要求模型暴露私有推理文本。

## 小数字手算：一次「能过」与稳定「总能过」差多远？

同一个 held-out bug 在相同预算下跑 5 次，测试与终态验证的结果为：

$$
(o_1,o_2,o_3,o_4,o_5)=(1,1,0,1,0)
$$

其中 $1$ 是所有硬验证都通过、$0$ 是至少一项失败。

1. 单次成功率估计：$\hat p=(1+1+0+1+0)/5=3/5=0.60$，即 **60%**。
2. 若产品可自动挑选 5 次中的任一合格 patch，观察到的 $pass@5=1$（这 5 次至少有一次成功）。把每次近似看成独立、成功概率仍为 $0.6$，则预计为 $1-(1-0.6)^5=1-0.4^5=0.98976$，约 **98.98%**。
3. 若每次都必须可靠地通过，同一假设下 $pass^5=0.6^5=0.07776$，只有 **7.78%**。所以不能拿「多采样后总有一次成功」冒充面向用户的稳定性。

再看质量—成本：配置 A 为 $(可靠性=0.60,\$0.08)$，B 为 $(0.78,\$0.16)$，C 为 $(0.75,\$0.22)$，D 为 $(0.82,\$0.45)$。C 比 B **更贵且更不可靠**，被 B 支配；A、B、D 才在 Pareto 前沿。报告应保留前沿与 p95 延迟，而不是把质量和美元粗暴加成一个神秘总分。

## 公式推导：从任务到可信指标

令 $t\in\mathcal T$ 为任务，$r\in\{1,\ldots,m\}$ 为试验。harness 从隔离初态 $E_0(t)$ 启动 agent，得到终态 $E_{t,r}$、可审计轨迹 $\tau_{t,r}$、成本 $c_{t,r}$ 与时延 $\ell_{t,r}$。硬结果由多个 grader 合取：

$$
o_{t,r}=g_{\mathrm{patch}}\!(E_{t,r})\land
g_{\mathrm{held\text{-}out}}\!(E_{t,r})\land
g_{\mathrm{regression}}\!(E_{t,r})\land
g_{\mathrm{integrity}}\!(\tau_{t,r})\in\{0,1\}
$$

$g_{\mathrm{patch}}$ 检查 patch 能应用、构建能运行；$g_{\mathrm{held\text{-}out}}$ 检查未暴露的目标行为；$g_{\mathrm{regression}}$ 守住原先正确的行为；$g_{\mathrm{integrity}}$ 检查只改允许路径、测试没有被改、没有利用残留状态或禁止通道。它们一起回答「真的解决了吗」，而不是「agent 是否输出了 `PASS`」。

全套任务的单次可靠性与平均成本为：

$$
\hat R=\frac{1}{|\mathcal T|m}\sum_{t\in\mathcal T}\sum_{r=1}^{m}o_{t,r},\qquad
\bar C=\frac{1}{|\mathcal T|m}\sum_{t,r}c_{t,r}
$$

对单一任务，$pass@k$ 衡量 $k$ 次里至少一次成功，$pass^k$ 衡量 $k$ 次全成功：

$$
pass@k=1-\prod_{r=1}^{k}(1-o_{t,r}),\qquad
pass^k=\prod_{r=1}^{k}o_{t,r}
$$

实践中还应报任务分层、置信区间、$p95(\ell)$、每成功任务成本 $\bar C/\hat R$。若配置 $a$ 满足 $R_a\ge R_b$ 且 $C_a\le C_b$，并至少一项严格更好，则 $a$ 支配 $b$；只比较不被任何配置支配的 Pareto 前沿。对 customer-facing 代码 Agent，优先关注 $pass^k$；对允许并行候选 patch 再验证的离线修复，$pass@k$ 也有价值。

## 手绘图：可靠性来自终态与隔离，而非一句「通过」

![[代码 Agent 评估-验证与反作弊.png]]

图中的红色防线强调四件事：任务与隐藏验证分离；每 trial 从 clean worktree 开始；测试、评分器和 Git 历史不向 agent 泄漏；评分器读取 patch 后的终态。`trace` 仍值得保留，用于复盘「失败是 agent、harness 还是 grader 的问题」，但不应把某一条预设工具序列当作唯一正确路径——能达成同一安全终态的多种路线都应允许。

## 可运行代码：❌ 只看可见 PASS 标记 vs ✅ 留出终态验证 + 反作弊

下面是一个不依赖第三方库的最小演示。❌ `benchmark_only` 让候选人改可见测试后即可得分；✅ `robust_grade` 只允许改 `src/`，每次重建干净状态，并在 agent 看不到的断言中检查退款记录这个**最终环境状态**。

```python
from copy import deepcopy

BASE = {
    "src/refund.py": """def approve(db, user, amount):
    return 'OK'  # 缺陷：只说 OK，没有真的写入退款记录
""",
    "tests/visible.py": """assert approve({}, 'u1', 100) == 'OK'""",
}

CHEAT = {  # ❌ agent 不修代码，只把可见测试改成永远通过
    **BASE,
    "tests/visible.py": "assert True",
}

FIX = {  # ✅ 只改允许的源文件，真正改变系统状态
    **BASE,
    "src/refund.py": """def approve(db, user, amount):
    if amount <= 0 or not user:
        raise ValueError('invalid refund')
    db.setdefault('refunds', []).append(
        {'user': user, 'amount': amount, 'status': 'paid'}
    )
    return 'OK'
""",
}

def load_approve(files):
    namespace = {}
    exec(files["src/refund.py"], namespace)  # 模拟在干净 worktree 导入候选 patch
    return namespace["approve"]

def benchmark_only(files):                    # ❌ 评分器信任候选人能改的可见测试
    namespace = {"approve": load_approve(files)}
    exec(files["tests/visible.py"], namespace)
    return True

def robust_grade(files):                       # ✅ 每次 trial 都从 BASE 的新副本开始
    forbidden = [p for p in files if p.startswith("tests/") and files[p] != BASE[p]]
    if forbidden:                               # 测试文件只读：先挡住「改题目」
        return False, f"作弊：改了 {forbidden}"
    db = deepcopy({"refunds": []})             # 无上一轮残留的最终状态
    approve = load_approve(files)
    try:
        result = approve(db, "u1", 100)
        approve(db, "", 100)                   # held-out：负例也必须被拒绝
        return False, "缺少输入校验"
    except ValueError:
        pass
    expected = [{"user": "u1", "amount": 100, "status": "paid"}]
    return (result == "OK" and db["refunds"] == expected), db

print("❌ benchmark-only:", benchmark_only(CHEAT))  # True：假阳性
print("✅ CHEAT:", robust_grade(CHEAT))              # (False, "作弊：...")
print("✅ FIX:", robust_grade(FIX))                  # (True, {'refunds': [...]})
```

真实 harness 要把这个模式扩大：每任务单独容器或 VM、一次性凭证与最小网络权限、只读测试/评分器、diff allowlist、移除 `.git` 历史与旧日志、超时/资源上限，以及对产物的静态分析和隐藏回归测试。**隐藏不等于安全本身**：还要让通过测试在语义上确实要求完成任务，避免空实现、硬编码输入或利用评分器漏洞。每个任务都应有能通过所有 grader 的 reference solution；若许多 trial 都是 0 分，先审任务、环境和 grader，而不要仓促断言 agent 无能。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q1：为什么代码 Agent 不能只报 `pass@1`？**

`pass@1` 只描述一次尝试的成功，不说明是否稳定，也不说明成本。对同一 held-out 任务多 trial，至少同时报单次可靠性、$pass^k$、失败分布、$p95$ 时延与每成功任务成本；可并行取最优 patch 的批处理场景再额外报 $pass@k$。把两者混用会把「偶尔灵光」误说成「用户每次可靠」。

**Q2：`task`、`trial`、`outcome`、`grader` 和 `eval harness` 分别是什么？**

task 定义输入和成功条件；trial 是对该 task 的一次完整独立运行；outcome 是试验结束时环境的客观终态；grader 是检测该终态或可观察轨迹的逻辑；harness 是创建隔离环境、提供工具、运行/记录/评分/汇总的基础设施。模型与 agent harness 一起才构成被评对象。

**Q3：反作弊为什么不只靠「隐藏测试」？**

隐藏测试防止按可见断言硬编码，却不能防改测试文件、利用 Git 历史泄漏、污染后续 trial、读取评分器、伪造输出或利用权限过大的工具。需要 clean worktree、独立 trial、最小权限、只读 grader、diff allowlist、终态校验及人工抽查可审计轨迹。也要记住：隐藏评测的断言若太弱，仍可能奖励空实现。

**Q4：为什么评轨迹却不评分 raw CoT？**

轨迹用于审计工具使用、文件变更、命令结果、预算和失败归因；它可以是结构化、可复现的外部证据。raw CoT 既非验证结果的可靠替代，也不该成为强制产物。评分优先看安全、正确的最终状态；必要时只对可观察行为设软约束，避免把某种预设操作顺序误当成唯一正确解。

**Q5：怎样做质量—成本选型？**

在同一 held-out 任务集、同一环境和相同预算定义下画 $(可靠性, 成本)$，剔除被支配的配置，再按产品约束在 Pareto 前沿取点。高风险自动合并偏向高 $pass^k$ 与严格验证；低风险离线修复可先用便宜配置生成候选，再由独立 verifier 选出合格 patch。

## 关键事实

- **Anthropic，2026-01-09**：其 Agent eval 指南将 task、trial、grader、transcript、outcome 与 eval harness 分开定义；明确指出 outcome 是 trial 结束时的环境状态，不是 agent 的口头声称。代码 Agent 可组合单元测试、静态分析、状态检查与工具调用/预算指标；多次 trial 用 $pass@k$ 与 $pass^k$ 区分「有一次成功」和「次次成功」。来源：[Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)。
- **Anthropic，2026-01-09**：每个 trial 应从干净隔离环境开始，残留文件、缓存或 Git 历史会破坏独立性并可能虚增成绩；grader 应抵抗绕过与漏洞，并应阅读大量可审计 transcript 来区分 agent 失败与评测失真。来源：[同一指南的 harness 与 grader 章节](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)。
- **Jimenez et al.，ICLR 2024；SWE-bench 官方项目**：SWE-bench 由真实 GitHub issue 与对应代码库构成，复现评测使用 Docker；其 `TestSpec` 明确包含 `FAIL_TO_PASS` 与 `PASS_TO_PASS`，后者用于防止 patch 修复目标问题时破坏原有行为。SWE-bench Verified 是 500 个经软件工程师确认可解的子集。主要来源：[SWE-bench 论文（ICLR 2024）](https://openreview.net/forum?id=VTF8yNQM66)；实现细节补充见 [SWE-bench repository](https://github.com/SWE-bench/SWE-bench) 与 [SWE-bench harness API](https://www.swebench.com/SWE-bench/api/harness/)。
- **边界条件**：SWE-bench 的隐藏/目标测试不是「绝对正确性证明」。官方 issue 讨论也显示，局部执行的测试可能漏掉未执行的开发者测试；因此自建评测应加入更广的回归、产物/状态检查与人工抽检，而不是只追一个 leaderboard 分数。来源：[SWE-bench issue #280](https://github.com/SWE-bench/SWE-bench/issues/280)。
