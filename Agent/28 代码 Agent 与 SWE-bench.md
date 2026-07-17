[[28 代码 Agent 与 SWE-bench|代码 Agent 与 SWE-bench]] 把“生成一段代码”扩展为受证据约束的工程循环：读 issue 与仓库、定位范围、修改最小补丁、运行目标测试、检查 diff，再决定继续还是交给人。SWE-bench 是用真实 GitHub issue 与仓库快照评测这种循环的基准；它不是“软件工程师替代率”。

## 直觉 / 生活类比

代码 Agent 像在陌生工厂修机器。报错文字只是故障单；要先看机器版本、零件关系、已有安全检测，再动手。只会把故障单“补全”为一段代码，既不知道改到了哪里，也不知道是否破坏别处。

SWE-bench 则像统一的修理考试：给定一个 issue、对应的历史仓库状态和隐藏评测，提交 patch 后由 harness 判定。它把结果变成可复现的分数，却仍只覆盖其数据集、提交时的依赖和测试口径。尤其要区分：**SWE-agent 是 2024 年提出的 agent scaffold/研究系统；SWE-bench 是 benchmark；某次 leaderboard 分数则是某模型加某 commit 加某配置在某日期、某 split 上的实验结果。**

## 小数字手算

假设一次提交在 `swe-bench_verified` 的 $500$ 个实例中解决 $320$ 个：

$$
\operatorname{resolve\ rate}=\frac{320}{500}=0.64=64\%
$$

此数字只有同时写出下列口径才有意义：数据集为 Verified、分母为 $500$、模型 ID、agent/harness commit、运行日期、超时/重试与评测 harness 版本。若换成原始全量集或换一版容器依赖，$64\%$ 不是可直接横比的“能力常数”。官方 CLI 将 Verified 标为 $500$ 个经验证问题；这个固定分母正是报告口径的一部分。[SWE-bench CLI](https://github.com/SWE-bench/sb-cli)

## 公式推导

把 instance 记为 $(r,i)$：$r$ 是仓库快照，$i$ 是 issue；agent 产出 patch $p$，评测器在受控环境运行隐藏测试 $T_{r,i}$：

$$
\operatorname{resolved}(r,i,p)=
\mathbf{1}\bigl[T_{r,i}(r\oplus p)=\operatorname{pass}\bigr]
$$

一次 run 的聚合分数为：

$$
\operatorname{score}=\frac{1}{|D|}\sum_{(r,i)\in D}\operatorname{resolved}(r,i,p_{r,i})
$$

这个推导也解释了三条工程约束：不得修改评测测试以伪造 $T$；不得把“成功生成 patch”当作 resolved；比较时必须固定 $D$ 与 evaluator。真实生产还多了需求歧义、跨服务协调、代码审查与上线责任，因此 benchmark 分数不能外推为通用工程产能。

## 手绘图

![[代码 Agent 与 SWE-bench-评测流程.png]]

## 可运行代码 / 配置

把以下文件存为 `repo_guard.py`，在 git 仓库中运行 `python3 repo_guard.py <base-ref>`。它没有调用模型，但演示代码 Agent 在执行测试前应当落实的可运行护栏：拒绝改测试文件、记录提交 SHA、让测试退出码决定结果。

```python
# repo_guard.py
import subprocess
import sys

BASE = sys.argv[1] if len(sys.argv) == 2 else "HEAD~1"

def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()

commit = run("git", "rev-parse", "HEAD")
changed = run("git", "diff", "--name-only", f"{BASE}...HEAD").splitlines()

# ❌ 只看“测试绿了”而允许改 tests，可能在奖励函数里作弊。
# subprocess.run(["pytest", "-q"], check=True)

# ✅ 先拒绝测试改动，再在确定的提交上运行测试。
for path in changed:
    if path.startswith(("tests/", "test/")) or path.endswith("_test.py"):
        raise SystemExit(f"拒绝评测：patch 修改了测试文件：{path}")

result = subprocess.run(["pytest", "-q"], text=True, check=False)
print({"commit": commit, "changed": changed, "pytest_exit": result.returncode})
raise SystemExit(result.returncode)
```

生产 agent 还应把 `issue_id`、基准 commit、模型快照、工具版本、最大步数/时间、完整 patch 与测试日志写入 trace。先用 `rg` 建索引、再精读相关文件、再跑最小测试，通常比把整个仓库塞进上下文更可控，见 [[24 Agentic Search：grep vs 向量检索|Agentic Search]]。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：SWE-agent 和 SWE-bench 是什么关系？**

答：SWE-bench 是基准与评测 harness；SWE-agent 是为“读 issue—改仓库—跑测试”设计的 2024 年 agent 系统。可以用 SWE-bench 评测 SWE-agent，也可以评测其他 agent；两者不能互作代名词。

**Q：如何正确引用一个 SWE-bench 比较结果？**

答：至少给出数据集/split、模型与版本、agent/harness 的 commit 或配置、评测日期和 evaluator 版本；若有预算/重试/超时差异也需披露。没有这些字段的“某 agent XX%”不可公平比较。

**Q：为什么隐藏测试仍不等于生产验证？**

答：它只验证 benchmark 定义的期望行为。生产还需审查安全、性能、迁移、回滚、需求与跨团队影响；因此把测试通过当作发布许可是错误的。

## 关键事实

- **SWE-bench 论文（ICLR 2024）**以真实 GitHub issue 与仓库任务构成评测；原始数据集的规模和 instance 构造以论文/官方仓库为准。[论文](https://arxiv.org/abs/2310.06770)｜[官方仓库](https://github.com/SWE-bench/SWE-bench)
- **SWE-agent 论文发表于 2024（NeurIPS）**，重点是面向软件工程任务的 Agent-Computer Interface 与运行时 scaffold；它不是后来所有代码 agent 的通用实现。[论文](https://arxiv.org/abs/2405.15793)｜[官方仓库](https://github.com/SWE-agent/SWE-agent)
- **Verified 的 $500$ 个问题是数据集口径**，而非任何模型的固定成绩；官方 CLI 文档将其列作 `swe-bench_verified`。[来源](https://github.com/SWE-bench/sb-cli)
- 代码 agent 的运行时护栏、工具与上下文管理见 [[23 Agent Harness 概览|Agent Harness]]、[[20 上下文工程|上下文工程]]；网页/GUI 自动化边界见 [[27 计算机使用与浏览器 Agent|计算机使用与浏览器 Agent]]。
