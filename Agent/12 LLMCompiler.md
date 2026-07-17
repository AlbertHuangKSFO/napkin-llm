[[12 LLMCompiler|LLMCompiler]] 把一组函数调用表示为带变量依赖的执行计划：没有依赖的调用可同时发出，有依赖的调用等输入就绪后再执行。它借用编译器的「分析依赖—调度执行」思想来减少工具调用的关键路径延迟。

它与 [[11 ReWOO|ReWOO]] 都先生成变量化计划，但原论文强调的是 parallel function calling；它不是任何工具链都必然更便宜或更快的万能调度器。

## 直觉：厨房不是只能一口锅

「洗菜」和「预热烤箱」可以同时做，「把菜放进烤箱」必须等洗菜完成。若把所有动作排成单队列，空闲资源被浪费；若忽略依赖就并发，又会在没有菜时执行后续动作。DAG 正是在「尽可能并行」和「必须先后」之间写下可检查的约束。

## 小数字手算：看关键路径，不只看节点数

设任务 A、B 无依赖，分别耗时 2 秒和 3 秒；任务 C 依赖 A、B，耗时 1 秒。

串行排队为 $2+3+1=6$ 秒。资源足够时，A 与 B 同发，先等较慢的 B：

$$
L_{\mathrm{DAG}}=\max(2,3)+1=4\text{ 秒}.
$$

速度比为 $6/4=1.5$。即使有两个并发任务，C 仍是等待点；因此「有并行」不等于「节点数倍加速」。这是调度算例，不是论文基准。

## 公式推导：延迟下界是最长依赖链

对任务图 $G=(V,E)$，节点 $v$ 的服务时间为 $d(v)$。任一有向路径 $p$ 上的任务都不能重叠，因此总完成时间满足：

$$
L\ge \max_{p\in\mathrm{paths}(G)}\sum_{v\in p}d(v).
$$

在资源无限、调度和工具无额外开销的理想条件下，DAG 的完工时间等于这个**关键路径**长度。实际系统还需计入模型生成计划、排队、限流、重试和网络延迟，所以只应把它当作设计下界。

## 机制：论文的三个组件

![[LLMCompiler.png]]

1. **Function Calling Planner**：生成函数调用计划，并用变量引用表示哪些输入来自上游调用。
2. **Task Fetching Unit**：持续读取计划，解析已满足的依赖，把就绪任务交给执行层；这使规划可以与部分执行重叠。
3. **Executor**：并行执行已就绪的函数调用并写回结果，供依赖它们的任务继续。

![[LLMCompiler-流水线.png]]

应用可以在全部任务完成后增加汇总或重规划节点，但那是编排选择；不应把它误写成论文声称的第四个固定核心组件。

## 原论文与可核验结论

Kim, Moon, Tabrizi, Lee, Mahoney, Keutzer 与 Gholami 的论文发表于 ICML 2024。论文在其多类函数调用任务中报告，相比所用 ReAct 基线，最高可达 **3.7× latency speedup**、**6.7× cost saving** 与约 **9% accuracy improvement**。这些是「up to」的实验观察，不是不同 provider、价格、并发限制下可承诺的工程指标。

- 论文（2023 arXiv，ICML 2024）：https://arxiv.org/abs/2312.04511
- 论文代码链接：https://github.com/SqueezeAILab/LLMCompiler

## 可运行的最小调度骨架

```python
from concurrent.futures import ThreadPoolExecutor

tasks = {
    "A": {"deps": set(), "run": lambda _: 2},
    "B": {"deps": set(), "run": lambda _: 3},
    "C": {"deps": {"A", "B"}, "run": lambda x: x["A"] + x["B"]},
}

# ❌ 朴素串行：即使 A、B 无依赖，也固定排队执行。
def run_serial(tasks):
    done = {}
    for name in ("A", "B", "C"):
        done[name] = tasks[name]["run"](done)
    return done

assert run_serial(tasks)["C"] == 5

# ✅ 改进：只要依赖满足就进入同一批；A、B 可由不同 worker 同时执行。
done = {}
while len(done) < len(tasks):
    ready = {k: v for k, v in tasks.items()
             if k not in done and v["deps"] <= done.keys()}
    if not ready:
        raise ValueError("依赖图有环或引用了不存在的任务")
    with ThreadPoolExecutor(max_workers=len(ready)) as pool:
        futures = {name: pool.submit(spec["run"], done) for name, spec in ready.items()}
        done.update({name: future.result() for name, future in futures.items()})

assert done["C"] == 5
```

示例只说明依赖屏障；真实 Executor 还需要每个工具的超时、并发配额、幂等键、结构化错误与追踪记录。

## 对比：限定在执行计划层

| 维度 | [[09 ReAct\|ReAct]] | [[11 ReWOO\|ReWOO]] | LLMCompiler |
|---|---|---|---|
| 计划粒度 | 通常逐轮决定下一动作 | 变量化的完整计划 | 函数调用及其数据依赖 |
| 主要反馈方式 | 观测进入下一轮 | 末尾统一合成 | 完成的节点解锁下游节点 |
| 并行来源 | 取决于实现 | 无依赖证据可以并发 | 调度器显式寻找就绪节点 |
| 核心风险 | 历史与往返增长 | 盲规划 | 图解析、关键路径与外部限流 |

这张表描述常见架构取向，不声明任何一种在准确率、成本或延迟上必然优于另一种。

## 何时用 / 边界

适合：有多个独立工具调用、可明确声明输入输出依赖、并且端到端延迟受工具往返影响的流程。

不适合：图几乎是一条链、工具有不可逆副作用却没有幂等或补偿机制、或 provider 限流使并发反而放大失败的流程。

- 对 Planner 产物做 schema、变量引用和环检测；一张错误 DAG 会并行地放大错误。
- 以关键路径、p95 延迟、调用量和成功率共同评估；只看平均并发数会掩盖瓶颈。
- 先为读操作并行；写操作需要幂等键、顺序约束或补偿事务。

## 面试高频

**Q：LLMCompiler 为什么能降延迟？**

答：它把函数调用依赖显式化，Task Fetching Unit 在输入就绪时立刻派发，独立节点可并行。延迟下界由最长依赖链决定；强串行任务并不会因为画成 DAG 就加速。

**Q：它与 ReWOO 的区别？**

答：二者都可使用变量化计划。ReWOO 的论文重点是避免把观测反复放回推理提示词；LLMCompiler 的论文重点是并行调度函数调用。真实系统可组合两种思想，但组合后的指标必须重新测量。

## 关键事实

- 原论文为 Kim et al. 的 *An LLM Compiler for Parallel Function Calling*（ICML 2024；Kim et al., 2023, [arXiv:2312.04511](https://arxiv.org/abs/2312.04511)）。
- 核心组件是 Planner、Task Fetching Unit、Executor；任务并发的前提是数据依赖已满足（Kim et al., 2023, [arXiv:2312.04511](https://arxiv.org/abs/2312.04511)）。
- 论文的 3.7×、6.7×、约 9% 都是特定实验中的最高/报告值，应连同比较基线和任务语境引用（Kim et al., 2023, [arXiv:2312.04511](https://arxiv.org/abs/2312.04511)）。
- [[06 Parallelization|Parallelization]] 解释并行本身；这里额外解决的是如何从模型生成的计划中恢复可安全执行的依赖图（Kim et al., 2023, [arXiv:2312.04511](https://arxiv.org/abs/2312.04511)）。
