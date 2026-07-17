[[35 Agent 成本与延迟优化|Agent 成本与延迟优化]] 的本质是：把一个任务拆成多次模型、工具与状态读写后，优化对象不再是“某次调用的单价”，而是**完成一个合格任务的期望成本、关键路径延迟与失败风险**。[[05 Routing|Routing]] 决定每一步用什么能力，[[06 Parallelization|Parallelization]] 决定哪些等待能重叠，[[21 上下文压缩与卸载|上下文压缩]] 与缓存决定每步带多少上下文；三者都必须由 [[38 Agent 评估与可观测性|结果与轨迹评估]] 约束。

## 直觉：像急诊分诊，而不是全员进 ICU

把每个请求都交给最强、最慢、最贵的模型，像把感冒和心脏骤停都送进 ICU：表面上“保险”，总体却更慢、更贵，还挤占真正难题的容量。合理做法是先设**质量下限与升级条件**：可验证的抽取、分类、格式化先走小模型或规则；不确定、长程、失败重试的步骤升级；有依赖的步骤串行，无依赖的工具调用并行。

“便宜”不是目标。若便宜模型把原本一次能完成的任务做成两次重试，单位成功任务成本反而上升。因此路由器要对失败兜底，并按任务结果校准，而不是只按输入长度猜难度。

## 小数字手算：看单位成功任务，而不是单次价格

某任务有两个步骤：抽取走小模型，输入/输出分别为 $2{,}000/200$ token；最终判断走强模型，分别为 $4{,}000/400$ token。假设小模型价格为输入 $\$0.20$/M、输出 $\$0.80$/M，强模型价格为输入 $\$2$/M、输出 $\$8$/M；工具固定花费 $\$0.001$。

$$
C=2000\cdot0.20/10^6+200\cdot0.80/10^6+4000\cdot2/10^6+400\cdot8/10^6+0.001
=\$0.01276
$$

若这条路的端到端成功率为 $p=0.92$，失败后独立重跑直到成功，则期望尝试次数为 $1/p$，单位成功任务的期望花费为：

$$
C_{\text{success}}=C/p=0.01276/0.92\approx\$0.01387
$$

若“全用强模型”令单次成本变成 $\$0.020$，但成功率为 $0.98$，则 $0.020/0.98\approx\$0.02041$。这个例子说明：先量**质量门槛后的总账**，再比较路由；数字只是演示，生产应读取已锁定模型版本和当前价表。

## 公式推导：成本是求和，延迟是关键路径

第 $i$ 个模型调用的输入、输出、缓存命中 token 分别为 $x_i,y_i,h_i$；它可路由到不同 provider/模型，因此输入、缓存、输出的逐调用单价分别写为 $p_{x,i},p_{h,i},p_{y,i}$。工具、存储与网络费用为 $c_i^{tool}$。一次运行的可审计成本是：

$$
C_{run}=\sum_{i=1}^{n}\left[(x_i-h_i)p_{x,i}+h_i p_{h,i}+y_i p_{y,i}+c_i^{tool}\right]
$$

其中 $p_{x,i},p_{h,i},p_{y,i}$ 都必须来自**该次调用的 provider、模型与缓存策略**价表；在混合模型路由中它们不是固定常数。缓存的工程目标是让稳定前缀命中 $h_i$，不是臆测某个永恒折扣。把系统提示、工具 schema 放前面，把每轮观察放后面，才有机会复用前缀计算。

将一次运行表示成依赖图 $G=(V,E)$，每个节点耗时为 $d_v$。端到端延迟由关键路径决定：

$$
L_{run}=\max_{P\in\text{paths}(G)}\sum_{v\in P}d_v
$$

所以三个互不依赖、各耗时 $0.8,1.1,0.9$ 秒的工具并行后是 $\max=1.1$ 秒，不是 $2.8$ 秒；有数据依赖时才必须相加。也不能把各步骤的 p95 机械相加当作端到端 p95——正确做法是用真实 trace 直接分位数统计整条关键路径。

对候选模型 $m$，路由不应只最小化价格，而应在质量、预算、延迟约束下求：

$$
\min_{\pi}\ \mathbb{E}[C_{success}]\quad
\text{s.t.}\quad \Pr(\text{grader pass}\mid\pi)\ge q,\quad
P95(L_{run}\mid\pi)\le \ell
$$

这里 $\pi$ 是可版本化的路由策略，$q$ 与 $\ell$ 是产品 SLA；grader 的定义见 [[38 Agent 评估与可观测性|评估与可观测性]]。

## 手绘图

![[Agent 成本与延迟优化.png]]

![[Agent 成本与延迟优化-routing与cache数据流.png]]

## 可运行代码：❌ 单价优先 vs ✅ 成功任务优先

下面只用 Python 标准库模拟路由账本，能直接运行。真实系统把 `estimate` 换成 provider 返回的 token、缓存与 trace 字段；不要把示例单价当作报价。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Model:
    name: str
    in_per_m: float
    out_per_m: float
    pass_rate: float

SMALL = Model("small", 0.20, 0.80, 0.92)
LARGE = Model("large", 2.00, 8.00, 0.98)

def call_cost(model: Model, inp: int, out: int, tool_cost: float = 0.0) -> float:
    return inp * model.in_per_m / 1_000_000 + out * model.out_per_m / 1_000_000 + tool_cost

# ❌ 只比较单次单价：遇到失败和重试时会误导。
def cheapest_once() -> tuple[str, float]:
    return SMALL.name, call_cost(SMALL, 6_000, 600, tool_cost=0.001)

# ✅ 先按复杂度路由，再比较“成功一次”的期望账；低置信度直接升级。
def route(complexity: float, confidence: float) -> Model:
    return LARGE if complexity > 0.6 or confidence < 0.8 else SMALL

def expected_cost_per_success(model: Model) -> float:
    run = call_cost(model, 6_000, 600, tool_cost=0.001)
    return run / model.pass_rate

if __name__ == "__main__":
    name, naive = cheapest_once()
    picked = route(complexity=0.35, confidence=0.91)
    print(f"❌ {name}: 单次=${naive:.5f}")
    print(f"✅ {picked.name}: 成功任务期望=${expected_cost_per_success(picked):.5f}")
    print(f"✅ {LARGE.name}: 成功任务期望=${expected_cost_per_success(LARGE):.5f}")
```

生产版还要记录 `route_reason`、模型/提示版本、cache hit、重试次数、工具费用和最终 verifier 结果；否则无法判断省下的钱是否靠牺牲质量换来。

## 落地顺序与常见误区

| 先做什么 | 可验证的产物 | 不要做什么 |
|---|---|---|
| 为每个 run 记 token、模型版本、工具费用、关键路径与 verifier 结果 | 单位成功任务成本、p50/p95、失败原因看板 | 只看 dashboard 上的平均单价 |
| 建一小套分层 eval：简单、边界、难题、工具故障 | 小模型承接率与质量下限 | 用“感觉复杂”当 router 训练标签 |
| 稳定前缀前置，并监测缓存命中 | cache hit 与命中/未命中的质量差 | 把动态时间、检索结果塞进稳定前缀 |
| 并行无依赖 I/O，设超时、取消与预算 | trace 中的关键路径缩短 | 为并行而并行，造成重复写入 |
| 设置 `max_steps`、`max_retries`、升级与降级规则 | 失控任务可停止、可解释 | 让 agent 无限反思或无限重试 |

⚠️ **模型路由不是安全边界。** 高风险动作即使由最强模型决定，也要先过参数校验、最小权限和人审闸门；参见 [[16 工具设计与工具层|工具设计与工具层]]。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：为什么不能只优化每百万 token 的价格？**  因为业务购买的是成功完成的任务。低价模型若降低通过率、增加循环或人工返工，$C_{success}=C_{run}/p$ 可能更高；还必须同时守住 p95 关键路径延迟。

**Q：并行一定能降低延迟吗？**  仅对依赖图中互不依赖的节点成立。端到端延迟是关键路径的和，并行层取最大值；写操作还要考虑幂等、超时和取消。

**Q：prompt cache 与语义 cache 有何区别？**  前者复用相同稳定前缀的计算；后者跨请求复用“相近问题”的结果。语义相近不等于答案可复用，涉及实时数据、权限或上下文时必须严格失效或绕过。

## 关键事实

- 成本应按 token、缓存、工具与重试逐 run 求和；延迟应从 trace 计算关键路径与端到端分位数，而不是拼凑各步骤均值。
- OpenAI 的 [Prompt caching 文档](https://developers.openai.com/api/docs/guides/prompt-caching) 与 [Latency optimization 文档](https://developers.openai.com/api/docs/guides/latency-optimization) 均强调按实际请求结构与当前模型配置优化；提供商定价与缓存规则会变，部署前应复核价表（文档核验：2026-07-17）。
- Anthropic 的 [Prompt caching 文档](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) 说明了显式缓存控制；它只能减少可复用前缀的重复计算，不能替代结果评估（文档核验：2026-07-17）。
- 学习型路由的一个代表性研究是 [RouteLLM（2024，arXiv:2406.18665）](https://arxiv.org/abs/2406.18665)；论文结果不能直接外推到你的模型、数据与 SLA，必须用本地 eval 校准。
- 可靠优化的北极星是“满足质量门槛的单位成功任务成本与延迟”，其验证闭环见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。
