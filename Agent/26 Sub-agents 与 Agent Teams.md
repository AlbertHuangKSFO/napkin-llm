[[26 Sub-agents 与 Agent Teams|Sub-agents 与 Agent Teams]] 讨论两种协作拓扑：**中心式 sub-agent** 由父 agent 分派、收集结果并作最终决策；**对等 team** 允许成员显式协商。两者不是某个 SDK 必然自带的产品功能，而是由 [[23 Agent Harness 概览|Agent Harness]] 实现的通信与状态管理选择。

## 直觉 / 生活类比

把中心式 sub-agent 想成主编派三位记者：主编给题、记者各自调查、最后只交一页摘要；记者之间是否能互相打电话，取决于编辑部是否给了这条线路。这样做的收益是把搜索噪声留在各自的笔记本里，主编只保留可执行的结论。

peer team 更像一个项目组：成员可把未完成的证据、反例和澄清请求互相发送，并共同更新任务板。它适合任务本身需要协商，例如「一人提出方案、一人找反例、一人裁决」。但它不是“更高级的 sub-agent”：消息往返、冲突状态与不收敛都会增加成本。

因此要先问通信需求，而不是先选名词：

| 需求 | 合适结构 | 必须显式规定 |
| --- | --- | --- |
| 可独立检索、跑测试、做局部审计 | 父代—子代 | 输入范围、返回摘要、超时、父代汇总 |
| 交叉质疑、共同写同一份方案 | 对等 team | 收件人、共享状态、仲裁者、轮次上限 |

“子 agent 不能彼此通信”只对**未提供横向通道的中心式 harness**成立；若 harness 提供队列、共享状态或 `send_message`，同一批 worker 也可以通信，此时它已采用部分 team 拓扑。不要把产品实现细节误说成通用定义。

## 小数字手算

一次调研拆成 $3$ 个互不依赖的问题。每个子任务原始轨迹约 $1\,200$ token，最后结论约 $120$ token。

中心式回传给父代的上下文负担为：

$$
3\times120=360\text{ token}
$$

若把三份完整轨迹全部广播，每位成员都读另外两份，横向阅读量约为：

$$
3\times2\times1\,200=7\,200\text{ token}
$$

这不是所有系统的精确账单，却说明了结构性差别：隔离降低的是**主上下文与消息重复量**，不是总推理成本。任务若需要两轮互相核验，则这 $7\,200$ 还会随轮次继续累积。

## 公式推导

令 $W_i$ 是第 $i$ 个 worker 的完整轨迹，$s(W_i)$ 是受长度约束的摘要，父代上下文为 $C_p$。中心式汇总为：

$$
C_p' = C_p \oplus \sum_{i=1}^{k}s(W_i),\qquad |s(W_i)|\ll |W_i|
$$

这里的关键不是 fork 这个词，而是**信息边界**：父代接收的是经任务契约筛过的结果。对等 team 则还需维护共享状态 $B$ 与消息集合 $M$：

$$
B_{t+1}=\operatorname{merge}(B_t, M_t),\qquad
M_t=\{m_{i\rightarrow j}\mid i\ne j\}
$$

因此 team 的正确性还依赖冲突策略 `merge`、谁有写权限、何时停止；仅“让所有人互聊”不构成协作设计。[[22 多智能体系统|多智能体系统]] 的协调成本正从这里出现。

## 手绘图

![[Sub-agents 与 Agent Teams.png]]

## 可运行代码 / 配置

下面示例只用 Python 标准库，运行 `python3 topology_demo.py`。`❌` 的广播既泄露无关轨迹，又把通信当成默认能力；`✅` 明确使用父代收集摘要。若要改成 peer team，必须由 harness 明确提供队列、身份与仲裁规则。

```python
# topology_demo.py
import asyncio

async def investigate(topic: str) -> dict:
    await asyncio.sleep(0.01)  # 代替检索/测试
    return {"topic": topic, "summary": f"{topic}: 已核验一条结论"}

# ❌ 把完整中间产物广播给所有 worker；无收件人和停止契约。
async def noisy_broadcast(topics: list[str]) -> None:
    traces = await asyncio.gather(*(investigate(t) for t in topics))
    for trace in traces:
        print("广播给所有成员：", trace)

# ✅ 中心式：父代只接收约定格式的摘要，并独占最终合成权。
async def parent_with_subagents(topics: list[str]) -> str:
    reports = await asyncio.gather(*(investigate(t) for t in topics))
    assert all(set(r) == {"topic", "summary"} for r in reports)
    return "\n".join(r["summary"] for r in reports)

async def main() -> None:
    result = await parent_with_subagents(["成本", "证据", "风险"])
    print(result)

asyncio.run(main())
```

生产配置至少应写下：`max_children`、`deadline`、每个子任务可访问的工具、摘要 schema、失败是否重试，以及 parent 是否允许把结果转发给其他成员。Anthropic 的多 agent research 系统（2025）报告其多 agent 任务的 token 使用约为普通 chat 的 $15\times$；这是该产品工作负载的观测值，不可当作所有框架的固定倍率。[一手工程报告，2025](https://www.anthropic.com/engineering/multi-agent-research-system)

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：sub-agent 与 team 的分界是什么？**

答：是通信和控制权的拓扑，不是名字。中心式模式由父代分派、收敛与决定；team 额外提供成员间消息和共享协调。sibling 能否直接通信完全取决于 harness，不能把某个产品的限制推广为通则。

**Q：为什么要 context isolation？**

答：让探索失败、长网页和工具日志留在 worker 内，只把有 schema 的结论交给父代；它减少主上下文污染，但会增加多个上下文的 token 与调度成本。

**Q：什么时候不要上多 agent？**

答：强顺序依赖、需要频繁共享同一细粒度状态、或任务价值不足以覆盖额外 token 时，先用单 agent 加工具循环。可并行且证据面广的研究任务才更可能受益。

## 关键事实

- **中心式 sub-agent 是模式，不是协议**：父代—子代报告路径、上下文派生和横向通信权限均由具体 harness 定义。它与 [[07 Orchestrator-Workers|Orchestrator-Workers]] 相近，但不要求某个固定 API。
- **peer team 必须有收敛设计**：共享任务板只是状态载体；仍要定义仲裁者、冲突合并、轮次/预算和责任边界。
- **量化结论必须带任务口径**：Anthropic 的一手报告发表于 **2025-06-13**，其中 $15\times$ 比较的是其研究产品与 chat 的 token 使用，而非模型或架构的普适常数。[来源](https://www.anthropic.com/engineering/multi-agent-research-system)
- 跨组织 agent 的标准化互通见 [[30 A2A 协议|A2A 协议]]；同一 harness 内该用中心式还是对等式，仍是本篇讨论的拓扑选择。
