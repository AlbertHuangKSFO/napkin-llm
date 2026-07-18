[[36 Agentic RAG|Agentic RAG]] 的本质是：将检索器、数据库、网页或代码搜索包装为 agent 可选择的**工具**，让策略根据当前证据决定“检不检索、检什么、下一步做什么、何时停”。它不是“多检索一次”的同义词；只有存在**状态—行动—观察—终止**的闭环控制时，才称得上 Agentic RAG。

它连接 [[15 Function Calling 工具调用|Function Calling 工具调用]]、[[09 ReAct|ReAct]] 和 [[RAG]]：传统 RAG 常把“检索一次→生成”写成固定管线；Agentic RAG 则让检索成为可审计、受预算约束的行动。对代码或精确标识符，工具也可以是 [[24 Agentic Search：grep vs 向量检索|grep 搜索]]，不必假装所有知识都是向量库。

完整的检索主题脉络见 [[RAG/RAG]]。

## 直觉：固定取书 vs 有目标地查资料

传统 RAG 像图书管理员无论问题是什么都取固定 $k$ 本书，然后要求学生作答。它可以做得非常好：混合检索、重排、过滤和高质量上下文本来就足够解决大量单跳问题。

Agentic RAG 像一名有预算的研究助理：先看已知证据；若缺少一个实体就查实体；发现来源冲突便换查询或来源；找到可核验的证据后停止，并说明依据。关键不是“模型有内心独白”，而是能观察到的**工具选择、证据标识、状态变化、停止原因和预算消耗**。

⚠️ 以下情况**不自动等于** Agentic RAG：

- 固定写死“改写 query → 检索两次 → 生成”的 DAG：它是增强 RAG 管线；若分支不由运行时证据改变，仍不是自治循环。
- 只加 reranker 或 query rewrite：它可能提高召回，但没有工具选择与终止策略。
- 多跳：只有后一跳的行动由上一跳 observation 决定时，才是适应式多跳；预先列好的三次查询只是三段管线。

## 小数字手算：多跳的调用账要数清楚

设检索本身不是 LLM 调用。固定 RAG 用一次检索和一次生成，LLM 调用数为 $1$。一个 Agentic RAG 循环有 $N$ 轮：每轮由一次模型决策选择“搜索或结束”；若另外用一次 grader 判检索证据质量，记为 $g=1$；结束后再生成一次答案。

$$
K_{\text{LLM}}=N(1+g)+1
$$

若 $N=3$、每轮有独立证据 grader（$g=1$），则 $K=3(1+1)+1=7$ 次 LLM 调用。若决策模型直接输出“继续/结束”并不另设 grader（$g=0$），则是 $4$ 次。**不是所有 Agentic RAG 必然 7 次**；调用数由控制策略决定，检索 API 的钱、每轮携带的证据 token、缓存与并行还要另记。

## 公式推导：把“自适应”写成可验证的闭环

令问题为 $q$，第 $t$ 轮累计证据为 $E_t$，历史工具事件为 $H_t$。策略根据可见状态选择行动：

$$
a_t=\pi(q,E_t,H_t,b_t)\in\{\text{answer},\text{retrieve}(s,query),\text{verify},\text{ask\_user}\}
$$

若 $a_t=\text{retrieve}(s,u)$，则工具返回 observation $o_t$，并更新：

$$
E_{t+1}=\operatorname{dedupe\_and\_cite}(E_t\cup o_t),\qquad
b_{t+1}=b_t-c(a_t)
$$

停止条件不能只写“模型觉得够了”。生产应至少满足：

$$
\text{stop}_t=\text{evidence\_sufficient}(q,E_t)\ \lor\ b_t\le0\ \lor\ t\ge T_{max}
$$

其中 `evidence_sufficient` 可以是可执行 verifier、来源覆盖规则或人工升级。最终答案应附证据 ID；证据不足时返回“不足以断言/请求澄清”，而不是用检索过的文本掩盖猜测。

## 手绘图

![[Agentic RAG.png]]

![[Agentic RAG-流程.png]]

## 可运行代码：❌ 固定一次检索 vs ✅ 证据驱动循环

下面用内存文档模拟检索。它不调用模型，却把控制边界写清：固定版本永远只查一次；循环版本先查 CEO，再利用 observation 构造下一跳查询，并在证据充分或预算耗尽时终止。

```python
from dataclasses import dataclass

DOCS = {
    "A公司 CEO": "A公司 CEO 是林岚。",
    "林岚 母校": "林岚毕业于海湾大学。",
    "海湾大学 城市": "海湾大学位于青岛。",
}

@dataclass
class Evidence:
    query: str
    text: str

def retrieve(query: str) -> Evidence:
    return Evidence(query, DOCS.get(query, ""))

def fixed_rag(question: str) -> str:
    # ❌ 查询和轮数写死；第一条证据不足也照样回答。
    evidence = retrieve("A公司 CEO")
    return f"问题：{question}\n证据：{evidence.text}\n结论：信息不足"

def agentic_rag(question: str, max_hops: int = 3) -> tuple[str, list[Evidence]]:
    facts, trace = {}, []
    next_query = "A公司 CEO"
    for _ in range(max_hops):
        ev = retrieve(next_query)
        trace.append(ev)
        if not ev.text:
            return "证据库没有足够信息，需换来源或请求澄清。", trace
        if "CEO 是" in ev.text:
            facts["ceo"] = ev.text.removesuffix("。").split("是")[1]
            next_query = f"{facts['ceo']} 母校"
        elif "毕业于" in ev.text:
            facts["school"] = ev.text.removesuffix("。").split("毕业于")[1]
            next_query = f"{facts['school']} 城市"
        elif "位于" in ev.text:
            city = ev.text.removesuffix("。").split("位于")[1]
            return f"{facts['ceo']} 的母校在 {city}。", trace
    return "达到 max_hops，证据仍不足。", trace

if __name__ == "__main__":
    question = "A公司 CEO 的母校在哪个城市？"
    print(fixed_rag(question))
    answer, trace = agentic_rag(question)
    print(answer)
    print([e.query for e in trace])  # 可审计的工具轨迹
```

接入真实模型时，`next_query` 由结构化 tool call 给出；必须校验 source 白名单、查询长度、重复查询、`max_hops`、每轮 token/工具预算和最终引用。

## 选型卡：什么时候值得加“agentic”

| 问题特征 | 优先方案 | 原因 |
|---|---|---|
| 单跳、低延迟、固定私有语料 | 固定 RAG + 过滤/重排 | 控制面小，容易评测与缓存 |
| 问题含糊但知识源单一 | query rewrite 或条件 RAG | 先验证检索质量，别急着加循环 |
| 实体链、多跳、来源需按观察切换 | Agentic RAG | 下一步由证据决定，能设预算与终止 |
| 精确代码符号、文件路径、命令 | grep/结构化索引工具 | 精确匹配与新鲜度优先 |
| 需要写入或执行外部动作 | agent + 权限/审批/verifier | 检索循环本身不能授权副作用 |

常见失败：重复改写同一查询、把低相关证据越堆越多、让模型自己给自己无条件打高分、没有最终引用或 `max_hops`。应把 rerank、来源质量、断言式 verifier 和用户澄清放在闭环内，而不是只增加轮数。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**Q：Agentic RAG 和传统 RAG 的边界是什么？**  传统 RAG 可以有高质量检索，但通常以固定流程完成；Agentic RAG 的检索是策略可选的工具，后续行动和停止由当前 observation 改变。关键证据是可见的状态、工具事件与预算，而不是“多调用了几次模型”。

**Q：Self-RAG、CRAG 是否等于生产方案？**  不等于。它们是提出自反思/纠错思想的研究方法；生产可借用“按需检索、评估证据、换源或停”的原则，但仍要做数据、版本、成本、安全和回归评测。

**Q：如何避免 Agentic RAG 无限循环？**  同时限制 $T_{max}$、token/工具预算、重复查询次数与时钟时间；遇到证据冲突或不足时升级到澄清、备用来源或明确失败，而非继续猜。

## 关键事实

- [Lewis et al., 2020，RAG，arXiv:2005.11401](https://arxiv.org/abs/2005.11401) 将参数化模型与非参数记忆结合；它不是“必须有 agent loop”的定义。
- [Self-RAG，2023，arXiv:2310.11511](https://arxiv.org/abs/2310.11511) 研究按需检索与反思 token；[CRAG，2024，arXiv:2401.15884](https://arxiv.org/abs/2401.15884) 用检索评估器触发不同知识获取动作。这些论文支持“反馈回路”的设计动机，不替代本地评测。
- 框架不定义 Agentic RAG。比如 [LlamaIndex Workflows 文档](https://developers.llamaindex.ai/python/llamaagents/workflows/) 描述事件/步骤式 workflow；用任何运行时都应记录工具选择、证据 ID、停止原因和 terminal verifier（文档核验：2026-07-17）。
- 成本与延迟的公式、缓存和路由约束见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]；轨迹与证据的安全审计见 [[38 Agent 评估与可观测性|Agent 评估与可观测性]]。宏观选型见 [[RAG/19 RAG vs 长上下文 vs Agentic Search|RAG、长上下文与 Agentic Search]]；多模态来源的路由、融合与后期交互见 [[RAG/23 多模态检索编排：路由、融合与后期交互|多模态检索编排]]。
