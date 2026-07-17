[[29 Deep Research Agent|Deep Research Agent]] 是一个带研究契约的 agent：先明确问题、来源范围和预算，再计划、检索/阅读、核验证据、写成可回链的报告。它的“深”不由搜索次数定义，而由**来源集合、显式规划、证据与引用忠实度、资源预算**共同定义。

## 直觉 / 生活类比

普通 RAG 像从固定资料柜抽几页卡片后回答；Agentic RAG 像馆员会根据缺口继续翻同一柜或调用工具；Deep Research 则像受托写尽调报告：先给研究提纲，约定只查哪些公开/私有来源，逐条留下“这句话由哪段材料支持”，并在时间或来源额度耗尽时停止。

| 形态 | 来源集合 | 规划 | 证据与引用 | 预算 |
| --- | --- | --- | --- | --- |
| RAG | 固定索引的 top-$k$ 片段 | 通常无显式计划 | 检索元数据可回传 | `k`、上下文窗口 |
| Agentic RAG | 索引加工具，按缺口扩展 | 可动态重写检索 | 可做验证，但不是定义上必然 | 工具轮数、检索次数 |
| Deep Research | 明确授权的网页、文件、连接器/数据库 | 可审阅的研究计划与子问题 | 逐条结论回链、检查引用是否真的支撑 | 来源数、时间、迭代轮数、token/费用 |

所以“一次搜索”既不自动是 RAG，也不自动排除 Deep Research；决定类型的是系统是否遵守上述研究契约。OpenAI 的产品流程也将来源选择、研究计划、进度控制与带引用报告作为独立步骤，而不是把它描述为单一 search call。[OpenAI Help Center，更新于 2026](https://help.openai.com/en/articles/10500283-deep-research)

固定语料检索的知识地图见 [[RAG/RAG]]。

## 小数字手算

设研究计划有 $4$ 个主张槽位，每个槽位至少要一个一手来源，允许的总来源数为 $12$。本次收集到 $10$ 个来源，其中 $4$ 个直接支撑四个主张，$2$ 个互相矛盾而被标为待确认。

$$
\operatorname{claim\ coverage}=\frac{4}{4}=100\%,\qquad
\operatorname{source\ budget\ used}=\frac{10}{12}\approx83.3\%
$$

覆盖率 $100\%$ 不表示真实正确：若引用段落不蕴含主张，引用忠实度仍可能很低。因此需要单独记录“支持/反驳/无关”判定；已用 $83.3\%$ 预算也提示系统只剩 $2$ 个来源去解决矛盾，不能无界重搜。

## 公式推导

令计划的主张集合为 $C$，来源为 $S$，关系 $E(c,s)=1$ 表示来源 $s$ 的具体片段支持主张 $c$。最低覆盖约束为：

$$
\forall c\in C,\quad \sum_{s\in S}E(c,s)\ge1
$$

令 $q(s)$ 是来源质量、$u(s)$ 是读取成本，预算为 $B$。一个有界研究目标可写为：

$$
\max_{S}\sum_{c\in C}\max_{s\in S}\bigl(E(c,s)\,q(s)\bigr)
\quad\text{s.t.}\quad \sum_{s\in S}u(s)\le B
$$

这说明为何“抓得越多越好”是反模式：更多低质页会占用 $B$、挤压核验和写作空间。深度研究的回环应由未覆盖主张或冲突证据触发，而不是固定再搜一轮。

## 手绘图

![[Deep Research Agent.png]]

## 可运行代码 / 配置

下例用 Python 标准库演示最小的“主张—证据”门槛。保存为 `evidence_gate.py` 后运行 `python3 evidence_gate.py`；它不会联网，但能防止写作器把没有对应证据的主张伪装成已引用结论。

```python
# evidence_gate.py
claims = {"发布日期", "正式规范版本"}
sources = {
    "a2a-spec": {"quality": "primary", "supports": {"正式规范版本"}},
    "launch-post": {"quality": "primary", "supports": {"发布日期"}},
}

# ❌ 只把搜索摘要拼进答案，无法检查“某句由什么支持”。
# answer = "A2A 已发布且规范稳定。"

# ✅ 每个主张至少映射一条允许的证据，缺证据则不输出为事实。
def cited_claims() -> dict[str, list[str]]:
    result = {}
    for claim in claims:
        refs = [name for name, item in sources.items()
                if item["quality"] == "primary" and claim in item["supports"]]
        if not refs:
            raise ValueError(f"未核验主张：{claim}")
        result[claim] = refs
    return result

print(cited_claims())
```

把 `sources` 换成真实抓取结果时，还应保存 URL、抓取时间、原文片段、出版日期、许可/访问限制和反驳证据。可用 [[26 Sub-agents 与 Agent Teams|中心式 sub-agent]] 并行做相互独立的资料收集，但最终引用审核应有单一责任方，避免多个 worker 相互转述后丢失原始出处。

## 面试高频

**Q：Deep Research 与 Agentic RAG 最大区别？**

答：不是“多搜几次”。Deep Research 要把来源边界、计划、逐条证据回链和时间/成本预算当作产品契约；Agentic RAG 可以有多轮检索，但未必需要公开可审的报告级引文与证据审计。

**Q：为什么 citation faithfulness 比“有很多链接”更重要？**

答：链接数量不证明结论被支持。要检查每一主张能否回到具体原文片段、该片段是否真的蕴含主张、是否遗漏反证与时间口径；没有通过的内容应降级为假设或删除。

**Q：如何控成本并避免研究跑偏？**

答：先限定来源集合与交付格式，再给来源数、并发、迭代和时间设硬上限；只在“主张未覆盖/证据冲突”时追加检索，并让用户可审阅研究计划。Anthropic 的 2025 工程报告观察到多 agent research 比 chat 消耗更多 token，说明并行只缩短墙钟时间，不免除预算治理。[来源](https://www.anthropic.com/engineering/multi-agent-research-system)

## 关键事实

- **OpenAI 于 2025-02 发布 Deep Research**，描述其为多步互联网研究并输出带引用报告；产品随后新增可选择来源、审核计划与进度控制等交互。[发布说明，2025](https://openai.com/index/introducing-deep-research/)｜[使用说明，2026](https://help.openai.com/en/articles/10500283-deep-research)
- **研究输出要验证而非堆链接**：来源质量、主张覆盖、引用忠实度和日期口径是互相独立的质量维度。
- **预算是定义的一部分**：来源数、抓取字数、模型层级、迭代轮数、并发和最长运行时间应写入 trace；不设上限会让检索与上下文成本失控，见 [[35 Agent 成本与延迟优化|Agent 成本与延迟优化]]。
- 需要固定私有语料的问答优先考虑 [[36 Agentic RAG|Agentic RAG]]；跨组织委派研究 agent 时才考虑 [[30 A2A 协议|A2A 协议]]。
