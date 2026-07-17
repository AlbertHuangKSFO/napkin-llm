[[14 GraphRAG 知识图谱检索|GraphRAG]] 是把非结构化语料转换为实体、关系、社区报告等结构化中间产物，再把它们作为回答上下文的 RAG 路径。它尤其适合**需要综合整库视角的 QFS（query-focused summarization，面向查询的摘要）**；这不表示 [[01 什么是 RAG|向量 RAG]] 不能回答全局问题，而是两者要用同一套问题集实测覆盖、可追溯性与成本后再选。

## 直觉：先做“主题档案”，再回答全局问题

查某一条制度时，[[02 Naive RAG 与失败模式|向量检索]] 可以找相近片段；问“这批事故反复出现哪些根因”时，系统还要把分散证据组织成可审计的概括。GraphRAG 的做法是：离线抽取实体与关系、识别社区、为社区生成报告；在线将相关报告作为综合的输入。它是一种**改善特定全局 QFS 覆盖的工程路径**，不是“图一定比向量好”的结论。

![[GraphRAG vs 向量RAG.png]]

## 小数字手算：把“覆盖”变成待测指标

设评测集中某个全局问题需要覆盖 3 个独立主题：$A,B,C$。一次方案的答案有可核验出处地覆盖了 $A,B$，没有覆盖 $C$，则该题的主题覆盖率是：

$$
\operatorname{coverage}=\frac{|\{A,B\}|}{|\{A,B,C\}|}=\frac{2}{3}
$$

另一方案若覆盖 $A,B,C$，但其中 $B$ 的引文无法回到原文，不能因此直接判胜；还要记录引用正确性、ACL 拒绝率、更新时间与端到端延迟。GraphRAG 的社区报告提供了另一种“按主题组织证据”的候选上下文，是否提升上述指标必须在自己的语料与问题上测试。

## 公式推导：图、社区与 QFS

从文本单元抽取图 $G=(V,E)$：$V$ 是实体，$E$ 是带来源文本单元的关系。社区检测把实体划分为层级集合 $\mathcal{C}_\ell$；每个社区报告应保存其来源集合：

$$
R(c)=\operatorname{summarize}(\{(v,e,\text{source})\mid v,e\in c\}),\quad c\in\mathcal{C}_\ell
$$

全局 QFS 不是“找到唯一正确 chunk”，而是先对报告生成候选观点，再由模型整合：

$$
a_c=\operatorname{map}(q,R(c)),\qquad
\text{answer}=\operatorname{reduce}(q,\{a_c\}_{c\in\mathcal{C}_\ell})
$$

`map`／`reduce` 都可能遗漏或误概括，因此最终答案应携带报告 ID 与可回溯的原文证据，而非把社区报告当成可信事实源。

![[GraphRAG-流程.png]]

## 运行代码：把 ACL、版本与 provenance 继承到社区报告

社区报告是派生数据，访问控制必须至少和所有输入源一样严格；同时记录来源 ID、版本与生成版本。下面代码可直接用 Python 3.10+ 运行，演示“只有同时有权访问全部输入源的人才能读报告”。

```python
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class Source:
    source_id: str
    version: str
    allowed_principals: FrozenSet[str]


@dataclass(frozen=True)
class CommunityReport:
    report_id: str
    source_ids: tuple[str, ...]
    source_versions: tuple[str, ...]
    allowed_principals: FrozenSet[str]
    pipeline_version: str


def build_report(report_id: str, sources: list[Source]) -> CommunityReport:
    assert sources, "社区报告必须有来源"
    # 派生报告取 ACL 交集；宽松合并会把受限内容泄露给只看得到部分源的人。
    acl = frozenset.intersection(*(s.allowed_principals for s in sources))
    return CommunityReport(
        report_id=report_id,
        source_ids=tuple(s.source_id for s in sources),
        source_versions=tuple(s.version for s in sources),
        allowed_principals=acl,
        pipeline_version="graphrag-config-v7",
    )


finance = Source("finance-q2", "2026-07-01", frozenset({"alice", "bob"}))
hr = Source("hr-plan", "2026-07-10", frozenset({"alice"}))
report = build_report("community-42", [finance, hr])
assert report.allowed_principals == frozenset({"alice"})
assert "bob" not in report.allowed_principals
print(report)
```

❌ 只保存“社区摘要文本”，丢掉源 ACL、版本和来源。
✅ 将 `source_ids`、`source_versions`、`allowed_principals`、抽取 prompt／模型／管线版本一起落库；源文档、ACL 或抽取配置变化后，把受影响报告标为待更新，并在发布前重新做权限测试。相关威胁与防护见 [[AI 安全/05 Prompt Injection 提示注入|提示注入]]、[[AI 安全/11 向量与嵌入弱点与 RAG 投毒|RAG 投毒]]、[[AI 安全/12 Unbounded Consumption 成本型 DoS|资源消耗型 DoS]] 与 [[AI 安全/24 沙箱、最小权限与人审闸门|最小权限与人审闸门]]。

Microsoft 当前 CLI（访问 2026-07-17）公开了 `update` 命令，以及 `standard-update`、`fast-update` 方法；因此不能断言“新增文档必定重跑 Leiden”。更新能否复用何种产物、是否重算社区，要以**锁定的库版本、配置和 dry-run／评测结果**为准：

```bash
# 先检查已安装版本支持的更新路径与配置，不把更新行为写死在经验里。
graphrag index --root ./project --method standard-update --dry-run
graphrag update --root ./project --method fast-update
```

## 选型卡：比较责任，不比较口号

| 维度 | 向量 RAG | GraphRAG |
|---|---|---|
| 首要责任 | 取回与问题相关的原文候选 | 为全局 QFS 提供按社区组织的候选上下文 |
| 主要产物 | chunk、向量、检索结果 | 图、社区报告、原文 provenance |
| 关键风险 | 漏召回、切块丢语义 | 抽取/归并/摘要误差与派生数据越权 |
| 更新判断 | 测新旧索引的检索与引用指标 | 测受影响报告、社区与权限传播；不预设全量或增量一定更好 |
| 上线门槛 | 原文引文可回查 | 报告引文可回查，且 ACL、版本、生成配置完整继承 |

官方索引文档说明标准管线会抽取实体/关系、做社区检测、在多粒度生成报告；`fast` 是不同的索引方法，不应把第三方整合、性能数字或基准结论混作 Microsoft GraphRAG 的承诺。

## 面试高频

**Q1：GraphRAG 是不是解决了向量 RAG 答不了全局问题？**
答：它为全局 QFS 增加了“社区报告 + 综合”的上下文路径，可能改善特定语料的主题覆盖；向量 RAG 也可以用多轮检索、分层摘要或更大的上下文回答。应以同一评测集比较覆盖、引文、ACL 和成本，不能做绝对断言。

**Q2：社区报告为什么要带 ACL、版本和 provenance？**
答：报告是多个源的派生内容，若丢失最严格 ACL 或来源版本，就会产生越权和不可审计的陈旧结论。把 provenance 当作与报告正文同等重要的数据模型字段。

**Q3：GraphRAG 更新是否一定要重跑社区检测？**
答：不是。当前官方 CLI 提供更新相关命令与 `standard-update`／`fast-update` 方法；具体运行范围取决于版本和配置，必须通过 dry-run、变更集和回归评测确认。

## 关键事实

- Edge et al., **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**（Microsoft Research，2024-04）：将全局问题界定为 QFS，并描述“实体知识图谱 → 预生成社区摘要 → 部分答案再综合”的方法。[一手论文页](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/)
- Microsoft GraphRAG **CLI Reference**（官方文档，访问 2026-07-17）：`index` 与 `update` 的方法枚举均包含 `standard`、`fast`、`standard-update`、`fast-update`。[官方 CLI 文档](https://microsoft.github.io/graphrag/cli/)
- Microsoft GraphRAG **Indexing Methods**（官方文档，访问 2026-07-17）：标准方法使用语言模型抽取和汇总；`fast` 是以传统 NLP 替代部分语言模型推理的方法，适用性需自行验证。[官方索引方法文档](https://github.com/microsoft/graphrag/blob/main/docs/index/methods.md)
