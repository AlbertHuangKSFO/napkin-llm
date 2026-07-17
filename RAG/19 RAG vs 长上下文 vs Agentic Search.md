[[19 RAG vs 长上下文 vs Agentic Search|RAG、长上下文与 Agentic Search]] 是外部知识使用的**工程责任选型**：RAG 负责从**已知、边界明确、可索引的语料**（公共或私有）取回证据；长上下文负责让已选定且有限的材料被模型一起阅读；Agentic Search 负责多步工具探索；Deep Research 负责资料位置未知时跨来源研究与带引文综合。ACL 是 RAG 上线时的访问控制约束，不是 RAG 的定义前提。它们能组合，但不能用某个固定倍率、窗口大小或产品名替代实测。

## 直觉：四种“找资料”工作，不是一条进化链

把它想成研究助理：

- 已知的一套公共法规库或私人档案库，先从其可索引的原件中找证据——这是 [[01 什么是 RAG|RAG]]；若语料有权限边界，再按调用者过滤。
- 桌上已经挑好几份材料，要求一起读并标注页码——这是长上下文。
- 不知道答案藏在哪个文件或接口，要“搜 → 读 → 决定下一步”——这是 Agentic Search。
- 连资料范围都尚未确定，需要在开放网站、上传文件或已连接资料源之间做多源研究——这是 [[Agent/29 Deep Research Agent|Deep Research]]。

**Deep Research 不是 RAG 的“下一代”**：前者的责任是规划、跨来源搜集、评估与综合；后者的责任是对既有、范围已知且可索引的语料做证据检索，语料可以公开也可以私有。Deep Research 可以调用检索，Agentic Search 也可以把 RAG 当工具，但三者并不因可组合而变成同一种系统。

![[RAG-长上下文-AgenticSearch 决策.png]]

## 小数字手算：按任务边界路由

有五个请求，但只落入四条路线：

1. “比较公开市场的三家供应商，资料位置未知” → Deep Research。
2. “只依据已知的 10,000 篇公开法规与判例回答” → RAG；语料虽公开，但规模大且边界明确，可先建索引。
3. “只依据有部门 ACL 的 30 份内部制度回答” → RAG，并在检索时做权限过滤。
4. “依据已选定的 6 份会议纪要写结论，并标到页码” → 长上下文，输入携带页/跨度锚点。
5. “排查代码库故障，必须在搜索、日志和测试间反复下钻” → Agentic Search。

这里的“四路”不是产品层级，而是四个责任边界。若第 5 题还需要先从私有文档取证，Agent 可以调用带 ACL 过滤的 RAG；若第 4 题的材料从 6 份增长到无法受控阅读，应回到检索或分阶段综合。

## 公式推导：选择的是约束集合

把请求写成约束向量：

$$
x=(O,I,S,T)
$$

其中 $O$ 表示资料位置是否开放未知，$I$ 表示语料是否已知、边界明确且可索引，$S$ 表示材料是否已选定且有限，$T$ 表示是否需要多步工具探索。对已澄清请求按责任优先级路由：

$$
\operatorname{route}(x)=
\begin{cases}
\text{Deep Research}, & O=1\\
\text{Agentic Search}, & O=0, T=1\\
\text{长上下文}, & O=T=0, S=1\\
\text{RAG}, & O=T=S=0, I=1\\
\text{澄清范围}, & \text{其他情况}
\end{cases}
$$

四条路线覆盖所有已澄清的请求；若四个条件都不成立，先补充资料范围而不是盲选。ACL 不改变 $I$ 的定义：当语料带权限时，RAG 在取回前过滤；公开语料则不需要这一过滤。真实系统还需把引用覆盖、权限拒绝、工具次数、输入长度、延迟与成本记录成实验指标。长上下文也**可以**有 span/page 引用：前提是传入文本保留文档 ID、页码或字符跨度，并在输出时要求和验证这些锚点。

## 选择表：先匹配资料边界

| 请求特征 | 首选 | 工程责任 | 引用与安全门槛 |
|---|---|---|---|
| 开放资料、位置未知、需多源研究 | [[Agent/29 Deep Research Agent\|Deep Research]] | 规划、搜索、评估来源、综合报告 | 保留外部 URL/访问日期；审查来源质量与引用 |
| 已知、边界明确、可索引的公共或私有语料 | [[01 什么是 RAG\|RAG]] | 从语料中取回问题相关的原始证据 | 语料有 ACL 时查询前过滤；chunk/答案保留文档、版本、span |
| 材料已经选定且有限 | 长上下文 | 一起阅读与跨材料推理 | 输入携带页码/span；输出引文逐条可回查 |
| 需要多步搜索、工具调用或下钻 | [[Agent/24 Agentic Search：grep vs 向量检索\|Agentic Search]] | 根据中间结果继续检索、读取或执行受限工具 | 工具 allowlist、step/token/超时预算、每步日志 |

跨边界时采用组合：[[Agent/36 Agentic RAG|Agentic RAG]] 可以让 agent 调用带 ACL 的 RAG，再将少量已核验材料放入上下文；但仍要独立记录每个组件的责任和失败模式。

## 运行代码：路由和长上下文引用锚点

下面代码可直接用 Python 3.10+ 运行。它刻意不用 token 数或厂商型号作路由条件，并展示长上下文如何把页码/span 一起交给模型提示词。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    open_unknown_sources: bool = False
    known_bounded_indexable_corpus: bool = False
    selected_limited_materials: bool = False
    multi_step_tool_exploration: bool = False
    acl_filtering_present: bool = False  # 生产约束，不决定是否属于 RAG


@dataclass(frozen=True)
class Span:
    document_id: str
    page: int
    start: int
    end: int
    text: str


def choose_primary_route(request: Request) -> str:
    if request.open_unknown_sources:
        return "deep_research"
    if request.multi_step_tool_exploration:
        return "agentic_search"
    if request.selected_limited_materials:
        return "long_context"
    if request.known_bounded_indexable_corpus:
        return "rag"
    return "clarify_scope"


def render_anchored_context(spans: list[Span]) -> str:
    return "\n\n".join(
        f"[source={s.document_id}; page={s.page}; chars={s.start}-{s.end}]\n{s.text}"
        for s in spans
    )


public_corpus = Request(known_bounded_indexable_corpus=True)
private_corpus = Request(known_bounded_indexable_corpus=True, acl_filtering_present=True)
assert choose_primary_route(public_corpus) == "rag"
assert choose_primary_route(private_corpus) == "rag"
assert choose_primary_route(Request(open_unknown_sources=True)) == "deep_research"
assert choose_primary_route(Request(selected_limited_materials=True)) == "long_context"
context = render_anchored_context([Span("memo-7", 3, 120, 180, "预算在 Q3 调整。")])
assert "page=3" in context and "chars=120-180" in context
print(context)
```

❌ “窗口够大就全部塞入”或“RAG 有引文就天然忠实”。
✅ RAG 从已知、边界明确的索引中取证并评测召回/引用；语料有 ACL 时再按调用者身份过滤。长上下文保留并核验 page/span；Agent 的每次工具调用设 allowlist、预算和审计日志；Deep Research 让用户控制可用资料范围并审查来源。安全侧需同时处理 [[AI 安全/05 Prompt Injection 提示注入|提示注入]]、[[AI 安全/11 向量与嵌入弱点与 RAG 投毒|RAG 投毒]]、[[AI 安全/12 Unbounded Consumption 成本型 DoS|资源消耗型 DoS]] 与 [[AI 安全/24 沙箱、最小权限与人审闸门|最小权限与人审闸门]]。

## 评测卡：把“取舍”变成可复现实验

同一问题集至少记录：答案是否有正确的证据锚点、授权用户能否访问、未授权用户是否被拒绝、工具轨迹是否可复现、实际输入长度、调用次数、端到端延迟与成本。不要把单针检索测试或厂商标称上下文窗口外推成真实多证据任务的固定表现。

长上下文仍应留意 **lost in the middle**：Liu et al. 在受控实验中报告，信息位置会影响模型对上下文的使用。它是应纳入本地评测的风险提示，不是所有模型、所有材料的固定百分比结论。

## 面试高频

**Q1：什么时候选 Deep Research，什么时候选 RAG？**
答：开放资料位置未知、需要跨来源寻找和综合时选 Deep Research；资料已知、范围明确且可索引时选 RAG，无论它是公共还是私有。若语料有 ACL，RAG 还必须按调用者过滤。两者可以相互调用，但不互相取代。

**Q2：长上下文能否提供页码或 span 引用？**
答：可以。把文档 ID、页码/字符跨度与内容一起传入，并要求输出引用，再逐条验证即可；“能塞进窗口”本身不自动带来可追溯性。

**Q3：什么时候要 Agentic Search？**
答：中间证据会改变下一步，要多轮搜索、读取或调用工具时。给它最小权限工具集、step/token/超时预算和完整轨迹；需要私有知识时，将 [[Agent/36 Agentic RAG|Agentic RAG]] 的 ACL RAG 作为一个受限工具。

## 关键事实

- Liu et al., **Lost in the Middle: How Language Models Use Long Contexts**（arXiv，2023-07；TACL，2024）：报告了不同上下文位置会影响模型利用信息的现象，应在目标模型与目标材料上复测。[一手论文](https://arxiv.org/abs/2307.03172)
- OpenAI，**Deep research in ChatGPT**（官方帮助文档，更新于 2026-07，访问 2026-07-17）：Deep Research 可在用户选择的公共网页、上传文件和已连接资料源上进行多步研究，并输出带引文/来源链接的报告。[官方文档](https://help.openai.com/en/articles/10500283-deep-research-in-chatgpt)
- OpenAI，**Research with ChatGPT**（OpenAI Academy，2026-04）：将 Deep Research 描述为会规划、搜索、评估来源、细化查询并综合的 agentic 研究过程。[官方说明](https://openai.com/academy/search-and-deep-research/)
