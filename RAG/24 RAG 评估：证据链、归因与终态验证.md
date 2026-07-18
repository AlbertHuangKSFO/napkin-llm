[[24 RAG 评估：证据链、归因与终态验证|RAG 全链评估]]不是再加一个答案分数，而是把一次 RAG run 当作**可验收的证据链**：请求被正确解析、证据被正确找回并定位、答案每个 claim 被可访问的证据蕴含、工具只留下允许的副作用且系统到达预期终态；任一层失败，端到端“答对”都不能算成功。它在 [[18 RAG 评估|基础 RAG 评估]] 的检索/生成归因之上，把权限、撤销、工具状态与成本也纳入同一份结构化 trace；这正是与 [[Agent/38 Agent 评估与可观测性|Agent 评估与可观测性]] 相接的边界。

## 直觉：像银行转账的对账单，而非只看余额

用户问“退款窗口多久”，最终答“30 天”像收款人看到余额增加；但审计还要核查：付款指令是否解析成了正确账户、钱是否来自正确账本、凭证是否还有效、引用的那一行是否真的写了“30 天”、是否误创建了退款工单、最终账本状态是否已提交。RAG 同理：

- **解析**错，会把“退款政策”查成“退款进度”；**检索**错，后面的生成没有原料。
- **定位**把旧版片段当新版，即使文档 ID 对了也不算证据；**引用/蕴含**防止“贴一条看似相关的链接，答案却另说一件事”。这与 [[11 生成层：引用归因与忠实度|引用归因与忠实度]] 的 claim 级检查相连。
- **ACL 与撤销**必须在取证时生效：不能因为模型曾经见过、缓存过或能猜到，就把无权文本带进上下文。投毒语料、诱导越权是 [[AI 安全/11 向量与嵌入弱点与 RAG 投毒|RAG 投毒]] 的评测负例，不是普通准确率能覆盖的。
- 若 RAG 能调用写工具，“回答正确”仍不能抵消一次错误扣款/发信；要验证允许的 action diff、幂等性、回滚与**终态**。短期凭证和撤销测试对应 [[16 Agent 身份与权限滥用(非人类身份 NHI)|非人类身份权限治理]]；测试动作还应遵循 [[AI 安全/24 沙箱、最小权限与人审闸门|沙箱、最小权限与人审闸门]]。

**留什么 trace？**留最小可复核结构：`run_id`、数据集/语料/ACL/提示词/工具模拟器版本、解析字段、`doc_id + revision + page/span`、授权判定、工具白名单入参和状态差异、token/时延/成本、指标与拒绝码。不要把原始 chain-of-thought 当审计数据；它既非稳定证据，也会扩大敏感数据面。需要人工复查时，保存可引用的输入、输出、证据与动作摘要即可。

## 小数字手算：单一答案分掩盖了 30% 的事故

10 个 held-out 用例各跑 3 次，得到 $N=30$ 次 run。最终答案文本正确 26 次，但只有 21 次同时满足证据、权限、终态和预算门：

| 层 | 通过 run 数 | 读法 |
|---|---:|---|
| 解析 | 29 | 1 次把过滤条件读错 |
| 检索 | 27 | 3 次漏掉 gold 文档 |
| 定位 + 修订版 | 25 | 2 次引用了过期段落 |
| 回答正确 | 26 | 单看这里会报 $86.7\%$ |
| ClaimSupportRate（逐 claim） | $84/90=93.3\%$ | 诊断性分数；6 个 claim 没有被证据支持 |
| 引用蕴含门 $e_i$ | 24 | 24 个 run 的所有 claim 都被支持；其余 6 个 run 直接失败 |
| ACL / 撤销 | 29 | 1 次撤权后仍能读缓存 |
| 工具终态 | 28 | 2 次副作用或状态机不对 |
| 成本预算 | 27 | 3 次答对但超预算 |

严格成功率不是各列平均，而是一次 run 的所有门都亮绿：

$$
\text{StrictPass} = \frac{21}{30}=0.70=70\%
$$

于是“答案正确率 $26/30=86.7\%$”与“可安全交付率 $70\%$”相差 $16.7$ 个百分点。每个用例多跑几次是为了看随机性：若某例 5 次只过 3 次，它的稳定通过率是 $3/5=60\%$，不能被一次幸运回答掩盖。

## 公式推导：从每层判定到严格验收

对第 $i$ 次 run，先设 $p_i,r_i,l_i,a_i,x_i,t_i,v_i,c_i\in\{0,1\}$ 分别表示解析、检索、定位、答案、ACL、工具终态、撤销、预算是否通过。各层先各自报告，才能归因：

$$
\operatorname{Recall@k}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{G_i\cap R_i^{(k)}\neq\varnothing\}
$$

其中 $G_i$ 是标注 gold 文档集合，$R_i^{(k)}$ 是 top-$k$ 结果。只命中同名旧版本不够，定位门还要求 `(doc_id, revision, span)` 与可复现证据一致：

$$
l_i=\mathbf{1}\{\text{quoted span} \subseteq \text{retrieved revision}\}
$$

把答案拆成 $m_i$ 个可核验 claim，令 $h_{ij}=1$ 表示第 $j$ 个 claim 被它**实际引用且有权访问**的 evidence 蕴含。先报告连续的诊断指标：

$$
\operatorname{ClaimSupportRate}_i=\frac{1}{m_i}\sum_{j=1}^{m_i}h_{ij}
$$

严格门不能把 $0.9$ 当成通过；它必须是二元的：

$$
e_i=\mathbf{1}\{\operatorname{ClaimSupportRate}_i=1\}=\mathbf{1}\{\forall j,\ h_{ij}=1\}
$$

写工具时，令 $\Delta_i$ 为可观察的状态差异、$A_i$ 为允许动作集合、$s_i$ 与 $s_i^\star$ 为实际/预期终态；终态门应同时验证动作和状态：

$$
t_i=\mathbf{1}\{\Delta_i\subseteq A_i\}\cdot\mathbf{1}\{s_i=s_i^\star\}
$$

成本门把模型、检索和工具统一成货币或内部额度：

$$
C_i=q_i^{\mathrm{in}}u_{\mathrm{in}}+q_i^{\mathrm{out}}u_{\mathrm{out}}+\sum_g n_{ig}u_g,\qquad c_i=\mathbf{1}\{C_i\le B_i\}
$$

最终的严格通过率只乘**二元门**，不允许用高的 `ClaimSupportRate` 或任何别层分数补偿越权或写错终态：

$$
\operatorname{StrictPassRate}=\frac{1}{N}\sum_{i=1}^{N}p_ir_il_ia_ie_ix_it_iv_ic_i
$$

对随机模型，固定 held-out 集，报告每例 $K$ 次的均值、最差值和 $\Pr(\text{strict pass})$；发布时记录随机种子、模型版本与 judge 版本，才可比较回归前后。

预算耗尽或异常放大的轨迹也应作为拒绝负例，见 [[AI 安全/12 Unbounded Consumption 成本型 DoS|成本型 DoS]]。

## 一张图：证据与状态必须闭环

![[RAG 评估-证据链与终态.png]]

## 可运行代码：别让“答案字符串对了”骗过 harness

下面的演示只用 Python 标准库，可直接保存为 `eval_harness.py` 后执行。真实系统把 `run_system` 换成被测服务的 sandbox adapter；**held-out 标签、对抗用例和期望终态放在评测仓/CI 密钥域，不给被测模型或提示词**。代码刻意只传递结构化审计字段，不收集 raw CoT。

❌ 朴素：只查答案是否含关键字。它会放过越权检索、假引用、错误副作用和超预算。

```python
def naive_pass(trace, expected_answer):
    return expected_answer.replace(" ", "") in trace["answer"].replace(" ", "")
```

✅ 高效：以一次 run 的结构化证据验收，并重复试验汇总严格通过率。

```python
from dataclasses import dataclass
import json
from statistics import fmean


@dataclass(frozen=True)
class Case:
    case_id: str
    required_docs: frozenset[str]
    allowed_docs: frozenset[str]
    expected_answer: str
    expected_terminal: str
    allowed_effects: frozenset[tuple[str, str]]
    budget_cents: float


# 演示用数据；生产中这个 held-out split 由评测方保管，不能进入 prompt 或检索索引。
HELD_OUT = (
    Case(
        case_id="refund-policy-v7",
        required_docs=frozenset({"policy-v7"}),
        allowed_docs=frozenset({"policy-v7"}),
        expected_answer="退款窗口是30天",
        expected_terminal="completed",
        allowed_effects=frozenset({("ticket.create", "created")}),
        budget_cents=2.00,
    ),
)


def normalize(text):
    return "".join(ch for ch in text if not ch.isspace() and ch not in "。,.，")


def run_system(case, seed):
    """替换为 sandbox 中的被测 RAG；这里只返回可审计摘要，绝不返回 raw CoT。"""
    evidence = {
        "id": "policy-v7",
        "revision": "2026-06-01",
        "text": "退款窗口是30天。",
    }
    return {
        "seed": seed,
        "parsed": {"intent": "policy_lookup", "entity": "退款窗口"},
        "retrieved": [evidence],
        "citations": [{
            "claim": "退款窗口是30天",
            "doc_id": "policy-v7",
            "revision": "2026-06-01",
            "start": 0,
            "end": 8,
            "quote": "退款窗口是30天",
        }],
        "answer": "退款窗口是 30 天。",
        "acl": {"visible_doc_ids": ["policy-v7"]},
        "revocation_probe": {"after_revoke": "denied"},
        "tools": [{"name": "ticket.create", "state": "created"}],
        "terminal_state": "completed",
        "usage": {"input_tokens": 600, "output_tokens": 500, "tool_calls": 1},
    }


def score_once(case, trace):
    by_id = {doc["id"]: doc for doc in trace["retrieved"]}
    retrieved_ids = frozenset(by_id)
    citations = trace["citations"]
    localized = all(
        cite["doc_id"] in by_id
        and by_id[cite["doc_id"]]["revision"] == cite["revision"]
        and by_id[cite["doc_id"]]["text"][cite["start"]:cite["end"]] == cite["quote"]
        for cite in citations
    )
    claim_support_rate = (
        sum(
            normalize(cite["claim"]) in normalize(cite["quote"])
            for cite in citations
        ) / len(citations)
        if citations else 0.0
    )
    effects = frozenset((item["name"], item["state"]) for item in trace["tools"])
    usage = trace["usage"]
    cost = usage["input_tokens"] * 0.001 + usage["output_tokens"] * 0.002 + usage["tool_calls"] * 0.10

    checks = {
        "parse": trace["parsed"]["intent"] == "policy_lookup",
        "retrieval": case.required_docs <= retrieved_ids,
        "localization": localized,
        "answer": normalize(case.expected_answer) in normalize(trace["answer"]),
        # ClaimSupportRate 是诊断分数；严格门只有全部 claim 都被蕴含才通过。
        "citation_entailment": bool(citations) and claim_support_rate == 1.0,
        "acl": retrieved_ids <= case.allowed_docs
               and frozenset(trace["acl"]["visible_doc_ids"]) <= case.allowed_docs,
        "revocation": trace["revocation_probe"]["after_revoke"] == "denied",
        "tool_terminal": effects <= case.allowed_effects
                         and trace["terminal_state"] == case.expected_terminal,
        "cost": cost <= case.budget_cents,
    }
    return {"case_id": case.case_id, "cost_cents": round(cost, 3),
            "claim_support_rate": round(claim_support_rate, 3),
            "strict_pass": all(checks.values()), **checks}


def evaluate(cases, trials=5):
    rows = [score_once(case, run_system(case, seed))
            for case in cases for seed in range(trials)]
    rates = {key: round(fmean(row[key] for row in rows), 3)
             for key in rows[0] if key not in {"case_id", "cost_cents"}}
    return {"runs": len(rows), "rates": rates,
            "mean_cost_cents": round(fmean(row["cost_cents"] for row in rows), 3),
            "rows": rows}


if __name__ == "__main__":
    report = evaluate(HELD_OUT, trials=5)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert report["rates"]["claim_support_rate"] == 1.0
    assert report["rates"]["strict_pass"] == 1.0

    # 一条未被同一证据支持的 claim 只能拉低诊断分数，不能穿过二元严格门。
    partial = run_system(HELD_OUT[0], seed=-1)
    partial["citations"].append({**partial["citations"][0], "claim": "退款窗口是15天"})
    partial_score = score_once(HELD_OUT[0], partial)
    assert partial_score["claim_support_rate"] == 0.5
    assert not partial_score["citation_entailment"]
    assert not partial_score["strict_pass"]
```

线上接入时，把 `run_system` 放在隔离测试租户：工具用可重置的 fake/transactional sandbox，检查 `$\Delta$` 而不是只信“工具返回成功”；对撤销测例先发出凭证再撤权，第二次访问必须是 `denied`。日志中 token 与工具调用量也应抵抗刷分/作弊：被测程序不可读取 held-out ID、金标答案、judge rubric 或未来语料快照。

## 条件化基准卡：先让任务形态决定基准

| 何时选 | 卡片 | 必报指标 | 不该由它推出的结论 |
|---|---|---|---|
| 语料是多语种长文，瓶颈在找对文档 | **[MLDR](https://huggingface.co/datasets/Shitao/MLDR)**：13 种语言的长文检索集 | 每语种 Recall@k、nDCG@10、文档长度分桶 | 它主要测检索，不能直接证明最终回答忠实或工具安全 |
| 问答对象是公开公司财报/SEC 文件，且必须给证据 | **[FinanceBench](https://arxiv.org/abs/2311.11944)**：金融开放书问答 | 答案正确、evidence 覆盖、引用蕴含、每题成本 | 不要把金融集分数泛化到通用知识库或动态 Web |
| 任务含时效事实、Web/KG 检索和工具式 QA | **[CRAG Benchmark](https://arxiv.org/abs/2406.04744)**：Comprehensive RAG Benchmark | 工具路由、时效分层、答案/引用、拒答与终态 | 它是 benchmark，不等于下面的 Corrective RAG 架构 |

**名称消歧。**[CRAG Benchmark](https://arxiv.org/abs/2406.04744) 是 Yang 等人在 2024 年提出的 *Comprehensive RAG Benchmark*；论文给出 4,409 个问答对和模拟 Web/KG API。[Corrective RAG](https://arxiv.org/abs/2401.15884) 则是 Yan 等人 2024 年的 *Corrective Retrieval-Augmented Generation* 方法：先评估检索质量，再决定纠正/扩展检索。前者是考场，后者是可能参加考试的系统设计，后者见 [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]]。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1：为什么不能只用 Recall@k + answer accuracy？**

它们只覆盖“找到了吗、说对了吗”。生产 RAG 还可能引用错页/错版本、拿无权内容回答、撤权后读缓存、工具误写状态或答对但成本失控。把解析、定位、引用蕴含、ACL、撤销、side effect 和终态做成独立门，才知道哪一层坏了；最终用 strict pass 防止高分互相抵消。

**Q2：如何评“引用存在且真的支持答案”？**

先把回答切成原子 claim；每个 claim 必须映射到可访问、可回跳的 `(doc_id, revision, span)`，再做蕴含三分判定：支持/矛盾/沉默。citation precision 问“引的证据有没有用”，citation recall 问“该引的 claim 有没有都引”。链接很多但蕴含为零，仍是失败。

**Q3：如何同时测 ACL 与撤销？**

为同一问题构造允许、拒绝、过期、撤销后四类凭证；评估不仅看返回文本，还断言检索到的 `doc_id` 全在当前 allow set、拒绝原因正确。撤销测试应发生在第一轮认证后，再进行第二次检索，防止只测“新 token 天生没权限”。拒绝是安全场景下的正确终态。

**Q4：怎样防止 eval 被刷分或泄题？**

密封 held-out 集，不给被测 agent 题号、金标、judge rubric 或测试语料；数据集、语料快照、ACL 策略、工具模拟器和 judge 全部版本化；加入错引、旧版本、投毒、越权、拒答、预算耗尽等对抗负例；每例重复运行并报告分布。把评分函数写进 CI、把测试工件访问隔离，避免模型靠背题而非取证通过。

**Q5：为什么 trace 不能等同于保存 CoT？**

归因需要的是可操作的事实：取了哪版文档、span 在哪里、授权是否通过、调用了什么工具、造成了什么状态差异、花了多少资源。原始 CoT 既不是可验证的系统事实，又增加敏感数据与治理风险；用结构化事件和证据摘要足以复现大多数故障。

## 关键事实

- **[MLDR 数据集卡](https://huggingface.co/datasets/Shitao/MLDR)** 明确说明：它覆盖 13 种语言，以 Wikipedia、Wudao 与 mC4 的长文构造多语检索任务。该卡把它关联到 Chen et al. 的 **[BGE M3-Embedding](https://arxiv.org/abs/2402.03216)**（2024，arXiv:2402.03216）。它适合作为“长文、多语检索”条件卡，而非端到端 RAG 安全认证。
- **[FinanceBench](https://arxiv.org/abs/2311.11944)**（Islam et al., 2023-11-20，arXiv:2311.11944）是金融开放书问答基准；论文报告 10,231 个带答案与 evidence strings 的公开公司问题。使用时要固定题集版本、财报快照和引用抽取规则。
- **[CRAG -- Comprehensive RAG Benchmark](https://arxiv.org/abs/2406.04744)**（Yang et al., 2024-06-07，arXiv:2406.04744；NeurIPS 2024 Datasets and Benchmarks Track）包含 4,409 个问答对，并以模拟 API 覆盖 Web 与知识图谱检索。它与 **[Corrective Retrieval-Augmented Generation](https://arxiv.org/abs/2401.15884)**（Yan et al., 2024-01-29，arXiv:2401.15884）同缩写但一是基准、一是方法，不能混报结果。
- **[ALCE](https://arxiv.org/abs/2305.14627)**（Gao et al., 2023，*Enabling Large Language Models to Generate Text with Citations*，arXiv:2305.14627）把带引用生成拆成 correctness 与 citation quality 等维度；本篇把这条“逐 claim 证据归因”再连接到 ACL、修订版与终态验证。
- 成本与资源门是安全门的一部分：无界检索、重试或工具 fan-out 会把“高准确率”变成不可部署的系统风险；高风险副作用应在沙箱/人审闸门中验证。
