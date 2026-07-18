[[18 RAG 评估|RAG 评估]]是给一个 [[01 什么是 RAG|RAG]] 系统建立**可归因、可复现、可追责**标尺的方法：同一条答复既可能是没捞到证据，也可能是拿到证据后仍答错，还可能是引用、工具执行或业务状态出了问题。只报一个“回答正确率”会把这些故障揉成黑盒；评估应把它拆成五条独立证据线，并把每次判定可回放地记录下来。

## 本质：从两段归因升级为五条证据线

一次普通问答至少有检索→生成两段；一旦答案带引用、会调工具或会改变业务状态，端到端质量还取决于“引用是否真的支撑 claim”与“动作是否真的达成”。因此评估顺序不是只看一个总分，而是以下五线并排：

| 证据线 | 固定输入 | 核心问题 | 典型指标 | 不可替代的原因 |
|---|---|---|---|---|
| ① gold-evidence retrieval | query + 人工 gold 文档/段落 ID | 该捞回的证据是否在 top-k？ | Recall@k、MRR、nDCG、ID precision | 不让生成器替检索器背锅 |
| ② oracle-context generation | query + **gold evidence** + reference | 若证据完美，生成器能否正确、完整地使用它？ | correctness、faithfulness、拒答正确率 | 隔离生成能力上限 |
| ③ retrieved-context generation | query + **真实召回上下文** + reference | 真实管线最后答得怎样？ | faithfulness、response relevancy、correctness | 衡量用户真正看到的体验 |
| ④ citation attribution | answer claims + citation + 对应文档版本 | 每个 claim 的引文是否存在、可定位、可蕴含？ | citation precision / recall、claim entailment | “有链接”不等于“被链接支撑” |
| ⑤ tool / business terminal state | 多步 trace + 预期终态/receipt | 工具、授权和最终业务结果是否正确？ | ToolCallAccuracy/F1、终态成功率、补偿率 | 文本答对不代表订单、审批或退款已正确完成 |

![[RAG 评估-二维.png]]

前两线给出“检索器”和“生成器”的单独诊断，第三线是端到端现实；后两线把 [[11 生成层：引用归因与忠实度|引用归因与忠实度]] 与 Agent 轨迹评估接回同一个验收面。[[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]] 的在线 critic 也可消费这些信号，但离线基准不能被在线自评取代。

高风险集还应同时覆盖 [[AI 安全/11 向量与嵌入弱点与 RAG 投毒|向量与嵌入投毒]]、[[AI 安全/12 Unbounded Consumption 成本型 DoS|成本型 DoS]] 与 [[AI 安全/24 沙箱、最小权限与人审闸门|最小权限与人审闸门]] 的负例。

**生活类比。**像医院接诊：先核验化验单是否拿对（gold retrieval），再看医生拿到正确化验单时能否做对判断（oracle generation），最后核对真实材料、病历引用和处方是否实际执行。只看“病人好没好”无法告诉你该修取样、诊断、归档还是执行环节。

## 机制

### 检索与生成各量什么

- **gold retrieval**：`Recall@k = 命中的 gold evidence 数 / gold evidence 总数`；MRR 看第一个正确证据是否靠前，nDCG 对分级相关性和位置同时打折。它只依赖 query 与 gold evidence，**不依赖 response/reference**。
- **oracle generation**：把 gold evidence 而非召回结果喂给同一生成器。若此线仍低，先修 prompt、模型、拒答策略或上下文压缩；不要先调 embedding。
- **real retrieved generation**：对真实 `retrieved_contexts` 打 groundedness / faithfulness、response relevancy 和 reference correctness。它依赖的字段随指标不同，不能把“所有指标都 reference-free”当作事实。
- **Context Precision 与 Context Utilization 不可混写**：v0.4 collections `ContextPrecision` 将每个 retrieved context 同 **reference** 比较，是 reference-based 的“片段是否有用且靠前”；没有 reference 时可用 `ContextUtilization`，它同 **response** 比较上下文的使用情况，是 response-based 的代理指标，不能证明外部真值或 gold evidence。

**Faithfulness 不等于 Correctness。**faithfulness 问“答案中的 claim 能否由**当前上下文**推出”；correctness 问“答案是否符合**外部真值/reference**”。上下文本身过期时，回答“严格引用了过期文档”可以 faithful 却 incorrect；模型凭参数记忆答对却未被上下文支撑，可以 correct 却 unfaithful。二者必须同时报，且 oracle / retrieved 两个生成条件都要分别报。

**小手算。**reference 有 4 个关键 claim，真实召回上下文只支持其中 3 个；答案有 3 个 claim，其中 2 个被其引用上下文支持，另一个错误地把 NeurIPS 写成 ICML：

$$
\text{Context Recall}=\frac{3}{4}=0.75,\qquad
\text{Faithfulness}=\frac{2}{3}\approx0.67
$$

它说明既有“证据漏召回”，也有“生成越过证据”；不能只用 0.67 断言检索无罪。

### 指标的字段依赖（先问数据够不够）

| 指标 | 依赖 response | 依赖 reference / gold | 依赖 retrieved contexts | 读法 |
|---|---:|---:|---:|---|
| Recall@k、MRR、nDCG | 否 | gold evidence | 召回 ID / 排名 | 纯检索线 |
| Context Precision | 否 | 是（reference） | 是 | 召回片段是否有用且靠前 |
| Context Utilization | 是 | 否 | 是 | response 用到了多少真实召回上下文；无 reference 的代理 |
| Context Recall | 否 | 是（reference） | 是 | reference 的 claim 是否被上下文覆盖 |
| Faithfulness | 是 | 否 | 是 | response claim 是否被上下文支持 |
| Response Relevancy | 是 | 否 | 否 | 是否对准用户问题，**不评事实真伪** |
| Correctness / Factual correctness | 是 | 是 | 视实现而定 | response 与外部 reference 是否一致 |
| Citation precision / recall | 是（claim） | 覆盖率需 gold claim | 是（被引文档版本） | 引得对不对、该引的有没有引 |
| Tool / terminal-state metric | 可选 | 预期调用或终态 | trace / receipt | 以结构化执行与业务结果为准 |

Ragas 的 Faithfulness 会将 response 拆成 claim，再检验每条是否可从 `retrieved_contexts` 推出；官方定义正是“被支持 claim 数 / 全部 claim 数”。Response Relevancy 则从 response 反推问题并算嵌入余弦相似度，因此它不应被当作正确率。

### 裁判、人审与一致性

LLM-as-judge 有位置、冗长与 self-preference 偏置，并且服务端模型别名、隐藏 prompt 与采样都可能变化。每次 run 都写入不可变 `run_manifest.json`：

```json
{
  "dataset_revision": "gold-rag-2026-07-17",
  "ragas": "0.4.3",
  "judge": {"provider": "…", "model": "…", "model_revision": "…", "temperature": 0, "seed": null},
  "embedding": {"model": "…", "revision": "…"},
  "metric_prompt": {"name": "Faithfulness", "source_or_hash": "…"},
  "retriever": {"index_revision": "…", "top_k": 8},
  "git_commit": "…"
}
```

`seed` 只有在供应商实际接受且回显时才记录数值；否则明确写 `null`，不能伪装成可复现。对每个裁判版本保留双人独立人审样本，分歧交由第三人裁决；同时在固定样本上重复 judge，报一致率/方差，并将 judge 与人审的一致性作为校准门槛。[[Agent/38 Agent 评估与可观测性|Agent 评估与可观测性]] 中的轨迹裁判需要同样的审计。

### 必须可回放的 trace

一条 trace 至少保存：`query` 与 locale、检索 `doc_id/chunk_id/doc_version`、response 的 `claim → citation_id → span` 映射、每次工具的请求/响应摘要、授权/拒绝决策及 policy revision、幂等键、最终 **receipt**（例如订单 ID、审批事件 ID 或只读查询快照）。敏感原文、PII 和工具密钥依 [[17 检索数据治理|数据驻留]] 做最小化与访问控制；“留 trace”不是无限制留数据。

## 可运行的最小代码：锁定 Ragas 0.4.3 collections API

下面是**执行日（2026-07-17）官方 release 线 `ragas==0.4.3`** 的 collections API。v0.4 仍保留 `from ragas import evaluate` 与 legacy `ragas.metrics` import 作为**已弃用**的兼容入口，但新项目不应再以它们为模板；实际风险是把 pre-v0.4 示例的已改名字段、已移除 API 或不匹配的安装版本混入当前代码，而不是旧 import 必然失败。先把直接依赖固定，再用 lock 文件固定完整解析结果与哈希：

```bash
# requirements.in：只放人工维护的直接依赖
echo 'ragas==0.4.3' > requirements.in

# 每次升级时在干净、受控的 Python 3.11 环境重新解析；提交 requirements.lock
uv pip compile requirements.in --generate-hashes -o requirements.lock
uv pip sync requirements.lock

# 运行前必须由调用者提供真实 API key、judge 的版本化模型名与记录值
export OPENAI_API_KEY='…'
export RAGAS_JUDGE_MODEL='你的版本化 judge model'
export RAGAS_EMBEDDING_MODEL='你的版本化 embedding model'
python eval_one.py
```

```python
# eval_one.py  —— 未安装依赖、未提供 key/模型时会失败；本库没有声称已实跑。
import asyncio, json, os
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    AnswerRelevancy, ContextPrecision, ContextRecall, ContextUtilization, Faithfulness,
)

JUDGE_MODEL = os.environ["RAGAS_JUDGE_MODEL"]
EMBEDDING_MODEL = os.environ["RAGAS_EMBEDDING_MODEL"]

async def main() -> None:
    client = AsyncOpenAI()
    judge = llm_factory(JUDGE_MODEL, client=client)
    embeddings = embedding_factory("openai", model=EMBEDDING_MODEL, client=client)
    row = {
        "user_input": "RAG 最初由谁在何年提出？",
        "response": "Lewis 等人在 2020 年提出了 RAG。",
        "retrieved_contexts": [
            "Lewis et al. 在 2020 年的 NeurIPS 论文提出 Retrieval-Augmented Generation。"
        ],
        "reference": "Lewis 等人在 2020 年提出 Retrieval-Augmented Generation（RAG）。",
    }
    results = {
        "faithfulness": await Faithfulness(llm=judge).ascore(
            user_input=row["user_input"], response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
        ),
        "response_relevancy": await AnswerRelevancy(llm=judge, embeddings=embeddings).ascore(
            user_input=row["user_input"], response=row["response"],
        ),
        "context_precision": await ContextPrecision(llm=judge).ascore(
            user_input=row["user_input"], reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        ),
        "context_utilization": await ContextUtilization(llm=judge).ascore(
            user_input=row["user_input"], response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
        ),
        "context_recall": await ContextRecall(llm=judge).ascore(
            user_input=row["user_input"], reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        ),
    }
    print(json.dumps({name: result.value for name, result in results.items()}, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
```

❌ 直接照抄 pre-v0.4 示例中的 `ground_truths`：文档化的字段迁移是 `ground_truths → reference`（单字符串）。`evaluate()` 与 legacy `ragas.metrics` import 在 v0.4 仍受支持但已弃用；真正会报错的是旧字段、已移除 API 或与安装版本不匹配，而不是“旧 import 必然失败”。
✅ 新项目从 `ragas.metrics.collections` 导入，以 `ascore(**metric_fields)` 传字段，并从 `MetricResult.value` 取分。本例故意让每个 metric 只接收其所需字段：`ContextPrecision` 接 `reference + retrieved_contexts`，`ContextUtilization` 接 `response + retrieved_contexts`；不要把二者都叫作 reference-free precision。

若要将一条记录作为样本保存，v0.4 仍可使用 `SingleTurnSample`；但 collections metric 接收的是关键字字段，而不是 `single_turn_ascore(sample)`：

```python
from ragas.dataset_schema import SingleTurnSample

sample = SingleTurnSample(
    user_input="RAG 最初由谁在何年提出？",
    response="Lewis 等人在 2020 年提出了 RAG。",
    retrieved_contexts=["Lewis et al. 在 2020 年的 NeurIPS 论文提出 RAG。"],
    reference="Lewis 等人在 2020 年提出 Retrieval-Augmented Generation。",
)
row = sample.model_dump(exclude_none=True)
faithfulness_fields = {
    "user_input": row["user_input"],
    "response": row["response"],
    "retrieved_contexts": row["retrieved_contexts"],
}
# await Faithfulness(llm=judge).ascore(**faithfulness_fields)  # collections API，不传 sample 对象
```

代码的依赖、模型、密钥和网络均不在本知识库中，因而这里只做 API/语法级示例，**不把它写成已执行的分数**。Ragas 0.4 的迁移还将 `ground_truths: list[str]` 改为 `reference: str`；若需要多 reference，应拆成独立评估记录而不是沿用旧字段。

## 选型卡：基准、采样与 CI

| 需要回答的问题 | 选择 | 应报告什么 | 约束 / 不能替代什么 |
|---|---|---|---|
| 多语言、长文档检索是否退化？ | **MLDR（Multilingual Long-Document Retrieval）** | language 分组的 Recall@k、nDCG@10、query/doc 长度桶 | 检验 retriever，不等于业务 QA 或引用质量 |
| 财报、10-K/10-Q 证据链是否可靠？ | **FinanceBench** | 答案正确性、evidence/citation 归因、拒答 | 金融域与其数据许可；不能外推到通用客服 |
| 动态、长尾事实及 web/KG API 的端到端表现？ | **CRAG（Comprehensive RAG Benchmark）** | correct / missing / incorrect，按动态性与实体热度分组 | **不是** Corrective RAG；后者是 Yan et al. 的检索纠错方法 |
| 自有高风险动作会不会误执行？ | 自有 trace + receipt gold set | tool 参数、授权决策、终态、补偿 | 公共 QA 基准没有你的 ACL 与业务副作用 |

公共集用于发现能力边界；发布门禁仍以自己的 versioned gold set 为准。风险分层采样至少交叉以下维度：影响（只读/可逆/资金或权限）、证据难度（单跳/多跳/表格或图像）、时效（静态/高变）、用户与语言、ACL 边界（可见/越权诱导）、行为（回答/拒答/工具动作）。每个高风险桶保留人工判定和 replay trace；不要让高频、低风险 query 稀释总分。

CI 做两层：每个 PR 跑小而稳定的 gold 子集，任何关键桶检索、citation、终态回退即阻断；定时任务跑完整集、重复 judge 和人工抽检。门槛以“相对已批准基线 + 最低绝对下限 + 置信区间”表达，并固定数据、索引、judge、prompt 与代码 revision，避免把版本漂移误报成模型进步。

## 关键事实

- RAGAS：Es et al.（2023），*RAGAS: Automated Evaluation of Retrieval Augmented Generation*，arXiv:2309.15217，提出以 faithfulness、answer relevance、context relevance 做 reference-free / 弱 reference 诊断的早期框架。
- Ragas 官方 v0.4 迁移文档：metrics 移入 `ragas.metrics.collections`，新 API 用 `ascore(**kwargs)` 返回 `MetricResult`；`evaluate()` 与 legacy `ragas.metrics` import 仍受支持但已弃用，新工作流应使用 collections API。
- ARES：Saad-Falcon et al.（2023），arXiv:2311.09476，以少量人工标注和 prediction-powered inference 校准裁判偏差；它支持“judge 必须有人审锚点”，而非替代人审。
- ALCE：Gao et al.（2023），arXiv:2305.14627，以 citation precision / recall 区分“引文支持该句”和“该引的句子都有引文”。
- FinanceBench：Islam et al.（2023），arXiv:2311.11944；CRAG（Comprehensive RAG Benchmark）：Yang et al.（2024），arXiv:2406.04744。缩写 **CRAG** 在论文语境中有歧义，首次出现必须写全称。

## 工业界实践

1. **数据资产**：版本化保存 query、gold evidence ID、reference、预期拒答或终态；文档更新时让 `doc_version` 触发样本复审，而不是悄悄沿用旧金标。
2. **先做三次生成**：同一 query 先跑 gold retrieval，再用 oracle context 生成，最后用真实 retrieved context 生成。只在第三次低分就改 prompt，很容易掩盖召回问题。
3. **引用与动作单列验收**：claim 先拆分，再做 claim→citation span 蕴含；工具调用以结构化 request/response、授权和 receipt 判定，不接受模型文本中的“已完成”。
4. **人审闭环**：高风险桶全审，其他桶按风险抽检；记录标注指南版本、标注者和分歧。把人审纠错回灌到 gold set 和 judge 校准集。
5. **线上与离线分工**：离线集用于可重复的回归，线上按风险抽样抓新 query、漂移和失败 receipt。线上样本经脱敏、去重和人审后才进入金标，不能让模型自己的评分直接变成真值。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1：RAG 答错，怎样知道是检索还是生成？**
先用 gold evidence 跑 Recall@k/nDCG；再把 gold evidence 喂给生成器（oracle-context generation）；最后跑真实 retrieved-context generation。oracle 都错是生成问题，oracle 对而真实错通常是检索/上下文构造问题；还要单验 citation 与业务终态。

**Q2：Faithfulness 高是否说明答案正确？**
不说明。它只说明答案可由当前上下文推出；上下文错或过期时可 faithful 但 incorrect。correctness 需要 reference/外部真值，二者在 oracle 与真实上下文条件下都要报。

**Q3：Ragas 0.4 怎么避免照抄旧博客？**
锁 `ragas==0.4.3` 与完整 `requirements.lock`，新代码用 `ragas.metrics.collections`、`ascore(**kwargs)`、`MetricResult.value`，字段用 `reference` 而非 `ground_truths`。模型、prompt、seed 和数据/索引 revision 一起写进 run manifest。

**Q4：为什么还要评 citation 和工具终态？**
答案文字即使正确，也可能错引、漏引、调用了未经授权的工具，或生成成功却未产生业务 receipt。claim→citation 蕴含与 trace→terminal state 是两条独立验收线。

## 知识拓展

- `Response Relevancy` 通过“由 response 反推问题”与原问题的嵌入余弦来近似切题度，底层数学见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]]；它不检查外部事实。
- 评估不是只给离线报表用。[[13 Modular RAG|Modular RAG]] 的可插拔模块要配合按线分解的回归门禁，才能判断替换 chunker、embedding 或 reranker 的实际影响。
- 多模态输入要把图页、表格和文本的 evidence version 一起写入 trace；只留 OCR 文本会让后续 citation 无法解释原图表为何被支持，详见 [[15 多模态 RAG|多模态 RAG]]。

## 来源

- [Ragas v0.4 migration guide](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/) 与 [Faithfulness collections API](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)（核验日：2026-07-17）。官方示例说明 `ascore(**kwargs)`、`MetricResult.value`、`reference` 字段和旧 API 的迁移状态。
- [Ragas releases](https://github.com/vibrantlabsai/ragas/releases)（核验日：2026-07-17）：v0.4.3 为本篇锁定的稳定 release；[Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)、[Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)、[Response Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) 的字段依赖以官方文档为准。
- [M3-Embedding（含 MLDR benchmark）](https://aclanthology.org/2024.findings-acl.137/)、[FinanceBench 官方仓库](https://github.com/patronus-ai/financebench)、[CRAG（Comprehensive RAG Benchmark）官方仓库](https://github.com/facebookresearch/CRAG)；[Corrective Retrieval-Augmented Generation](https://arxiv.org/abs/2401.15884) 是不同方法。
- Es et al.（2023）arXiv:2309.15217；Saad-Falcon et al.（2023）arXiv:2311.09476；Gao et al.（2023）arXiv:2305.14627。
