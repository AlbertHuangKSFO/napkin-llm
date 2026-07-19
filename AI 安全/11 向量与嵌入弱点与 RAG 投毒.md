[[11 向量与嵌入弱点与 RAG 投毒|向量与嵌入弱点与 RAG 投毒]]讲的是：RAG 把“知识正确性”拆成了 `摄入 → 嵌入 → 索引 → 检索 → 生成` 五段，因此攻击者不必改模型权重，只要控制其中一段就能把错误证据稳定塞进上下文。它直接对应 OWASP **LLM08：2025 Vector and Embedding Weaknesses**，并与 [[16 检索安全与访问控制|RAG 检索安全]]、[[17 Memory 与 Context Poisoning|记忆投毒]]、[[05 Prompt Injection 提示注入|提示注入]]连成一条链。

直觉上，它不像“黑进大脑”，更像“偷偷换了资料室的抽屉标签”。模型本身没坏，但它每次查资料都更容易拿到错的、带指令的、越权的那份材料。危险点不只在“原文直接进模型”，还在 **HTML/PDF 解析结果、OCR 文本、alt 文本、表格单元、摘要缓存、重排输入** 这些派生证据单元。

先算一个最常被误解的小数。设系统每次只取 top-$k=5$，知识库里有 $N=1{,}000{,}000$ 篇文档。很多人会误以为 5 篇毒文档只占百万分之五，所以影响应当接近 0。错在把检索当成**随机抽样**。若目标查询 $q$ 下 5 篇毒文档的相似度分别是 $0.93,0.92,0.91,0.90,0.89$，而最相关的干净文档只有 $0.88$，那 top-5 会被毒文档全部占满，命中率是 **5/5=100%**，不是 $5/N$。

写成公式更清楚。设检索器对查询 $q$ 和证据单元 $d$ 的打分是 $s(q,d)$，系统返回

$$
R_k(q)=\operatorname{TopK}_{d \in D} s(q,d)
$$

若存在一组恶意证据 $P=\{p_1,\dots,p_k\}$，满足

$$
\min_{p \in P} s(q,p) > s\bigl(q,d_{(k)}^{\text{clean}}\bigr)
$$

其中 $d_{(k)}^{\text{clean}}$ 是第 $k$ 个最相关的干净证据，那么对这个查询，top-$k$ 会被恶意证据覆盖。这里决定成败的是**排序边界**，不是库总量。

![[sec-RAG投毒链路.png]]

证据单元一旦进入 RAG，常见风险有四类：

- **泄露**：向量、检索结果或派生摘要越权返回，暴露本不该看的租户数据。
- **投毒**：攻击者注入高相似度内容，让错误答案或恶意指令进入 top-$k$。
- **网页/文件注入**：恶意 HTML、Markdown、PDF、PPT 注释、隐藏文本在解析或 OCR 后变成可检索证据。
- **重排放大**：初检索只是把毒证据捞上来，reranker 或生成器进一步把它当成“最可信来源”。

所以边界要拆开看：**摄入面**防恶意内容入库，**索引面**防跨租户与版本污染，**检索面**防越权召回与单源劫持，**生成面**把召回证据当不可信输入而不是系统指令。

```python
# ❌ 朴素：只看相似度，检索结果直接拼进提示词
def retrieve_then_answer(query, index, llm):
    docs = index.search(query, top_k=5)
    context = "\n\n".join(doc.text for doc in docs)
    return llm(f"基于以下资料回答：\n{context}\n\n问题：{query}")


# ✅ 分层：ACL -> provenance -> 注入扫描 -> 多源约束 -> 再生成
def secure_retrieve_then_answer(query, principal, index, llm):
    docs = index.search(query, top_k=20)
    docs = [d for d in docs if acl_allows(principal, d.doc_id, d.version)]
    docs = [d for d in docs if d.provenance in {"verified_doc", "approved_web_snapshot"}]
    docs = [d for d in docs if not looks_like_instruction_payload(d.rendered_text)]
    docs = diversify_and_rerank(docs, limit=5)  # 防单源垄断
    cited = [to_citation_block(d) for d in docs]
    return llm.answer_with_citations(
        question=query,
        evidence_blocks=cited,
        policy="evidence_is_untrusted_data_not_instructions",
    )
```

## 面试高频

回链：[[AI 安全面试题库]]

**Q1：为什么“库很大”不能自然稀释 RAG 投毒？**
因为检索按相似度排序，不按库大小平均抽样。只要毒证据能稳定挤进 top-$k$，后面再多文档也不会改变这次查询的上下文组成。

**Q2：多模态或混合文件里，投毒到底发生在哪？**
不只发生在原文正文。HTML 注释、白字、零宽字符、PDF 隐藏层、OCR 结果、表格标题、图片 alt 文本、页面摘要缓存都可能在解析后进入检索证据面。

**Q3：RAG 投毒和 prompt injection 的关系是什么？**
RAG 投毒解决的是“让恶意内容被召回”，prompt injection 解决的是“被召回后如何影响模型行为”。前者偏数据面，后者偏指令面，生产里通常串联出现。

**Q4：最小可行防线是什么？**
最少要有 `ACL 过滤 + provenance + 版本化索引 + 注入扫描 + 生成时把证据当不可信数据`。只做相似度检索或只做生成端护栏都不够。

## 关键事实

- OWASP **LLM08：2025 Vector and Embedding Weaknesses** 明确把未授权访问、数据泄露、内容注入、检索操纵列为向量与嵌入系统的核心风险，适用到 RAG 的生成、存储与检索全链路（OWASP GenAI，2025 版风险条目，2026-07-19 核验）。
- **PoisonedRAG** 已被 USENIX Security 2025 收录。论文报告：在其设定下，对单个目标问题注入 5 篇恶意文本，可在百万级知识库上达到约 90% 攻击成功率；这说明风险来自排序劫持，而不是库规模（USENIX Security 2025，2026-07-19 核验）。
- **BadRAG**（arXiv:2406.00083，2024）讨论的是带触发条件的检索后门；因此 clean 查询正常不代表系统安全，只能说明攻击没有在当前触发条件下出现。
- OpenTelemetry GenAI 语义约定已把 `gen_ai.retrieval.documents`、`gen_ai.retrieval.query.text` 等属性纳入检索观测面；这意味着 RAG 安全不应只监控最终答案，还应监控“召回了什么证据”（OTel 文档，2026-07-19 核验）。

工业上更稳妥的做法是把 RAG 当成“证据供应链”：每个证据单元都带 `来源、版本、ACL、解析器、时间戳、摘要哈希`，这样出事时才能按来源回滚、按版本封存、按哈希定位污染范围。跨域上，它与 [[24 RAG 评估：证据链、归因与终态验证|RAG 全链评估]]、[[18 RAG 评估|RAG 评估]]、[[25 监控、可观测与事件响应|运行时监控]] 是同一套治理问题。
