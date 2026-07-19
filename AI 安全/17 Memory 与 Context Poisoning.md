[[17 Memory 与 Context Poisoning|记忆投毒]]的本质是：攻击者不再只争夺“这一轮 prompt”，而是争夺 **未来会被再次检索、再次采信的状态**。OWASP 在 2025-12 发布的 Agentic Top 10 中把它单列为 **ASI06 Memory & Context Poisoning**，因为一旦不可信内容被写进 durable memory、共享摘要或长期检索层，影响面会从单轮回答扩到后续决策。

直觉类比是“便签本污染”。一次性 prompt injection 像有人在你桌上放了一张误导纸条；session 结束纸条就没了。记忆投毒像有人把那句话抄进了你的常用笔记本、通讯录或 SOP 卡片里，以后每次翻本子都可能再信一次。

先把持久化边界算清。假设一条低可信记忆每天被召回 5 次、保留 14 天、被 10 个共享同一团队记忆空间的用户都可能命中，那么潜在触发次数是

$$
5 \times 14 \times 10 = 700
$$

次。一次写入，不是“一次攻击”，而是**700 次未来决策暴露机会**。这也是为什么 durable memory 的写入门禁，比单轮 prompt 过滤更要保守。

更一般地，若某条记忆的日均召回次数为 $r$、驻留天数为 $d$、潜在受影响主体数为 $u$，则暴露面可粗略写成

$$
E = r \cdot d \cdot u
$$

其中 session-only context 的 $d$ 往往接近 0，而 durable memory、共享摘要、长期向量记忆的 $d$ 可能很大。**不是所有 context 都会持久化**；真正危险的是那些会跨会话、跨任务、跨主体复用的部分。

![[sec-记忆投毒持久化.png]]

工程上最好把它拆成三层：

- **session context poisoning**：污染当前会话窗口，本轮结束通常消失。
- **durable memory poisoning**：污染长期记忆、摘要、偏好、项目状态、向量记忆，后续会反复召回。
- **retrieval-fed memory poisoning**：先污染 RAG，再被“总结”“写回”到长期记忆，形成二次持久化。

成立条件也要讲清：

1. 系统必须存在**写入路径**，例如自动总结、偏好学习、对话沉淀、项目状态回写。
2. 该路径缺少**信任分级**，把网页、工具输出、他人输入与经验证事实同等处理。
3. 后续检索缺少 **provenance、TTL、撤权和回滚**，导致错误状态能被持续复用。

```python
# ❌ 朴素：任何外部内容都可直接写成长效记忆
def save_memory(text, source, store):
    store.put({"text": text, "source": source})


# ✅ 分级：不可信内容先进待核区，带 provenance / TTL / 租户边界
def save_memory(text, source, principal, store):
    trust = classify_source(source)  # verified_doc / user / tool / web / other_agent
    record = {
        "text": text,
        "source": source,
        "tenant": principal.tenant,
        "owner": principal.user_id,
        "trust": trust,
        "provenance": make_provenance(source),
        "ttl_hours": 24 if trust != "verified_doc" else 24 * 30,
        "state": "pending_review" if trust != "verified_doc" else "active",
    }
    if looks_like_instruction_payload(text):
        quarantine(record)
        return "blocked"
    store.put(record)
    return record["state"]
```

防御重点不是“绝不记忆”，而是**把记忆当数据库对象治理**：

- durable memory 与 session scratchpad 分开。
- 写入时强制 `trust tier / provenance / tenant / ttl / revocation handle`。
- 召回时对低可信记忆降权，关键动作回源验证，不直接当事实。
- 团队共享记忆比个人记忆更高风险，应有更严格审批与更短默认 TTL。

## 面试高频

回链：[[AI 安全面试题库]]

**Q1：记忆投毒和普通 prompt injection 的本质差别是什么？**
prompt injection 主要影响当前轮；记忆投毒影响之后的多轮、多任务甚至多主体，因为被污染的是会复用的状态，而不是一次性输入。

**Q2：是不是所有上下文污染都算记忆投毒？**
不是。只有进入 durable memory、长期摘要、共享状态或可反复检索层，才具备记忆投毒的持续性风险。临时上下文污染更接近一次性注入。

**Q3：为什么 provenance 和 TTL 比“更强分类器”更关键？**
分类器会漏报。provenance 决定你能否事后追溯来源，TTL 决定错误能活多久，revocation handle 决定你能否批量撤回。它们是止损与回滚的基础设施。

**Q4：共享团队记忆为什么比个人记忆危险？**
因为 $u$ 变大了。按 $E=r \cdot d \cdot u$ 看，跨用户共享会把同一条毒记忆的影响面直接放大。

## 关键事实

- OWASP 在 **2025-12-09** 发布的 Top 10 for Agentic Applications 中将 **ASI06** 定义为 **Memory & Context Poisoning**；官方后续说明强调，风险核心不是“看见恶意内容一次”，而是“系统把它持续带到未来推理与行动里”（OWASP，2026-07-19 核验）。
- OWASP 在 **2026-05-13** 的《Memory Is a Feature. It Is Also an Attack Surface》明确指出：memory file、hook、local configuration 都可能成为 agent 的 trusted operating environment，一旦被污染，会持续影响 planning、tool use 与行为。
- 这类风险通常与 [[11 向量与嵌入弱点与 RAG 投毒|RAG 投毒]] 串联：恶意检索内容先被召回，再被自动摘要或偏好学习写回长期状态，形成“先检索污染，再记忆固化”。
- OpenTelemetry GenAI 语义约定已把 `gen_ai.retrieval.documents` 等属性纳入可观测范围；因此 durable memory 与 retrieval memory 的治理需要监控“写入了什么、从哪来、之后又被谁召回”。

如果把模型当作“推理引擎”，那记忆系统就是它的“状态数据库”。数据库不做租户隔离、字段来源、TTL 与撤权，迟早会从准确率问题演变成权限问题、审计问题和事故响应问题。这也是它和 [[19 Agent 记忆系统|Agent 记忆系统]]、[[20 上下文工程|上下文工程]]、[[25 监控、可观测与事件响应|监控与 IR]] 必须一起设计的原因。
