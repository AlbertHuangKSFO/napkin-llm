# RAG 学习地图

> 把外部知识喂给模型,让它答得有据、可更新、可溯源。这一域从「检索增强生成」的本质讲起,顺着一条真实管线展开:**索引怎么切、向量怎么存、查询怎么变换、召回怎么混合、结果怎么重排、生成怎么带引用**;再往上是把整条管线做成自适应环的**架构级 RAG**(Self-RAG / CRAG / Modular),按知识形态分化的**专门化 RAG**(GraphRAG / 多模态),以及 2026 年被当成一等公民的**安全与治理**(检索原生访问控制、知识运行时)。最后落到**评估选型**与**开源生态**。检索作为 agent 的工具那一面,见 Agent 域的 [[36 Agentic RAG|Agentic RAG]]。这里只放定位与链接,不写正文。

- [[RAG 面试题库]]

## ① 基础
- [[01 什么是 RAG|什么是 RAG]]
- [[02 Naive RAG 与失败模式|Naive RAG 与失败模式]]

## ② 索引与切分
- [[03 分块策略 Chunking|分块策略 Chunking]]
- [[04 Embedding 与向量数据库|Embedding 与向量数据库]]
- [[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|进阶索引：Contextual Retrieval、RAPTOR、Late Chunking]]
- [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]]

## ③ 检索与查询
- [[07 查询变换 Query Transformation|查询变换 Query Transformation]]
- [[08 混合检索 Hybrid Search|混合检索 Hybrid Search]]
- [[09 多跳检索：IRCoT、Self-Ask、FLARE|多跳检索：IRCoT、Self-Ask、FLARE]]

## ④ 后处理
- [[10 重排序 Reranking|重排序 Reranking]]

## ⑤ 生成
- [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]]

## ⑥ 架构级 RAG
- [[12 Self-RAG、CRAG 与 Adaptive RAG|Self-RAG、CRAG 与 Adaptive RAG]]
- [[13 Modular RAG|Modular RAG]]

## ⑦ 专门化 RAG
- [[14 GraphRAG 知识图谱检索|GraphRAG 知识图谱检索]]
- [[15 多模态 RAG|多模态 RAG]]

## ⑧ 安全与治理
- [[16 检索安全与访问控制|检索安全与访问控制]]
- [[17 检索数据治理|检索数据治理]]

## ⑨ 评估与选型
- [[18 RAG 评估|RAG 评估]]
- [[19 RAG vs 长上下文 vs Agentic Search|RAG vs 长上下文 vs Agentic Search]]

## ⑩ 生态
- [[20 RAG 开源生态全景|RAG 开源生态全景]]

## ⑪ Agent 与安全桥接
- [[36 Agentic RAG|Agentic RAG]]
- [[29 Deep Research Agent|Deep Research Agent]]
- [[17 Memory 与 Context Poisoning|Memory 与 Context Poisoning]]

## ⑫ 2026 混合文件与多模态 RAG
- [[21 混合文档摄入：解析、OCR、版面与阅读顺序|混合文档摄入：解析、OCR、版面与阅读顺序]]
- [[22 多模态检索的证据单元：文本、表格、图像与父子文档|多模态检索的证据单元：文本、表格、图像与父子文档]]
- [[23 多模态检索编排：路由、融合与后期交互|多模态检索编排：路由、融合与后期交互]]
- [[24 RAG 评估：证据链、归因与终态验证|RAG 评估：证据链、归因与终态验证]]
