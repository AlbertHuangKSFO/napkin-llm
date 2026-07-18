[[15 多模态 RAG|多模态 RAG]] 是让 RAG 把同一证据保留为文本、结构、视觉和坐标等多种表征，并在命中后回到原文档对象作答；它不是“统一 embedding”与“整页图像”二选一，而是围绕 query 与语料模态建立可追溯的检索矩阵。

## 本质：检索到“哪一块证据”，必须可复现

一张图、一个表格行组和它所在 PDF 页可以同时表达同一事实。只保留 OCR 文本会丢版面，只保留页图又可能让精确文字检索昂贵。多模态 RAG 的关键因此不只是让模型看图，而是：**按对象保留互补表征，命中子对象后回到同版本的父对象、页或 crop 取证。**

类比档案馆：OCR 是卡片目录，页图是原始档案照片，caption 是人工索引。三者相互校验而非互相取代；卡片命中“总金额”后，答题人仍要打开原页里对应的表格单元格。

![[多模态 RAG-两路线.png]]

## 机制：多表征矩阵，不是一条万能索引

| 证据表征 | 适合的内容 | 常见检索方式 | 命中后的证据回跳 |
|---|---|---|---|
| text block | 段落、标题、代码、原生 PDF 文本 | lexical、dense、rerank | parent section / 原始 range |
| table object | 表头、行组、单元格、公式/显示值 | header+row-group 文本、结构特征 | 整表、表头、对应 rows/range |
| page image / crop | 扫描件、版面、图表、字体和空间关系 | 视觉单向量或多向量 late interaction | 原 page 或高分 patch 对应 crop |
| OCR / caption | 扫描文字、图中标签、图题与附近说明 | 文本索引，必要时与视觉并行 | page/figure + bbox；回看原图确认 |

CLIP 类双塔将图文编码到同一空间，适合图文互检和自然图像；对页面密集的文档，单向量会压缩大量空间结构。ColPali 论文提出把文档页图直接编码为多向量，再以 late interaction 进行细粒度匹配；论文报告其在 ViDoRe 设定中的结果，并不推出“所有企业 PDF 都该改用页图检索”。[Radford et al., 2021](https://arxiv.org/abs/2103.00020) [Faysse et al., arXiv v6 2025；ICLR 2025](https://arxiv.org/abs/2407.01449)

### 先定义摄入边界，才谈模型

| 输入 | 规范化边界 | 特别需要验证的内容 |
|---|---|---|
| PDF | document → page → text/table/figure；扫描页同时保存 page image 与 OCR | 原生文字或 OCR、双栏阅读顺序、页眉脚、图题、跨页表 |
| DOCX | document → heading/paragraph/list/table/embedded figure；必要时另存渲染页 | XML 逻辑顺序与渲染页不同；图片中的文字不能假定已被正文抽取 |
| PPTX | deck → slide → title/body/notes/table/chart/media object | slide 编号、对象坐标与层级、讲者备注是否属于可检索范围 |
| XLSX | workbook → sheet → table/range → row/cell/chart | sheet 名、公式与显示值、冻结表头、合并格、隐藏 sheet 的权限 |
| HTML | snapshot → DOM semantic object → linked asset | DOM 顺序、动态内容快照、script/style/导航噪声、资源 URL 与 hash |

解析器的输出只是候选结构。对 PDF 抽查阅读顺序，对扫描件记录 OCR 置信度和语言；对 Office 文档保留原对象 ID；对 HTML 固化抓取时间与资源 hash。官方解析器支持哪些格式是版本化能力，不等于某一份文档已经被正确理解。[Docling Supported formats，访问于 2026-07-17](https://docling-project.github.io/docling/usage/supported_formats/)

### 证据层级：子对象召回，父对象扩展，父级去重

每个 child hit 都要指向一个 parent（例如 `table-rowgroup-2 → table-4 → page-12`）。检索阶段可召回细粒度 text block、row group、figure crop 或 OCR span；构造答案上下文时沿 `parent_id` 展开表头、caption、邻近说明或父段；最后按 `(doc_id, version, parent_id)` 去重，避免同一表格的数个子块挤满 top-k。这样既不会因只送一个 cell 而失去表头，也不会因整页重复命中而浪费上下文。

应随对象保存：`doc_id`、`doc_version`、`content_hash`、`parent_id`、`block_id/type`、`page/slide/sheet`、`bbox/range`、`parser_version`、`ocr_model`、`embedding_model` 与 `acl`。命中后以这些字段回原 page/crop 或结构范围；若 hash/version 不匹配，应重新定位而非引用旧证据。

### 路由与融合必须看 query 和语料

先判断 query 需要的证据：精确名称、公式或代码通常优先 text/table；“图中箭头”“扫描发票金额”“本页布局”需要 page image/crop；不确定时并行发往多个可用表征。路由也受语料覆盖率、权限和解析质量影响——没有可靠 OCR 的扫描件不能假装是纯文本语料。

不同路由的原始分数尺度不可直接平均。若只有排序，Reciprocal Rank Fusion (RRF) 可免去直接比较原始分数：

$$
s_{\mathrm{RRF}}(d)=\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)}.
$$

例如取 $k=60$，某 page 在视觉路由第 1、OCR 路由第 3，则得 $1/61+1/63$；只在视觉路由第 2 的 page 得 $1/62$。前者因两路一致而更高。$k$、路由配额和 top-k 都要在版本化验证集调节。若有标注数据，可先按路由校准分数，或用带 query 模态、对象类型、解析置信度、ACL 等特征的 learned ranker；不要把未校准分数做裸均分。[Cormack, Clarke & Buettcher, SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)

## 可运行代码：父级去重 + RRF（不是裸均分）

```python
from collections import defaultdict

def rrf(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """rankings 的 value 按相关性从高到低排列，返回 object_id 的融合分。"""
    score = defaultdict(float)
    for route, object_ids in rankings.items():
        for rank, object_id in enumerate(object_ids, start=1):
            score[object_id] += 1 / (k + rank)
    return dict(score)

def expand_then_dedup(object_scores, metadata, limit=5):
    """child 命中后扩到 parent，并按同一文档版本的父对象去重。"""
    parents = {}
    for child_id, score in object_scores.items():
        m = metadata[child_id]
        key = (m["doc_id"], m["doc_version"], m["parent_id"])
        parents[key] = max(parents.get(key, 0.0), score)
    return sorted(parents, key=parents.get, reverse=True)[:limit]

# ❌ 不同模型的余弦/MaxSim/OCR 分数尺度不同，直接平均没有语义
# final = (text_score + page_score + ocr_score) / 3
# ✅ 每路先给排名，再 RRF；命中子对象后只保留一个父对象进入上下文
rankings = {
    "text": ["caption-7", "table-4-rowgroup-2"],
    "page_image": ["figure-7-crop-2", "table-4-rowgroup-2"],
    "ocr": ["table-4-rowgroup-2"],
}
metadata = {
    "caption-7": {"doc_id": "r1", "doc_version": "v3", "parent_id": "figure-7"},
    "figure-7-crop-2": {"doc_id": "r1", "doc_version": "v3", "parent_id": "figure-7"},
    "table-4-rowgroup-2": {"doc_id": "r1", "doc_version": "v3", "parent_id": "table-4"},
}
parents = expand_then_dedup(rrf(rankings), metadata)
```

## 评估：分开测解析、检索、定位、答案与权限

端到端回答好坏不能掩盖上游错误。每次 parser、OCR、embedding、ACL 或文档版本变更都应跑同一批带版本和权限标签的样本。

| 层 | 要回答的问题 | 例子 |
|---|---|---|
| 解析 | 对象、阅读顺序、表头与 OCR 是否正确？ | 双栏顺序、跨页表、caption-figure 绑定、OCR 错词 |
| 检索 | 正确 child/parent 是否进入候选？ | text、table、page image、crop 各路的证据覆盖 |
| 定位 | 是否回到了正确页/slide/sheet 与 bbox/range？ | 引用框是否落在真实表格行或图题 |
| 答案 | 答案是否正确、可由回跳证据支持、引用是否可打开？ | 人工或任务标注的正确性/忠实度检查 |
| ACL / 撤权 | 无权或已撤销版本会不会被检索、缓存或引用？ | 拒绝案例为 0、旧 hash 不可再返回、撤权传播时间 |

⚠️ **“页图检索绕过 OCR” vs “不需要解析”**：视觉检索可减少对 OCR 的依赖，但 citation、crop、权限和回答时的可读上下文仍需要对象与定位链。

⚠️ **“统一向量空间” vs “统一分数尺度”**：前者让某模型内的图文可比较；跨模型、跨路由分数仍要 RRF、校准或 learned fusion。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1：多模态 RAG 是两个方案二选一吗？**

A：不是。文本、表格、页图/crop、OCR/caption 是互补表征；应按 query、语料模态、解析质量和成本路由。统一图文 embedding 与页图多向量都是矩阵中的某些检索器。

**Q2：为什么 child hit 后还要 parent 扩展和去重？**

A：child 提供精确召回，parent 恢复表头、caption 或父段；按同文档版本的 parent 去重，避免同一对象的多条子块占满上下文。所有证据再按 page/slide/sheet 与 bbox/range 回原始对象。

**Q3：为什么不能把各路分数直接平均？**

A：余弦、late-interaction 和 OCR/解析置信度的尺度含义不同。可先融合排名（RRF），或在版本化标注集上校准/学习融合；路由和权重同样应由 query、对象类型及语料验证决定。

**Q4：多模态 RAG 的安全评估为什么要有撤权？**

A：索引、父对象缓存、页图和 crop 可能各自保留证据。ACL 过滤只在答案端做不够；必须验证无权对象与旧版本不会在任一路由被召回或被引用。

## 关键事实

- 多模态 RAG 的单位应是可回跳的证据对象，而不是把所有文件压成同一种文本或图像。
- PDF、DOCX、PPTX、XLSX、HTML 的摄入边界不同；PDF 的阅读顺序和扫描 OCR 必须在切块前抽样验证。
- 至少保留 text、table、page image/crop、OCR/caption 等互补表征，并用统一 parent/版本/坐标链关联。
- 命中 child 后扩展 parent，再按 `(doc_id, version, parent_id)` 去重；回答必须回原 page/crop 或结构 range。
- 融合按 query/语料模态路由；RRF、校准或 learned fusion 比未校准的裸均分更稳妥。
- 评估至少区分解析、检索、定位、答案、ACL 与撤权；模型或解析版本变化应触发回归测试。

## 来源

- Alec Radford et al. [《Learning Transferable Visual Models From Natural Language Supervision》](https://arxiv.org/abs/2103.00020), 2021。CLIP 的图文对比学习与共享表示空间。
- Manuel Faysse et al. [《ColPali: Efficient Document Retrieval with Vision Language Models》](https://arxiv.org/abs/2407.01449), arXiv v6，2025-02-28；ICLR 2025。页图多向量、late interaction 与 ViDoRe。
- [Docling Supported formats](https://docling-project.github.io/docling/usage/supported_formats/)，官方文档，访问于 2026-07-17。格式能力是版本化信息；应单独验收解析结果。
- Gordon V. Cormack、Charles L. A. Clarke、Stefan Buettcher, [《Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods》](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf), SIGIR 2009。RRF 融合公式。
