[[03 分块策略 Chunking|分块策略 Chunking]] 是把**已解析且可追溯的文档对象**切成检索单元的过程；目标不是套一个固定数字，而是在不破坏对象语义、权限和定位信息的前提下，让 [[04 Embedding 与向量数据库|embedding]] 能匹配 query，再把命中的子对象扩回足够的父上下文。

## 本质：分块是“对象边界 + 模型边界”的协商

把整篇文档压成一个向量，多个主题会彼此稀释；切得过小，又会把限定条件、指代和表头切走。因此真正的矛盾仍是：**小单元利于定位，大上下文利于解释**。[[06 检索粒度：父文档与句子窗口检索|small-to-big]] 用“小对象召回、父对象扩展”把两者解耦。

但“长文本”不是唯一对象。把一本混排报告当作一卷字符，像把仓库里所有零件倒在同一箱：螺丝可以按长度分，电路板却不能被锯开。分块前应先判断它是 text、list、table、figure、caption、code，还是 page、slide、sheet 这样的容器；对象边界错了，后面的 token 数调得再精也无济于事。

![[分块策略 Chunking.png]]

## 机制：先解析对象，再决定文本窗口

### 文本基线不是通用标准

对阅读顺序可靠的**连续文本**，递归按段落、句子、词等自然边界切，是低成本 baseline；固定 token 窗口是它的保底退化，而不是所有语料的默认答案。语义分块可在句间语义变化处断开，但阈值、语言和语料相关，须与基线做离线对照。结构感知也不是天然“最高质量”：只有解析器给出的层级、阅读顺序和对象关系可信时，它才值得优先。

`256–512 token` 与 `10–20% overlap` 可以作为**纯文本试验的起点**，不可写成生产规范。先查所选 encoder 的最大输入、特殊 token 开销、检索任务上的有效长度和语言表现；再以答案集评估。反例是 BGE-M3：其论文报告可处理最长 8192 input tokens，说明“embedding 都只有 512 token”并不成立；反过来，能输入 8192 也不证明一整个 8192-token 文档应只做一个向量。[Chen et al., 2024](https://arxiv.org/abs/2402.03216)

### 对象类型决定不能切坏什么

| 对象 | 先保留的边界 | 检索单元与父子关系 |
|---|---|---|
| text | 标题、段落、句子、脚注归属 | 长 section 再按 token 切；每块保留 section/parent |
| list | 列表层级、序号、引入句 | item 可为子块，整个 list 为父对象；不要把嵌套层级拆断 |
| table | 表题、表头、合并单元格、行组 | `header + row group` 为子块，table 为父对象；命中行组时一并取表头和表题 |
| figure / caption | 图、图题、相邻说明的绑定 | 图像/crop、OCR、caption 可是不同表征，但共享同一 figure parent |
| code | 文件、module、class、function、依赖接口 | 函数/类优先；超长函数再按语句或 AST 子树切，保留签名与文件路径 |
| page / slide / sheet | 页、幻灯片、工作表及其坐标系 | 容器用于定位和扩展，不应仅因“每页”就抹平其内部 text/table/figure 对象 |

PDF 尤其要先验证阅读顺序：抽样检查双栏、页眉页脚、跨页表、脚注与图题是否归位；扫描 PDF 还要区分 OCR 输出与原生文本。顺序未通过，就保留 page image / OCR 的并行表征并标记待复核，不能把错误顺序的文本直接递归切块。DOCX、PPTX、XLSX、HTML 也应分别沿逻辑段落/表格、slide canvas、sheet-range、DOM 语义树解析，而不是先粗暴转成一段纯文本。Docling 的官方文档把这些输入格式归入统一文档表示；它是可选实现，不是解析正确性的保证。[Docling 官方文档，访问于 2026-07-17](https://docling-project.github.io/docling/usage/supported_formats/)

### overlap 只服务于连续文本的边界风险

令正文 token 总数为 $L$，窗口大小为 $S$，重叠为 $O$，且 $0\le O<S$。相邻窗口步长为

$$
\Delta=S-O,\qquad
W_i=[i\Delta,\ \min(i\Delta+S,L)).
$$

当 $L>S$ 时，窗口数为

$$
N=1+\left\lceil\frac{L-S}{S-O}\right\rceil.
$$

例如 $L=1000,S=512,O=64$，则 $\Delta=448$，$N=1+\lceil488/448\rceil=3$；三窗从 token 0、448、896 起始。重叠让跨窗口的一句条件完整出现在后窗，却也带来重复嵌入和重复召回。对表格行组、代码函数、图题绑定等**有天然对象边界**的内容，优先用父对象扩展或显式上下文，而不是机械 overlap；是否需要重叠应由边界错误样本和离线评估决定。

## 命题分块：更小，但必须回到证据对象

命题(proposition)是去指代且自包含的原子事实，例如把“它在 1955 年去世”改写为“爱因斯坦在 1955 年去世”。Chen et al. 在 passage、sentence、proposition 粒度上比较检索，报告命题作为检索单元在其 QA 设置中的优势；这不是所有语料的普适结论。它需 LLM 改写、可能损失叙事关系，故应把命题作为可追溯子对象，命中后回到原 paragraph/table/figure 父对象取证。[Chen et al., 2023，EMNLP 2024](https://arxiv.org/abs/2312.06648)

## 可运行代码：真 tokenizer 的 token 窗口（仅文本 baseline）

下面用所选模型的 tokenizer 计数，并实际生成 overlap 窗口；`max_input_tokens` 必须由**该模型版本的配置/模型卡和一次真实 encode**确认，不能从 `len()` 或别的 encoder 借用。它故意不处理表格、代码、页图等对象，它们应先走上节的结构解析。

```python
# 依赖：pip install transformers；首次加载模型需本地缓存或获准下载。
from transformers import AutoTokenizer

MODEL_ID = "BAAI/bge-m3"  # 换模型时，连同 max_input_tokens 一起复核
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def token_windows(text: str, *, max_input_tokens: int,
                  chunk_tokens: int = 512, overlap_tokens: int = 64):
    """返回真正的 tokenizer token window；special tokens 预留在上游配置中处理。"""
    if not 0 <= overlap_tokens < chunk_tokens <= max_input_tokens:
        raise ValueError("要求 0 <= overlap < chunk <= 已验证的模型输入上限")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    stride = chunk_tokens - overlap_tokens
    windows = []
    for start in range(0, len(token_ids), stride):
        end = min(start + chunk_tokens, len(token_ids))
        windows.append({
            "token_range": [start, end],
            "text": tokenizer.decode(token_ids[start:end],
                                     skip_special_tokens=True),
        })
        if end == len(token_ids):
            break
    return windows

# ❌ len(text) 是字符数，且“此处略 overlap”会让参数承诺落空
# chunk = text[:512]
# ✅ 真 tokenizer + 显式步长（overlap 可为 0）；BGE-M3 的 8192 是论文报告的模型能力，不是本函数的默认块大小
chunks = token_windows("一段待切的连续文本", max_input_tokens=8192,
                       chunk_tokens=512, overlap_tokens=64)
```

切块时，最小 metadata 至少包含如下字段；`bbox` 与 `range` 允许命中后精确回跳，而 `version`、`hash` 和 ACL 让重建与撤权有据可查。

```yaml
doc_id: policy-2026-07
doc_version: 2026-07-17
content_hash: sha256:...
block_id: table-4-rowgroup-2
block_type: table_row_group
parent_id: table-4
page: 12                 # PDF；不适用则为 null
slide: null              # PPTX
sheet: null              # XLSX
bbox: [72, 314, 528, 486] # 原始容器坐标系
range: {char: [2480, 3120], token: [448, 960], rows: [12, 24]}
parser_version: docling-...
chunker_version: token-window-v2
embedding_model: BAAI/bge-m3@...
acl: [finance-read]
```

## 选型卡与常见误区

| 情况 | 起步策略 | 验证重点 |
|---|---|---|
| 连续 prose | 自然边界递归 + 小窗口 | encoder 实际 token、边界问题、答案证据覆盖 |
| Markdown / HTML | 标题或 DOM 对象后再切过长正文 | 语义层级、导航/样式噪声、DOM 顺序 |
| PDF（原生/扫描） | 先做页面级解析与阅读顺序验收；保留 OCR / page image | 双栏、页眉脚、跨页表、OCR 与 bbox 回跳 |
| 表格 / 工作表 | header + row group / range，父表或父 sheet 扩展 | 表头、合并单元格、公式/显示值、跨页关系 |
| 图文/代码密集 | figure-caption 或 AST 对象为先 | caption 是否随图、函数签名/依赖是否仍可读 |

⚠️ **“所有块都 overlap” vs “对象扩展”**：前者只复制邻近 token，后者恢复真正需要的表头、图题或父段。对象内容优先后者。

⚠️ **“模型最大长度” vs “有效 chunk 长度”**：前者是能否编码的硬约束；后者要由 query、语料、成本和端到端答案评估决定。两者不可互推。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1：为什么不能用一个固定 chunk_size 解决所有 RAG？**

A：chunk_size 同时受对象边界、encoder 输入与检索任务约束。256–512 只能是连续文本的试验起点；例如 BGE-M3 论文明确给出 8192-token 能力。先保住 table/figure/code 等对象边界，再用真实 tokenizer 与标注 query 调参。

**Q2：PDF 为什么应先验阅读顺序？**

A：双栏、页眉脚、跨页表和扫描 OCR 都可能把无关句拼在一起。错误文本再做“结构感知”只会把错误固化；应保存 page、bbox、parser/OCR 版本，并在不可靠时保留页图并行检索。

**Q3：overlap 什么时候不如父对象扩展？**

A：当答案依赖表头、caption、函数签名或父 section 时。overlap 只复制附近 token；命中子对象再按 parent_id 获取父对象，才能恢复真正的结构上下文。

## 关键事实

- 分块没有跨语料的固定最优值；`256–512` 和 `10–20%` 只可作为连续文本 baseline 的实验起点。
- encoder 的最大输入长度必须查模型版本；BGE-M3 论文报告的 8192-token 能力是“所有 encoder 均为 512”的反例。
- 分块前先解析和验收阅读顺序；PDF、DOCX、PPTX、XLSX、HTML 的正确对象边界不同。
- table 保留 header + row group + parent；figure 与 caption 绑定；code 按 AST 边界；page/slide/sheet 是可回跳的容器。
- metadata 要能重建证据与权限：block、parent、page/slide/sheet、bbox、range、version/hash、parser/model 与 ACL 缺一不可。
- 子对象用于召回，父对象用于解释；这比把所有对象机械 overlap 更可靠。

## 来源

- Jianlv Chen et al. [《BGE M3-Embedding》](https://arxiv.org/abs/2402.03216), 2024。报告多语言、多功能 embedding，并称可处理最长 8192 input tokens。
- Tong Chen et al. [《Dense X Retrieval: What Retrieval Granularity Should We Use?》](https://arxiv.org/abs/2312.06648), 2023；EMNLP 2024。比较 passage、sentence 与 proposition 检索粒度。
- [Docling Supported formats](https://docling-project.github.io/docling/usage/supported_formats/)，官方文档，访问于 2026-07-17。统一文档表示与受支持输入格式；实际阅读顺序仍应以样本验收。
