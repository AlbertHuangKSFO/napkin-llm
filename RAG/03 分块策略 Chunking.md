[[03 分块策略 Chunking|分块策略 Chunking]] 是把长文档切成可被独立 embedding、独立检索的小片段(chunk)的过程。它是 [[01 什么是 RAG|什么是 RAG]] 整条管线里**最被低估、却最影响召回质量**的一步:切法决定了「一个 chunk 装的语义是否自洽」,而这直接决定 [[04 Embedding 与向量数据库|Embedding 与向量数据库]] 能不能把它编码准、检索时能不能召回对。一句话——**RAG 的天花板,很多时候是分块定的**。

## 本质:为什么必须切,以及切引出的根本矛盾

为什么不能整篇文档直接喂?三个硬约束:

- **embedding 模型有最大输入长度**。多数 bi-encoder(如 BGE、E5)上限 512 token,长文档塞不进;就算塞得进(长上下文 embedding 模型),把整篇压成**一个**向量也会把多个主题糊成一团,语义被稀释。
- **生成端上下文窗口有限 + 成本**。检索回来的内容要拼进 prompt 喂给 LLM,块越大越占 token、越贵、越容易 lost-in-the-middle([[02 Naive RAG 与失败模式|Naive RAG 与失败模式]] 的典型坑)。
- **检索粒度 = 召回粒度**。你 embed 的单位就是你能召回的最小单位。

于是引出分块的**根本矛盾**(贯穿全篇):

> **块太大** → 一个向量糊了多个主题,embedding 被稀释,query 来了相似度不突出 → **召回不准、噪声多**。
> **块太小** → embedding 很精准,但一个 chunk 装不下完整论述,**丢上下文** → 召回到了也答不好。

像把一本书剪成索引卡片:卡片剪太大,一张上又是「相对论」又是「布朗运动」,你按主题翻找时哪张都沾边、哪张都不精准;卡片剪太小,只剩半句「他于此处出生」,翻到了也不知道「他」是谁、「此处」在哪。理想是「一卡一事、事事自洽」——可惜文章本身不是按这粒度写的。

这个矛盾没有「切多大」的银弹解,只有针对内容选策略 + 用 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 的 small-to-big 把「embed 粒度」和「喂 LLM 粒度」**解耦**来缓解。

![[分块策略 Chunking.png]]

## 机制:四类切法,从粗暴到智能

### 1. 固定大小(fixed-size)
按固定 token / 字符数硬切,设一个 overlap。最简单、最快、最可预测,但**死板**——会从句子甚至单词中间切断,把「不…」和「正确」切到两个 chunk,语义破碎。仅适合格式极规整、或要求极低延迟的场景。

### 2. 递归字符(recursive character)
**工程默认首选**。LangChain 的 `RecursiveCharacterTextSplitter` 是代表:给一个**分隔符优先级列表** `["\n\n", "\n", "。", " ", ""]`,先尽量按最大粒度(段落 `\n\n`)切;某段仍超过 chunk_size,就降一级用 `\n`(行)再切;还超就用句号、空格,最后才落到字符。这样**尽量在自然边界断开**,比固定大小友好得多,又不需要模型调用。

### 3. 语义分块(semantic chunking)
不靠分隔符,靠**含义**。把文档拆成句子,对每句算 embedding,扫描相邻句的相似度:相似度高就归一块,**相似度骤降的地方=话题切换=断点**。LlamaIndex 的 `SemanticSplitterNodeParser` 即此思路。优点是块边界贴合语义、块内主题自洽;代价是要为每句跑一次 embedding,**慢、贵**,且阈值需要调。

### 4. 文档结构感知(structure-aware)
顺着文档**天然结构**切,而非把它当纯文本流:
- **Markdown**:按 `#`/`##` 标题层级切,每个 section 一块,标题可作为 chunk 的上下文前缀。
- **代码**:按函数 / 类边界切(LangChain 有按语言的 splitter),别把一个函数劈两半。
- **表格 / HTML**:表格不拆行、保留表头;HTML 按 DOM 节点。
- **PDF**:先做版面解析(标题、正文、图表分离)再切。
这类切法质量最高,但**依赖好的解析器**,脏文档(扫描件、错乱 PDF)上会翻车。

### chunk_size 与 overlap 的权衡(再具一点)
- **chunk_size**:经验区间 **256–512 token**;问答类偏小(精准召回),需要长论述上下文的偏大。
- **overlap**:相邻块重叠 **10–20%**。作用是给被切断的信息一个**缓冲**——某个关键句正好落在边界,overlap 让它在两个块里都出现,避免任何一块都召回不全。代价是冗余、存储与索引变大。

**overlap 救边界句手算**。设 `chunk_size=1000` 字、`overlap=60` 字(6%),关键句「该药与酒精同服可致呼吸抑制」(共 15 字)正好横跨边界:前 8 字落在第 1 块末尾(第 993–1000 字),后 7 字落在第 2 块开头。若 overlap=0,两块谁都只拿到半句,query「这药能配酒吗」对哪块都召不全。开了 60 字 overlap,第 2 块从第 $1000-60=940$ 字起算,把这句完整的 15 字(第 993–1007 字)全包进来——它在第 2 块里完整出现,召回得回、答得对。代价是这 60 字在两块各存一份,索引体积约 $+6\%$。

## 命题分块(proposition chunking):把粒度推到极限

是否能比「句」更细?**命题(proposition)**——一个原子事实,自包含、不依赖指代的最小陈述单位。例:把「他 1955 年出生于此」改写成「爱因斯坦 1955 年去世」「爱因斯坦出生于乌尔姆」两条独立命题。

学界证据来自 **Chen et al. 2023《Dense X Retrieval: What Retrieval Granularity Should We Use?》(arXiv:2312.06648,后收录 EMNLP 2024)**:系统比较 passage / sentence / proposition 三种检索粒度,发现**以命题为检索单元在多数 QA 任务上显著优于段落和句子**——因为每个命题信息密度高、几乎无无关内容,召回的都是「干货」。代价是要先用 LLM 把文档**改写**成命题列表(贵),且过度原子化可能丢失叙事连贯。这是「块太小但靠去指代/自包含来补上下文」的极端探索,实践中常和 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 配合(命题用来 embed,返回时映射回原段)。

## 可跑最小代码

```python
# ① 递归字符:工程默认。按 分隔符优先级 尽量在自然边界切
def recursive_split(text, chunk_size=400, overlap=60,
                    seps=["\n\n", "\n", "。", " ", ""]):
    if len(text) <= chunk_size or not seps:
        return [text]
    sep, *rest = seps
    parts = text.split(sep) if sep else list(text)
    chunks, buf = [], ""
    for p in parts:
        piece = p + sep
        if len(buf) + len(piece) > chunk_size:
            if buf: chunks.append(buf)
            # 单段仍超长 → 降一级分隔符继续递归切
            buf = piece if len(piece) <= chunk_size else \
                  "".join(recursive_split(piece, chunk_size, overlap, rest)) and ""
            if len(piece) > chunk_size:
                chunks += recursive_split(piece, chunk_size, overlap, rest)
                buf = ""
        else:
            buf += piece
    if buf: chunks.append(buf)
    # 加 overlap:每块末尾 overlap 字符前缀进下一块(此处略,生产用库)
    return chunks

# ② 语义分块:相邻句 embedding 相似度骤降处断
import numpy as np
def semantic_split(sentences, embed, drop=0.2):
    embs = [embed(s) for s in sentences]
    sims = [float(np.dot(embs[i], embs[i+1])) for i in range(len(embs)-1)]
    chunks, buf = [], [sentences[0]]
    thresh = np.percentile(sims, 100*drop)  # 最低 20% 相似度处视为话题切换
    for i, s in enumerate(sims):
        if s < thresh:                       # 断点
            chunks.append("".join(buf)); buf = []
        buf.append(sentences[i+1])
    if buf: chunks.append("".join(buf))
    return chunks
# 真用别手搓:LangChain RecursiveCharacterTextSplitter / LlamaIndex SemanticSplitterNodeParser
```

## 对比表

| 策略 | 切割依据 | 质量 | 成本 | 适合 |
|---|---|---|---|---|
| 固定大小 | token/字符数 | 低(切断语义) | 极低 | 规整文本、低延迟 |
| 递归字符 | 分隔符优先级 | 中高 | 低 | **通用默认** |
| 语义分块 | 句间 embedding 相似度 | 高 | 中(逐句 embed) | 主题混杂的长文 |
| 结构感知 | 文档天然结构 | 高 | 中(依赖解析) | Markdown/代码/表格 |
| 命题分块 | LLM 改写成原子事实 | 最高(密度) | 高(逐文档 LLM) | 高精度事实 QA |

## 何时用 / 坑

**选型一句话**:先上**递归字符**当 baseline;文档结构清晰(Markdown/代码)就上**结构感知**;主题混杂的长文用**语义**;要冲召回极限再考虑**命题**。别一上来就追求花哨切法。

**坑**:
- **切断关键信息**:固定大小最常见。永远配 overlap,或换递归。
- **chunk 丢上下文**:小块召回到了却答不全。**不要靠把块切大来补**——用 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 的 small-to-big,或 [[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|进阶索引：Contextual Retrieval、RAPTOR、Late Chunking]] 给块补上下文。
- **语义分块阈值难调**:相似度阈值/百分位敏感,不同语料要重调,且慢。先确认递归不够用再上。
- **脏文档毁结构感知**:扫描 PDF、错乱 HTML 会让结构解析输出垃圾。先治理解析([[17 检索数据治理|检索数据治理]]),再谈结构切。
- **块大小与 embedding 模型不匹配**:切出 800 token 的块喂给 512 上限的模型,要么截断要么报错。切之前先看模型上限。
- **元数据丢失**:切块时把来源、标题、页码一并存进 chunk metadata,否则后面做 [[11 生成层：引用归因与忠实度|生成层：引用归因与忠实度]] 和 [[16 检索安全与访问控制|检索安全与访问控制]] 全抓瞎。

## 工业界实践

**生产分块的事实标准**:递归字符分块当 baseline,结构化文档走 structure-aware,然后**几乎一定配 small-to-big**(检索小块、喂 LLM 大块,见 [[06 检索粒度：父文档与句子窗口检索|06]])。纯靠调 chunk_size 撞最优是新手做法。

**主流工具**:
- **解析(切之前先把文档变干净)**:`Unstructured.io`(多格式、版面识别)、`LlamaParse`(LlamaIndex,擅长复杂 PDF/表格)、`Azure Document Intelligence`、`Docling`(IBM 开源)。脏 PDF/扫描件的版面解析质量,直接决定后续切块上限。
- **切块器**:LangChain `RecursiveCharacterTextSplitter`(默认)/ `MarkdownHeaderTextSplitter` / 按语言的 `from_language` code splitter;LlamaIndex `SentenceSplitter`(默认)/ `SemanticSplitterNodeParser`(语义)/ `HierarchicalNodeParser`(多粒度,配 small-to-big)。
- **元数据是命脉**:切块时务必把 `source / title / page / section / acl / timestamp` 写进 chunk metadata,后面 [[11 生成层：引用归因与忠实度|引用归因]]、[[16 检索安全与访问控制|访问控制]]、[[17 检索数据治理|治理]] 全靠它。

**典型生产配置**(经验值,起点而非真理):

```yaml
# 通用知识库的稳妥起手配置
splitter: recursive_character
chunk_size: 512        # token,问答密集型可降到 256
chunk_overlap: 64      # ~12%,缓冲被切断的边界句
separators: ["\n\n", "\n", "。", "!", "?", " ", ""]
metadata: [source, title, page, section]   # 切块时一并落库
# 结构化文档(Markdown/代码)→ 换 structure-aware
# 长文档检索答不全 → 不调大 chunk,改上 small-to-big(父文档/句子窗口)
```

**结构感知 + small-to-big 的真实组合**:

```python
# Markdown 按标题切(structure-aware),标题作为上下文前缀注入每块
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
md_chunks = MarkdownHeaderTextSplitter(headers).split_text(doc)  # 每块带标题元数据
# 再对过长 section 二次递归切;标题前缀拼进正文,让小块也自带上下文
final = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64).split_documents(md_chunks)
# 生产里通常还会把 final(小块)embed,检索命中后映射回原 section(大块)喂 LLM → small-to-big
```

**规模化权衡**:chunk 越小 → 向量数越多 → 索引内存/构建成本越高、召回更精但片段更碎;chunk 越大 → 向量少、省存储,但 embedding 稀释、检索准度降。语义分块和命题分块在大语料上**贵到劝退**(逐句/逐文档跑模型),通常只用于高价值小语料或离线一次性处理。

**踩坑速记**:
- **chunk_size 超过 embedding 模型上限**(如 512 token 模型喂 800 token 块)→ 静默截断或报错。切之前先查模型 `max_seq_length`。
- **脏文档毁 structure-aware**:扫描 PDF / 错乱 HTML 让解析输出垃圾结构。先治理解析,再谈结构切。
- **语义分块阈值跨语料不通用**:相似度百分位阈值换语料要重调,且慢。先确认递归不够用再上。
- **overlap 设过大**:冗余暴涨、同一句反复召回挤占 top-k,反而稀释多样性。10–20% 够了。

## 面试高频

**Q1:为什么 RAG 必须分块,不能整篇文档 embedding?**
标准答:三个硬约束——① embedding 模型有最大输入长度(多数 bi-encoder 512 token);② 整篇压成一个向量会把多主题糊成一团、语义稀释;③ 检索粒度=召回粒度,你 embed 的单位就是能召回的最小单位,且块越大越占生成端 token、越易 lost-in-the-middle。

**Q2:分块的根本矛盾是什么?怎么破?**
标准答:**块大→一个向量糊多主题、embed 稀释、召回不准;块小→embedding 精准但丢上下文、答不全**。无「切多大」的银弹。破法是**解耦**:用 [[06 检索粒度：父文档与句子窗口检索|small-to-big]]——小块 embed 保证召回精度,命中后返回所在父块/窗口喂 LLM 保证上下文。

**Q3:常见分块策略有哪些,怎么选?**
标准答:固定大小(最快最差,切断语义)< 递归字符(**工程默认**,按 `\n\n→\n→句→词` 优先级在自然边界切)< 语义分块(句间 embedding 相似度骤降处断,高质但慢贵)/ 结构感知(顺文档天然结构,质量最高但依赖解析器)。选型:递归当 baseline → 结构清晰上 structure-aware → 主题混杂长文上语义 → 冲召回极限再上命题。

**Q4:chunk_size 和 overlap 怎么定?**
标准答:chunk_size 经验区间 **256–512 token**(问答偏小、长论述偏大);overlap **10–20%**,作用是给被切断的边界信息一个缓冲——关键句正好落边界时,overlap 让它在相邻两块都出现,避免谁都召不全。代价是冗余、索引变大。

**Q5(进阶):什么是命题分块?它解决什么,代价是什么?**
标准答:命题(proposition)= 自包含、去指代的最小原子事实(如把「他 1955 年出生于此」改写成独立的「爱因斯坦出生于乌尔姆」)。出处 **Chen et al. 2023 Dense X Retrieval(arXiv:2312.06648,EMNLP 2024)**:以命题为检索单元在多数 QA 上优于段落/句子,因信息密度高、几乎无噪声。代价:需逐文档用 LLM 改写(贵),过度原子化可能丢叙事连贯,常配 small-to-big(命题 embed、返回映射回原段)。

## 知识拓展

- **分块是抬天花板的一环,不是孤立步骤**:它的上限要靠 [[06 检索粒度：父文档与句子窗口检索|small-to-big 解耦]] + [[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|进阶索引补上下文]] 一起抬。
- **Contextual Retrieval(Anthropic 2024)**:正面回应「小块丢上下文」——给每个 chunk 用 LLM 生成一句文档级上下文摘要再 embedding,combined Contextual Embeddings + Contextual BM25 把 top-20 检索失败率降 49%(5.7%→2.9%),叠重排降 67%。详见 [[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|05]] 与 [[#来源]]。
- **Late Chunking(Jina AI 2024)**:换个思路——先用长上下文 embedding 模型把整篇编码,再在 token 级表示上切块取均值,让每个块向量天然带全文上下文,避免「先切后 embed」丢语境。见 [[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|05]]。
- **RAPTOR(Sarthi et al. 2024,ICLR)**:对块做递归聚类 + 摘要,建一棵多层摘要树,检索时跨层取证据,缓解「小块答不了需要全局综合的问题」。
- **边界与反模式**:① 不要靠把块切大来补上下文(治标,稀释召回)——用 small-to-big。② 不要无脑追花哨切法,递归字符能覆盖大多数场景。③ 切块丢元数据 = 后续归因/治理/访问控制全断,这是最隐蔽的生产坑。④ 命题/语义分块在大语料上成本爆炸,别在百万级文档上无脑铺开。
- **底层联系**:embedding 把块编码成向量、靠余弦相似度检索,几何直觉见 [[深度学习基础/03 点积、范数与相似度|点积、范数与相似度]];切多大与 embedding 模型能力强相关,选型见 [[04 Embedding 与向量数据库|04]]。

## 关键事实

- 分块的根本矛盾:**块大→embed 稀释、召回不准**;**块小→丢上下文、答不好**。无银弹。
- 四类切法:固定大小 < 递归字符(默认) < 语义 / 结构感知(高质高成本)。
- 递归字符(`RecursiveCharacterTextSplitter`)按 `\n\n→\n→句→词` 优先级在自然边界切,是工程默认首选。
- 典型参数:chunk_size **256–512 token**,overlap **10–20%**;overlap 缓冲被切断的边界信息。
- 命题分块(**Chen et al. 2023, arXiv:2312.06648, Dense X Retrieval, EMNLP 2024**):把文档拆成自包含原子事实,检索粒度推到极限,多数 QA 上优于段落/句子,代价是逐文档 LLM 改写。
- 切块同时**存元数据**(来源/标题/页码),否则归因、访问控制、治理全断。
- 分块不是孤立步骤:它的上限要靠 [[06 检索粒度：父文档与句子窗口检索|检索粒度：父文档与句子窗口检索]] 解耦 + [[05 进阶索引：Contextual Retrieval、RAPTOR、Late Chunking|进阶索引：Contextual Retrieval、RAPTOR、Late Chunking]] 补上下文 一起抬。

## 来源

- Chen et al. 2023.《Dense X Retrieval: What Retrieval Granularity Should We Use?》arXiv:2312.06648(收录 EMNLP 2024)。命题(proposition)作为检索单元。
- LangChain `RecursiveCharacterTextSplitter` 文档;LlamaIndex `SemanticSplitterNodeParser` 文档。
