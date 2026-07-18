[[21 混合文档摄入：解析、OCR、版面与阅读顺序|混合文档摄入]]的本质是：把异构原件变成**可检索、可回放、可判错**的结构化证据，而不是“抽出一串文字”就入库。每份 PDF、扫描件、DOCX、PPTX、XLSX 或 HTML 都要先选对提取路径，再保存版面、阅读顺序与版本血缘；质量未达标的页面应进入有限重试或隔离队列，不能悄悄污染 [[01 什么是 RAG|RAG]] 的索引。

## 直觉：图书馆收书不是只把字抄下来

把它想成图书馆新书编目。印刷清楚的书可以直接抄目录（born-digital 文档的文本层）；一摞照片或扫描件则要先由馆员识字（OCR）。但即使每个字都认对了，若把报纸的右栏接到左栏、把表格列打散、把图注塞进正文，读者仍会读错。因此编目卡还要记住“这段来自第几页、先读哪里、这张图和哪段文字相连”，并留下本次用的识别器和原件指纹，之后才找得到责任链。

**先归一化，再分路。**接入层保存原始字节并计算内容哈希；用真实格式签名和容器内容判断类型，而不是只信扩展名。PDF 先检查是否有可靠文本层：有则走原生提取，并保留页码、字符框、字体/块等可用结构；没有或文本层异常则渲染页图后 OCR。DOCX/PPTX/XLSX 是 Office Open XML 容器，应分别保留段落/幻灯片/工作表、单元格坐标和公式显示值；HTML 则先去脚本与样式噪声，再按 DOM 语义提取标题、段落、表格、链接与图片。

**OCR 不是“失败版文本提取”。**它把页图变为文字框，必须记录语言、版面模式和模型版本；born-digital 提取也可能失败，例如 PDF 的字符映射损坏、文字被画成路径，或正文只是扫描图。两路最终都输出同一份中间表示：`page → block → line/span/table/figure`，每个节点附页坐标、来源方法和置信/状态。

**版面是语义的一部分。**阅读顺序先按列、块和锚点重建，再在块内按行排序；表格需要行列网格、跨行/跨列和单元格文本，而不是把所有单元格平铺；图片、图注和邻近解释应显式关联。入库前应抽检：页数与源文件是否一致、文本覆盖是否异常、相邻块顺序是否正确、表格单元格是否完整、图和图注是否可定位。通过后才交给 [[03 分块策略 Chunking|分块策略]]；失败时按原因进入“换解析器/提高渲染质量/换 OCR 配置”的**有界**回退链，超过上限则隔离并生成可人工处理的工单。

每次运行至少写入一条 ingestion manifest：`source_uri`、字节 `sha256`、格式和路由、parser/OCR/版面模型版本、配置哈希、页数、质量指标、尝试次数、产物 ID 与最终状态。这样同一原件可去重、同一结果可重放，也能解释“为什么这批 chunk 与昨天不同”。这与 [[17 检索数据治理|检索数据治理]] 的派生资产血缘相连：隔离原件及其产物均不可进入在线召回。

## 小数字手算：一张表格如何挡住坏文档

一份 4 页扫描发票完成 OCR 与版面还原后，抽检到四个归一化指标：文本覆盖 $C=0.98$、阅读顺序正确率 $O=0.94$、表格单元格召回 $T=1/2=0.50$、图和图注关联率 $F=3/3=1.00$。设质量分数对文本、顺序、表格、图的权重分别为 $(0.35,0.25,0.20,0.20)$：

$$
Q=0.35C+0.25O+0.20T+0.20F
$$

逐项代入：

$$
\begin{aligned}
Q&=0.35\times0.98+0.25\times0.94+0.20\times0.50+0.20\times1.00\\
 &=0.343+0.235+0.100+0.200\\
 &=0.878
\end{aligned}
$$

即使 $Q$ 看起来不低，表格门 $T\ge0.90$ 仍不通过：金额列少了一半，不能靠其他指标“平均”掉。第一次回退改用表格网格解析后 $T=1.00$，得到 $Q=0.978$，且所有单项门通过，才发布。若两次尝试仍未通过，就把原件和失败原因放入 quarantine，而不是以半张表进入向量库。

## 公式推导：为什么质量、重试和血缘必须一起存

令原始字节为 $b$，其不可变内容标识为：

$$
h=\operatorname{SHA256}(b)
$$

用路由器 $r$（例如 `pdf-native`、`pdf-ocr`、`docx`）和版本化配置 $v$ 处理，得到结构化产物 $y$：

$$
y=E_{r,v}(b),\qquad m=\operatorname{manifest}(h,r,v,y)
$$

因此比较两次产物不能只问“文件名是否相同”，而要比较输入和处理契约：

$$
(h,r,v)_{\text{old}}=(h,r,v)_{\text{new}}
\Rightarrow \text{应可重放同一处理条件；}
\quad
v_{\text{new}}\ne v_{\text{old}}
\Rightarrow \text{必须视为新派生版本。}
$$

把四类可观测指标组成向量 $z=(C,O,T,F)$，质量函数和准入谓词分别为：

$$
Q(z)=\sum_i w_i z_i,
\qquad
\operatorname{admit}(z)=
\bigl[Q(z)\ge0.90\bigr]
\land[C\ge0.90]\land[O\ge0.90]\land[T\ge0.90]\land[F\ge0.90]
$$

“总分”与“逐项下限”并存，避免高文本覆盖掩盖坏表格。设最大尝试数为 $A=2$，第 $a$ 次的状态递推为：

$$
s_{a+1}=
\begin{cases}
\text{published}, & \operatorname{admit}(z_a)\\
\text{retry}, & \neg\operatorname{admit}(z_a)\land a<A\\
\text{quarantined}, & \neg\operatorname{admit}(z_a)\land a=A
\end{cases}
$$

这使重试既可解释又有上界：错误的 PDF 不会无限烧算力，也不会因“先入库再修”而让下游检索到未验证证据。

![[混合文档摄入-解析与质量门.png]]

## 可运行代码：manifest、质量门与有界隔离

下面仅使用 Python 标准库，模拟一个有文本层的 PDF 和一个始终失败的扫描件。`❌` 的做法只留文本，无法追溯来源或拦截坏表；`✅` 的做法从字节哈希、解析器/OCR 版本到每次质量指标都写入 manifest，并强制两次内发布或隔离。把代码保存为 `ingest_gate.py` 后运行 `python ingest_gate.py` 即可。

```python
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory


# ❌ 朴素：只存“抽到的文本”。重跑、换 OCR、坏表格都不可追溯。
naive_index = {"invoice.pdf": "总额 1280 供应商 示例公司"}
assert "invoice.pdf" in naive_index


# ✅ 高效：最小 manifest + 分项质量门 + 有界重试 + quarantine。
WEIGHTS = {"coverage": 0.35, "reading_order": 0.25, "tables": 0.20, "figures": 0.20}
FLOOR = 0.90
MAX_ATTEMPTS = 2


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def route(has_text_layer: bool) -> str:
    return "pdf-native" if has_text_layer else "pdf-ocr"


def quality(metrics: dict[str, float]) -> tuple[float, bool]:
    score = sum(WEIGHTS[name] * metrics[name] for name in WEIGHTS)
    passed = score >= FLOOR and all(metrics[name] >= FLOOR for name in WEIGHTS)
    return round(score, 3), passed


def ingest(path: Path, *, has_text_layer: bool,
           attempts: list[dict[str, float]]) -> dict[str, object]:
    manifest: dict[str, object] = {
        "source_uri": path.name,
        "sha256": digest(path),
        "format": "pdf",
        "route": route(has_text_layer),
        "parser": {"name": "example-parser", "version": "2.1.0"},
        "ocr": {"name": "example-ocr", "version": "5.0.0" if not has_text_layer else None},
        "layout": {"name": "reading-order-grid", "version": "1.4.0"},
        "config_sha256": sha256(b"dpi=300;lang=chi_sim").hexdigest(),
        "attempts": [],
    }
    for number, metrics in enumerate(attempts[:MAX_ATTEMPTS], start=1):
        score, passed = quality(metrics)
        action = "publish" if passed else (
            "quarantine" if number == MAX_ATTEMPTS else "fallback"
        )
        manifest["attempts"].append({
            "number": number, "metrics": metrics, "quality": score,
            "action": action,
        })
        if passed:
            manifest["status"] = "published"
            manifest["artifact_id"] = f"parsed:{manifest['sha256'][:12]}"
            return manifest
    manifest["status"] = "quarantined"
    manifest["reason"] = "quality gates failed after bounded retries"
    return manifest


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        born_digital = root / "invoice.pdf"
        scan = root / "blurry-scan.pdf"
        born_digital.write_bytes(b"%PDF-1.7 text layer example")
        scan.write_bytes(b"%PDF-1.7 scan image example")

        published = ingest(
            born_digital,
            has_text_layer=True,
            attempts=[
                {"coverage": 0.98, "reading_order": 0.94, "tables": 0.50, "figures": 1.00},
                {"coverage": 0.98, "reading_order": 0.94, "tables": 1.00, "figures": 1.00},
            ],
        )
        quarantined = ingest(
            scan,
            has_text_layer=False,
            attempts=[
                {"coverage": 0.76, "reading_order": 0.62, "tables": 0.00, "figures": 0.40},
                {"coverage": 0.81, "reading_order": 0.70, "tables": 0.20, "figures": 0.55},
            ],
        )
        assert published["status"] == "published"
        assert published["attempts"][0]["quality"] == 0.878
        assert published["attempts"][0]["action"] == "fallback"
        assert published["attempts"][1]["quality"] == 0.978
        assert published["attempts"][1]["action"] == "publish"
        assert quarantined["status"] == "quarantined"
        assert quarantined["attempts"][-1]["action"] == "quarantine"
        print(json.dumps([published, quarantined], ensure_ascii=False, indent=2))
```

生产实现还应把 manifest 与解析产物一起版本化：原件哈希相同但 parser、OCR、语言包、渲染 DPI 或阅读顺序模型变化时，产生新 artifact version；索引切换只能指向质量门已通过的版本。重试应按失败原因选择后备策略，不能盲目重复同一条失败命令。

## 面试高频

> 面试地图：[[RAG 面试题库]]

**Q1：为什么不能让所有 PDF 都直接 OCR？** 许多 PDF 有可提取文本层，原生提取通常还能保留更精确的字符坐标和结构；扫描件或异常文本层才需要 OCR。统一用 OCR 会丢失可用结构，也掩盖“原生提取失败还是识别失败”的诊断边界。

**Q2：OCR 置信度高，是否就能入库？** 不能。单字置信度不检验两栏阅读顺序、表格网格、合并单元格或图注关联。应同时测文本覆盖、顺序、表格、图像/图注等任务级质量门，并对高风险文档抽样人工核验。

**Q3：为什么 manifest 要记哈希和版本，文件名不够吗？** 文件名可被覆盖或复用；内容哈希标识具体字节，parser/OCR/配置版本标识处理条件。两者共同决定一个派生产物能否重放、去重、比较或撤回。

**Q4：为什么要 quarantine，而不是“先入库、以后再修”？** 检索系统会把不完整的表格或乱序段落当证据，错误会在 chunk、embedding、回答和引用中放大。隔离把不确定性保留在摄入边界，并给人工修复明确的失败证据。

## 关键事实

- Smith（2007），*An Overview of the Tesseract OCR Engine*，ICDAR 2007，DOI: 10.1109/ICDAR.2007.4376991。OCR 引擎是将图像中的印刷文字转为可处理文本的一类组件；实际运行须把引擎、语言数据与配置版本记入 manifest。
- Zhong、Tang、Jimeno Yepes（2019），*PubLayNet: largest dataset ever for document layout analysis*，arXiv:1908.07836。论文把非结构化数字文档的版面识别定位为转成下游可用结构化表示的重要步骤，说明“文本抽取”不能替代版面解析。
- ECMA-376（*Office Open XML File Formats*，第 5 版；其中 Part 2《Open Packaging Conventions》为 2021 年 12 月版）定义 Office Open XML 的词汇、文档表示与封装要求；在摄入 DOCX、PPTX、XLSX 时应保留段落、工作表/单元格、幻灯片等结构，而不是统一降成纯文本。
- Tesseract 官方文档（5.x）将其定义为 OCR 文本识别引擎，并提供输入、语言数据和使用方式文档；不同引擎/语言数据/布局模式是可复现实验条件，不应只记录“做过 OCR”。

**出处。**

- [Smith（2007），An Overview of the Tesseract OCR Engine](https://doi.org/10.1109/ICDAR.2007.4376991)；[Tesseract 官方用户手册](https://tesseract-ocr.github.io/tessdoc/)（核验日：2026-07-17）。
- [Zhong et al.（2019），PubLayNet](https://arxiv.org/abs/1908.07836)：版面分析与结构化文档解析的一手论文。
- [ECMA-376：Office Open XML File Formats，第 5 版（Part 2，2021 年 12 月）](https://ecma-international.org/publications-and-standards/standards/ecma-376/)：DOCX、PPTX、XLSX 的格式语义来源。
