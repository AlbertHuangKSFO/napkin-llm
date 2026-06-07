[[117 多模态推理服务：图像、音频 token 化与缓存]] 处理**当请求里带图像/音频时,服务侧多出来的那些活儿**:非文本模态先经**专用 encoder**(图像走 ViT/CLIP、音频走 Whisper 类)编码,**投影对齐**到语言模型的词嵌入维度,变成一串 **visual / audio token** 拼进序列,再送进 decoder-only 主干。后果是 **prefill 变重**:一张图常展开成数百到上千个 token,prefill 计算量随之上涨,且 encoder 本身要先跑一遍(额外前置开销),TTFT 升高。缓解靠**多模态 KV 缓存**(visual token 的 KV 同样进 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]] 块)、**图像 embedding / encoder 输出缓存**(同图多轮零重编码)、以及把 **视觉 encoder 拆成独立 worker**(EPD 解耦)。模型结构见 [[LLM/041 多模态 LLM 架构(ViT、CLIP、投影对齐)简介|多模态架构]],它和 [[032 前缀缓存：RadixAttention 树结构|前缀缓存]]、[[042 chunked prefill：切块融合|chunked prefill]] 紧密配合。

## 直觉类比
纯文本请求像**只带一段文字**进门;多模态请求像**带着一摞照片和一段录音**进门。门口先有两位**翻译官**(视觉/音频 encoder)把照片、录音翻成模型能读的「外语单词」(visual/audio token),再交给同一个**主审**(LM)和文字一起读。问题:一张照片翻出来可能是**上千个单词**,主审的「初读」(prefill)一下子变长变慢。优化就是:同一张照片反复被问时,**翻译稿存起来**别重译(embedding 缓存);翻译官忙不过来就**单独设个翻译间**(encoder 拆独立 worker),别堵着主审。

## 小数字例子
一条「看图回答」请求:文本 prompt 50 token,1 张 1024×1024 图。
- 视觉 encoder 把图切 patch → 投影后约 **576~1000+ 个 visual token**(随模型/分辨率)。
- prefill 序列长 ≈ 50 + 600 = **650 token**,其中 **>90% 来自图像** → prefill 计算量主要被图撑大,TTFT 明显高于纯文本。
- 同一张图被连续追问 3 轮:若缓存图像 embedding / 前缀 KV,后两轮**跳过 encoder + 跳过图像段 prefill**,只为新问题算 → 大幅降时延。
- 视频/多图场景 token 数翻几倍,prefill 压力线性放大,需 KV 驱逐/压缩控显存。

## 原理:token 化、prefill 加重与缓存
图像/音频经 encoder $E$ 得到特征,再经投影 $P$ 映到 LM 词嵌入空间,与文本嵌入**拼接**成统一序列:

$$
\mathbf{e} = \big[\,\underbrace{P(E_{\text{img}}(x_{\text{img}}))}_{N\ \text{个 visual token}},\ \underbrace{P(E_{\text{aud}}(x_{\text{aud}}))}_{M\ \text{个 audio token}},\ \underbrace{\mathrm{Embed}(x_{\text{txt}})}_{L\ \text{个文本 token}}\,\big]
$$

prefill 计算量 ∝ 序列总长 $N+M+L$,而 $N$(visual)常远大于 $L$,故 prefill 由图像主导、**算力受限**加剧。这些 token 算出的 KV 与文本 KV 一样进分页块管理;可缓存三层:**① encoder 输出 / 图像 embedding**(同图多轮免重编码)、**② 前缀 KV**(同图+同前缀的多问命中)、**③ 多模态 KV 驱逐/压缩**(visual token 冗余高,可裁剪控显存,如 LOOK-M)。服务侧还可把 encoder 拆成独立 worker——**Encode-Prefill-Decode(EPD)解耦**:encoder 与 LM 的资源画像不同,分开扩缩。

![[srv-多模态推理流水线.png]]

## 配置 / 代码
```python
# vLLM:多模态请求(图像 + 文本)
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    limit_mm_per_prompt={"image": 4},   # 每条请求最多几张图(控 prefill 膨胀)
    # 引擎自动:跑视觉 encoder → 投影 → 拼 token → 入 PagedAttention
)

sp = SamplingParams(temperature=0.2, max_tokens=256)

# OpenAI 兼容多模态消息:image_url + text
prompt = {
    "prompt": "USER: <image>\n这张图里有什么?ASSISTANT:",
    "multi_modal_data": {"image": Image.open("cat.jpg")},
}
out = llm.generate(prompt, sp)
print(out[0].outputs[0].text)
# 同图多轮追问:命中前缀缓存 → 跳过图像段 prefill,只算新问题
```

```text
❌ 把每张图都当全新输入、每轮都重跑 encoder + 重算图像段 prefill → TTFT 高、算力浪费;不限图数则 prefill 序列爆长、batch 干扰严重
✅ 缓存 encoder 输出/图像 embedding + 前缀 KV,同图多轮零重编码;limit_mm_per_prompt 限图数;必要时 encoder 拆独立 worker(EPD)分开扩缩
```


![[srv-117三层缓存与EPD.png]]
![[srv-117多模态prefill膨胀.png]]

## 面试高频
- **多模态请求服务侧多了哪些步骤?** 非文本模态先过专用 encoder(ViT/CLIP、Whisper)→ 投影对齐 → 变 token 拼进序列;比纯文本多了 encoder 前向与更长的 prefill。
- **为什么 prefill 变重?** 一张图展开成数百~上千 visual token,prefill 计算 ∝ 序列长,图像 token 远多于文本 → 算力受限加剧、TTFT 升高。
- **怎么缓存优化?** 三层:图像 embedding / encoder 输出缓存(同图多轮免重编码)、前缀 KV 缓存(同图同前缀多问命中)、多模态 KV 驱逐/压缩(visual token 冗余高,可裁)。
- **EPD 解耦是什么?** 把 Encode(encoder)、Prefill、Decode 拆成独立 worker;encoder 与 LM 资源画像不同,分开部署与扩缩,避免 encoder 堵住主干。
- **多模态 KV 和文本 KV 一样吗?** 在 attention 里同质,visual token 的 KV 一样进 PagedAttention 块;区别在量大、冗余高,更需驱逐/压缩。
- **batch 里图文混排有什么坑?** 图像请求 prefill 重、文本请求 decode 轻,混批干扰大,需调度感知(chunked prefill、PD 分离缓解)。

## 关键事实
- 主流结构:decoder-only LM 主干 + 非文本 encoder;LLaVA 用 CLIP 编码图像,Qwen2-Audio 用 Whisper 编码音频,经投影对齐后拼入序列(2024–2025)。
- visual token 数随分辨率/模型变化,单图常达数百~上千;prefill 算力受限随之加重,TTFT 上升。
- vLLM V1(2025)加速多模态推理:encoder 输出缓存、多模态前缀缓存;`limit_mm_per_prompt` 限制每请求模态数量。
- **EPD 解耦**:把视觉 encoder 拆独立 worker(vLLM RFC #20799、NVIDIA Dynamo 多模态、vLLM-Omni 等 2025–2026 方向);encoder 与 prefill/decode 分开扩缩。
- 多模态 KV 压缩:LOOK-M(2024)、面向 VLM 的分层自适应驱逐与金字塔 token 合并(2025–2026)控显存。
- 结构基础见 [[LLM/041 多模态 LLM 架构(ViT、CLIP、投影对齐)简介|多模态架构]];服务侧靠 [[032 前缀缓存：RadixAttention 树结构|前缀缓存]] 与 [[042 chunked prefill：切块融合|chunked prefill]] 缓解 prefill 压力。
