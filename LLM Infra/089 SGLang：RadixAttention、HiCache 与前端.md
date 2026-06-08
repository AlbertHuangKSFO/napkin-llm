[[089 SGLang：RadixAttention、HiCache 与前端]]:[[087 引擎全景：六大 runtime 选型|六引擎]]里的「共享前缀之王」——LMSYS 主导的 SGLang 用 **RadixAttention(前缀树自动复用 KV)+ HiCache(GPU/host/分布式三层 KV)+ 前端 DSL(结构化 LLM 程序)** 在 chat/[[RAG/01 什么是 RAG|RAG]]/[[Agent/01 什么是 AI Agent|Agent]] 这类大量共享上下文的负载上把吞吐拉满。是 [[032 前缀缓存：RadixAttention 树结构|RadixAttention]] 概念落到生产引擎的代表。

## ① 类比:把「公共开头」存进一棵家谱树

普通前缀缓存像「整段精确匹配」:两个请求开头必须**一字不差**才能共享。RadixAttention 把所有请求的前缀组织成一棵**基数树(radix tree)**——像家谱:同一个 system prompt 是「祖先」,挂同一篇文档的若干问题是「子孙」,**任意分叉点之前的 KV 都只算一次**。HiCache 再加一层「仓储分级」:最热的 KV 放 GPU(L1),温的下沉到 CPU 内存(L2),冷的进分布式存储/SSD(L3)——像热门书放手边、旧书进书库,需要时再调上来。

## ② 小数字例子:共享前缀越多,赢越多

- **RAG / 同文档多问**:几千个问题共享同一 system prompt + 同一批检索文档。普通缓存可能只命中 system 段;RadixAttention 连「文档段」也复用,共享前缀的 prefill **几乎不重算**,共享比例高时吞吐相对 vLLM 显著领先。
- **HiCache(2025-09 公布)**:把 KV 扩到 host 内存和分布式存储后,LMSYS 报告**最高 6× 吞吐、最高 80% TTFT 下降**——因为大量历史/共享 KV 不必重算、也不必挤在 GPU 显存里。
- **多轮 chat**:第 N 轮把前 N-1 轮的 KV 沿树命中,只为新一轮算 KV。

(数字随负载/共享率/版本变动,共享前缀比例低时优势收窄。)

## ③ 原理:三个支柱

**1. RadixAttention —— 前缀树自动复用。** 在 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]] 之上,把所有请求前缀组织成 radix 树,节点存对应 token 前缀的 KV。新请求沿树匹配**最长已缓存前缀**,命中部分**直接复用 KV、跳过 prefill 重算**,只为新增 token 算 KV;树节点用 LRU 驱逐管显存。比「整段精确前缀」粒度更细——**任意分叉处都能共享**(见 [[032 前缀缓存：RadixAttention 树结构|RadixAttention 树结构]]、[[033 自动前缀缓存的命中与失效|命中与失效]])。

![[eng-089基数树复用.png]]

**2. HiCache —— 分层 KV(L1/L2/L3)。** 扩展 RadixAttention:GPU 显存=L1、host CPU 内存=L2、分布式存储/SSD=L3(可接 Mooncake store)。热 KV 留 GPU,温/冷 KV 下沉,需要时调回。本质是把 [[036 KV 分层 offload：GPU、CPU、SSD(LMCache)|KV 分层 offload]] 与 [[037 Mooncake：KVCache 中心的存储池|Mooncake]] 思想做进引擎。2025 还引入 UnifiedRadixTree、DeepSeek 适配、SSD offload 等。

![[eng-089HiCache三层.png]]

**3. 前端 DSL —— 结构化 LLM 程序。** 提供 `gen`/`fork`/控制流/约束解码(JSON、正则)的前端语言,把「多步、有分支、要结构化输出」的 LLM 程序写得像普通代码;编译后**天然产出大量共享前缀的请求**,与 RadixAttention 形成正反馈。再叠加 EAGLE/EAGLE-3 [[073 投机解码系统：draft-verify 全流程|投机解码]]、wide-EP 等并行,覆盖 LLM 与多模态。

![[eng-SGLang架构.png]]

底层仍是连续批 + PagedAttention 显存,与 [[088 vLLM V1 架构剖析|vLLM]] 同源概念;差异在 ③ 显存/前缀复用做得更激进(树 + 分层)和前端 DSL。

## ④ 代码/配置:启动与前端程序

```bash
# 启动 OpenAI 兼容 server(默认开 RadixAttention 前缀缓存)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --tp 2

# 开启 EAGLE 投机解码 / HiCache 分层(示意,具体 flag 随版本)
python -m sglang.launch_server --model-path <model> \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --enable-hierarchical-cache
```

```python
# 前端 DSL:fork 出多分支共享同一 system + 文档前缀
import sglang as sgl
@sgl.function
def qa(s, doc, questions):
    s += sgl.system("你是文档助手。")
    s += doc                       # 共享前缀:RadixAttention 只算一次
    forks = s.fork(len(questions))
    for f, q in zip(forks, questions):
        f += sgl.user(q) + sgl.assistant(sgl.gen("a", max_tokens=128))
```

❌ 反模式:在 RAG / 多轮负载上用「整段精确前缀缓存」的引擎,文档段稍有差异就 miss,反复重算长 prefill。
✅ 正解:共享前缀重 → SGLang 的 radix 树在分叉粒度共享;历史 KV 大 → 开 HiCache 把温冷 KV 下沉 host/SSD,腾出 GPU 显存。

## 面试高频

- **「RadixAttention 比普通前缀缓存强在哪?」** 粒度:radix 树在**任意分叉点**共享,不要求整段精确匹配;天然适配「同 system/同文档、不同问题」的分叉结构。
- **「HiCache 解决什么?」** GPU 显存装不下所有热 KV;分层把温/冷 KV 放 host(L2)/分布式 SSD(L3),命中即调回,避免重算又不挤显存 → 报告最高 6× 吞吐、80% TTFT 下降。
- **「前端 DSL 有什么用,不是多此一举?」** 它把多步/分支程序结构化,编译出的请求**自带共享前缀**,与 RadixAttention 正反馈;还能约束解码出 JSON/正则结构。
- **「SGLang vs vLLM 选型?」** 共享前缀比例高(chat/RAG/Agent)选 SGLang;通用在线、常换模型选 vLLM。底层概念同源,差异在前缀复用激进度与前端。
- **「SGLang 怎么提速 decode?」** EAGLE/EAGLE-3 投机解码 + 各类并行(TP/PP/EP/wide-EP)。

## 关键事实

- SGLang 由 **LMSYS** 主导,仓库 `sgl-project/sglang`;定位低延迟 + 高吞吐 LLM/VLM serving。
- **RadixAttention**:radix 树自动检测复用共享前缀 KV,LRU 驱逐。
- **HiCache** 2025-09 LMSYS 博客公布:L1 GPU / L2 host / L3 分布式,接 Mooncake;最高 **6× 吞吐**、**80% TTFT** 下降。
- 2025 更新:UnifiedRadixTree、DeepSeek 适配、SSD offload(Mooncake store)、Speculative Decoding V2 / **EAGLE-3**。
- 提供 **前端 DSL**(结构化 LLM 程序)+ OpenAI 兼容 `/v1/chat/completions`(端口默认 30000)。
- **最新进展(2025-2026)**:HiCache 除 Mooncake/3FS 外已支持 **NIXL** 与本地文件后端;2025-10 起经 **SGLang-Jax** 原生跑 **TPU**;并对新开源模型做 day-0 适配(MiniMax M2、Mistral Large 3、LLaDA 2.0 扩散 LLM 等,2025-12)。(来源:LMSYS Blog "SGLang HiCache" 2025-09 / SGLang GitHub README,2025–2026)
