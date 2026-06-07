[[012 自回归推理全流程：一个 token 的旅程|自回归推理全流程]]指 LLM 在推理时**逐个生成 token**的完整张量流水线:一个 token id 先查 [[LLM/102 KV-Cache|embedding]] 表变成向量,顺序流过 N 层 Transformer(每层 self-attention + FFN + 残差/LayerNorm),最后经 LM Head 投影成词表维 logits,再由[[LLM/100 解码策略：贪心与 Beam|解码]]策略采样出下一个 token——然后把它接回输入,循环往复。整条流水线天然分裂成两个特性迥异的阶段:[[013 Prefill 阶段：计算受限|Prefill]] 把整段 prompt 一次性灌进去,[[014 Decode 阶段：访存受限|Decode]] 之后每步只喂 1 个新 token。理解这条旅程是理解一切推理服务指标的地基。

## 直觉

把 Transformer 想成一条**装配线**,工位固定(权重不变),零件(token 向量)从一头进、另一头出。

- **Prefill** = 一整托盘零件(整个 prompt)同时上线,装配线一次走满,产能拉满。
- **Decode** = 之后每次只放 1 个零件上线,装配线为这 1 个零件空跑一整趟。工位还是那些工位,但每趟只服务 1 件——极度浪费产能,瓶颈从"算得快不快"变成"把工位的模具(权重)搬到手有多快"。

"自回归"的含义就是:第 $t+1$ 个 token 的输入依赖第 $t$ 个的输出,**必须串行**,无法像 prefill 那样并行展开。这串行性正是 decode 慢、且决定 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|TPOT]] 的根因。

## 例子

给定 prompt "今天天气怎么" 续写,词表大小 $V=128000$,$d_\text{model}=4096$,32 层:

1. **查表**:"今"→id 1234 →embedding 表第 1234 行,得 `[4096]` 向量。prompt 4 个 token 一起 →`[4, 4096]` 矩阵。
2. **过 32 层**:每层做 attention(写入 KV-Cache)+ FFN,形状始终 `[4, 4096]`。
3. **LM Head**:取最后一个位置的 `[4096]` ×`[4096, 128000]`→`[128000]` logits。
4. **采样**:softmax 后 top-p 采样,得 "样"→喂回。
5. **decode 循环**:第 2 步只输入 "样" 这 1 个 token,形状 `[1, 4096]`,过 32 层(读全部 KV-Cache),出 1 个 logits,采 "?"……直到 EOS。

Prefill 一次处理 4 个 token;decode 之后每步处理 1 个。若续写 200 token,就是 1 次 prefill + 199 次 decode。

## 原理

设隐状态 $h\in\mathbb{R}^{S\times d}$($S$=本步 token 数)。每层核心两步:

$$
\text{Attn}(h)=\mathrm{softmax}\!\Big(\frac{(hW_Q)(hW_K)^\top}{\sqrt{d_k}}\Big)(hW_V),\qquad
\text{FFN}(x)=\sigma(xW_1)W_2
$$

- **Prefill**:$S=$ prompt 长度,上式是**矩阵×矩阵**(大 GEMM),激活又高又宽,权重读一次被 $S$ 行复用 → 算术强度高。
- **Decode**:$S=1$,上式退化成**向量×矩阵**(GEMV),激活只有 1 行,权重读一次只服务 1 行 → 算术强度低。

末层后 logits $z=h_\text{last}W_\text{LM}\in\mathbb{R}^{V}$,采样得 $x_{t+1}\sim P(\cdot\mid x_{1:t})$,再 append 回序列——这就是"自回归"。详见 [[004 算力 vs 带宽：Roofline 与算术强度|Roofline]] 对两阶段落点的分析。

## 图

![[pd-一个token的张量旅程.png]]

![[pd-012自回归循环.png]]

![[pd-012两阶段形状对比.png]]

## 代码

PyTorch 模拟两阶段、并对比"每步重算整段历史"的错误写法:

```python
import torch, time

@torch.no_grad()
def autoregressive(model, prompt_ids, n_new, use_cache=True):
    # ✅ 正确:prefill 一次性灌入 + decode 复用 KV-Cache
    out = model(prompt_ids, use_cache=use_cache)      # prefill:整段并行
    past = out.past_key_values
    next_id = out.logits[:, -1].argmax(-1, keepdim=True)
    generated = [next_id]
    for _ in range(n_new - 1):                        # decode:每步喂 1 token
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_id)
    return torch.cat(generated, dim=1)

# ❌ 错误:每步把"全部历史"重新喂一遍、不带 cache
def autoregressive_bad(model, prompt_ids, n_new):
    seq = prompt_ids
    for _ in range(n_new):
        out = model(seq)                  # 每步重算整段 → O(L^2) FLOPs，慢几十倍
        seq = torch.cat([seq, out.logits[:, -1].argmax(-1, keepdim=True)], 1)
    return seq
```

`❌` 每步把整段序列重新 forward,prefill 计算被白白重复 $L$ 次;`✅` 用 `past_key_values` 让 decode 每步只算新 token,这正是 [[LLM/102 KV-Cache|KV-Cache]] 的意义。

## 面试高频

- **Q:推理为什么分 prefill / decode 两阶段?** prompt 已知可并行(prefill,大 GEMM);生成必须串行依赖前一个 token(decode,GEMV),二者算术强度和瓶颈完全不同。
- **Q:一个 token 从输入到输出经过哪些张量操作?** embedding 查表 → N×(attention+FFN+残差/LN)→ LM Head 投影到词表 → 采样。
- **Q:为什么 decode 比 prefill 慢得多(按 token 算)?** decode 每步 $S=1$,权重搬运字节不被复用,是 [[014 Decode 阶段：访存受限|访存受限]];prefill 大 batch 摊薄了权重搬运。
- **Q:自回归的"串行"为何无法绕开?** 第 $t+1$ token 的输入是第 $t$ 的采样结果,数据依赖天然串行(投机解码只是用小模型猜、大模型批量校验来缓解)。

## 关键事实

- 续写 $L$ 个 token = **1 次 prefill + (L−1) 次 decode**;这是推理成本结构的根本拆分(2023 起成为服务系统标准心智模型)。
- vLLM/TensorRT-LLM 等(截至 2025)均把 prefill 与 decode 分别调度;DistServe(2024)进一步在不同 GPU 上**物理分离**两阶段以优化 goodput。
- 不带 KV-Cache 的朴素自回归是 $O(L^2)$ 投影计算;带 cache 降到 $O(L)$,实际加速数倍到数百倍。
