[[031 KV 显存碎片与 block 管理|KV 显存碎片与 block 管理]]把 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]] 里"碎片"这件事拆细:**内部碎片**(块内/预留内填不满)、**外部碎片**(零碎空闲凑不出连续块),以及 `block_size` 取舍、预留 watermark、抢占触发与显存**利用率**。它是把 [[015 KV-Cache 的显存账(逐层手算)|KV 显存账]]那笔账"实际能用多少"讲清的一篇——账算得再准,碎片吃掉一半也白搭。

## 直觉

显存碎片有两类,像停车场:
- **内部碎片**:你停一辆小车,却占了一个加长车位(老式按 max_len 预留 → 序列短也整段占着;分页时则是末页半满)。空间分给你了但没用满。
- **外部碎片**:整个停车场总空位很多,但都是东一个西一个的零散车位,来一辆大巴(需要连续大段)却停不进。空间有但**凑不出连续块**。

老式连续 KV 两种碎片都中招,利用率常 **<40%**。PagedAttention 用定长 block 一举消掉外部碎片(任何空闲块都能用),内部碎片压到只剩每序列末页。

## 例子

`block_size=16`,三个并发序列长 100 / 50 / 200:
- **末页浪费**:$100\to\lceil100/16\rceil=7$ 块,浪费 $7\cdot16-100=12$;$50\to 4$ 块浪费 14;$200\to 13$ 块浪费 8。三者共浪费 $12+14+8=34$ token,占 $34/(350+34)\approx\textbf{8.9\%}$。
- **碎片率随 `block_size` 变**:`block_size=1` → 浪费 0 但块表巨长、kernel 聚集开销大;`block_size=256` → 末页上界 255,短序列浪费率飙升。vLLM 默认 16 是折中。

对比老式:三序列都按 max_len=2048 预留 → 用 $350/(3\cdot2048)\approx\textbf{5.7\%}$,其余全废。

## 原理

总末页浪费 = 各序列末页空槽之和,期望 $\approx N\cdot(P-1)/2$($N$=并发序列数,$P$=block 大小)。碎片率:

$$
\text{frag\_rate}=\frac{\sum_i\big(\lceil S_i/P\rceil P - S_i\big)}{\sum_i \lceil S_i/P\rceil P}\le\frac{N(P-1)}{\sum_i S_i + N(P-1)}
$$

$P$ 越大碎片率越高、块表越短;$P$ 越小反之。**预留 watermark**:vLLM 不会把页池用到 100%,留少量空闲块给即将 append 的 token 与 swap;触及 `gpu_memory_utilization` 上限即停止接新请求或**抢占**已有序列(swap 到 CPU 或 recompute)。利用率 = 实际承载 KV / 页池容量,目标顶到 90%+ 而不溢出。

## 图

![[kv-碎片对比.svg]]

![[kv-031碎片率手算.svg]]

## 代码

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    block_size=16,                  # ✅ 折中：小→块表长&索引贵；大→末页浪费大
    gpu_memory_utilization=0.90,    # ✅ 留 watermark，触顶触发抢占而非 OOM
    swap_space=4,                   # GiB，抢占时 KV 换出到 CPU 的空间
    # preemption_mode 默认 recompute（重算前缀常比 swap 往返还快）
)

# 估碎片率：给定并发序列长，算末页浪费占比
import math
def frag_rate(seq_lens, block=16):
    used  = sum(seq_lens)
    alloc = sum(math.ceil(s / block) * block for s in seq_lens)
    return (alloc - used) / alloc

print(frag_rate([100, 50, 200], 16))   # ≈ 0.089  ✅ 分页：碎片个位数百分比
print(frag_rate([100, 50, 200], 256))  # ≈ 0.59   ❌ block 过大：末页浪费吃掉一半

# ❌ 反面：老式按 max_len 预留 —— 利用率 <40%，并发被碎片压死
#   waste = sum(max_len - s for s in seq_lens)   # 正比于预留量，可达 95%+
```

`❌` `block_size` 不是越大越好:大块缩短块表却放大末页浪费,长上下文 + 高并发下碎片率反而难看;短而合理(16)兼顾索引开销与浪费。

## 面试高频

- **内部 vs 外部碎片?** 内部 = 分到手但没填满(预留内/末页);外部 = 总空闲够但不连续、凑不出块。老式连续 KV 两者都中,分页消外部、只留末页内部。
- **碎片率怎么估?** $\sum(\lceil S/P\rceil P - S)/\sum\lceil S/P\rceil P$,上界 $\approx N(P-1)/(\sum S + N(P-1))$。
- **block_size 怎么选?** 小:碎片低但块表长、kernel 聚集/索引开销大、前缀共享细;大:反之。vLLM 默认 16。
- **预留/watermark 为何要?** 留空闲块给 append 与 swap,避免瞬时 OOM;触顶就抢占(swap/recompute)而非崩。
- **抢占何时触发?** 页池将耗尽、新 token 无块可分时,挑序列换出到 CPU 或丢弃重算,配合连续批处理。
- **利用率为何 90%+ 就到顶?** 末页碎片 + watermark 占掉个位数百分比,90%+ 已接近可用上界。

## 关键事实

- 碎片两类:**内部**(预留/末页填不满)、**外部**(零碎空闲不连续);老式连续 KV 利用率常 **<40%**(vLLM, arXiv:2309.06180, 2023)。
- PagedAttention 消外部碎片,内部碎片上界 = `block_size−1` 个 token/序列;vLLM 默认 `block_size=16`。
- watermark + 抢占(swap KV 到 CPU 或 recompute prefill)在页池将满时兜底,避免 OOM,利用率顶到 90%+。
- 碎片是显存管理问题,与 [[LLM/019 GQA 分组查询注意力|GQA]]/[[034 KV 量化部署：FP8、INT8 KV|量化]](减总量)正交;两者叠加才把一张卡的并发与上下文榨满。
