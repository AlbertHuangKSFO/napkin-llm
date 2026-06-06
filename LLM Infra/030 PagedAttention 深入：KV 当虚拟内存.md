[[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention 深入]]把 [[LLM/102 KV-Cache|KV-Cache]] 当**操作系统的虚拟内存**来管:KV 切成定长**物理 block**散落显存,每序列一张**逻辑 block table**做"逻辑块号→物理块号"映射,**按需逐页分配**而非按 max_len 预留,碎片只剩末页(近零碎片),并用 **copy-on-write** 让多请求共享前缀 KV。它是 [[LLM/026 PagedAttention 与 KV 分页|PagedAttention]] 的系统级深入,把 [[015 KV-Cache 的显存账(逐层手算)|KV 显存账]]里那笔账的**碎片浪费**几乎清零,是 vLLM 吞吐提升 2–4× 的根。

## 直觉

老式服务器为让一条序列的 KV **物理连续**,得按 `max_len` 整块预留——像订酒店时不管住几晚都先包下整层楼,大半空着。PagedAttention 照搬 OS 分页:**别要求连续**。给进程的是连续虚拟地址,背后映射到散落物理页帧;进程以为内存连续,其实页表在变魔术。这里逻辑 block table = 页表,物理 block = 页帧。序列以为自己 KV 连续,注意力 kernel 按块表去散落的物理块取数,**结果与连续存储逐位相同**(不是近似)。

**生活类比**:把显存想成一个停车场,每条序列是一辆车。老办法像「订酒店包整层」——A 车明明只停 1 格,却按最长可能(max_len=6 格)把整片连号车位都包下,结果 5 格永远空着,利用率只有 1/6≈17%(对应内部碎片)。PagedAttention 改成「按需发零散空位 + 一张车位登记表」:A 车真正用到几格就给几格,1 号和 4 号给 A、2 号 5 号给 B、3 号给 C,各车的零散车位混在同一层,只剩最后一格可能半空,浪费**最多 1 格**(对应 `block_size−1`,与序列多长无关)。怎么取车?查那张「车位登记表」(就是 block table:A→[1,4]、B→[2,5]),车位散落也照取不误。于是同一停车场从塞 1 辆变成塞 5 辆,利用率 40%→90%+,并发翻 2–4 倍。

![[kv-030类比停车场.svg]]

## 例子

每个 block 放 16 个 token 的 K/V,某请求生成 100 token:
- **老式连续**:按 max_len=2048 预留,实际用 $100/2048\approx\textbf{5\%}$,其余永远空着(内部碎片);剩下的零碎块还凑不出连续大段(外部碎片)。利用率常 **<40%**。
- **分页**:分配 $\lceil 100/16\rceil=7$ 个 block,前 6 个满(96 token),第 7 个放 4 个、半满。浪费 = 末页 12 个空位 = **总量 12/112≈11%,且与 max_len 无关**。

碎片上界恒为 `block_size−1`=15 个 token:序列无论 100 还是 100000 token,浪费最多 15 个槽位。利用率 40%→90%+,同显存塞 **2–4× 并发** → 吞吐 2–4×。

## 原理

设 block 大小 $P$(token/块),序列长 $S$。每序列占用 $\lceil S/P\rceil$ 个物理块,浪费(末页未填)上界:

$$
\text{waste}(S)=\big(\lceil S/P\rceil\cdot P - S\big)\le P-1
$$

对比老式预留浪费 $=\text{max\_len}-S$,可逼近 $\text{max\_len}$。PagedAttention 把浪费从**正比于预留量**压成**小于一页的常数**,这是利用率跃升的数学根源。注意力计算 $q_i\cdot k_j$ 时,先查 block table 把逻辑位置 $j$ 映射到物理块再取 $k_j$,运算本身不变。

## 图

![[kv-分页块表映射.svg]]

![[kv-030虚拟内存类比映射.svg]]

![[kv-030预留vs分页浪费手算.svg]]

## 代码

```python
from vllm import LLM
# ✅ vLLM 默认就是 PagedAttention，block_size 默认 16；显存利用率上限可调
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    block_size=16,                 # 每页 16 token：块表开销 ↔ 末页浪费的折中
    gpu_memory_utilization=0.90,   # 页池占显存上限；触顶触发抢占/换出
    enable_prefix_caching=True,    # copy-on-write 共享前缀（见 032/033）
)

# 块表的本质：逻辑块号 -> 物理块号
class PagedKV:
    def __init__(self, total_pages, page_size, d):
        import torch
        self.pool = torch.zeros(total_pages, page_size, d)  # 统一物理页池（共享）
        self.free = list(range(total_pages))                # 空闲物理页号
        self.block_tables = {}                              # seq_id -> [物理页号...]

    def append_block(self, seq_id):
        tbl = self.block_tables.setdefault(seq_id, [])
        tbl.append(self.free.pop())   # ✅ 按需才申请；物理页可乱序、来自任意位置

    def share_prefix(self, src, dst, n):
        # ✅ copy-on-write：dst 块表前 n 块直接指向 src 的物理页，不拷数据
        self.block_tables[dst] = self.block_tables[src][:n].copy()

# ❌ 老式：按 max_len 预留连续大块 —— 序列再短也整段占着，内部碎片爆炸
#   torch.zeros(n_seq, max_len, d)   # 利用率 <40%，batch 开不大
# ❌ 易错：以为 PagedAttention 减少了 KV 总量 —— 不，它减的是碎片；减总量靠 GQA/量化
```

`❌` 把 PagedAttention 当成"省 KV 总量"是常见误解:它是**显存管理**层面的工程,减的是碎片;减总量要靠 [[LLM/019 GQA 分组查询注意力|GQA]] 与 [[034 KV 量化部署：FP8、INT8 KV|KV 量化]],两类正交可叠加。

## 面试高频

- **PagedAttention 解决什么?** [[LLM/102 KV-Cache|KV-Cache]] 的**显存碎片**(内部 + 外部):老式按 max_len 预留连续块,利用率常 <40%;分页后碎片只剩末页,90%+。
- **灵感来源?** OS 的**虚拟内存与分页**——逻辑连续、物理非连续,块表=页表。
- **它减少 KV 总量吗?** **不**。减碎片浪费(同显存塞更多请求 → 吞吐 2–4×);减总量要 [[LLM/019 GQA 分组查询注意力|GQA]]/量化,正交叠加。
- **copy-on-write 省在哪?** 多请求共享 system prompt、并行采样/beam 同前缀分叉时,公共 KV 页只存一份;分叉写入才复制那页 → 省显存 + 免重算前缀,即 [[032 前缀缓存：RadixAttention 树结构|前缀缓存]]。
- **block_size 怎么权衡?** 太小 → 块表长、kernel 索引开销大;太大 → 末页浪费上界大、前缀共享粒度粗。vLLM 默认 16。
- **显存还不够怎么办?** 抢占 + 换出(swap KV 到 CPU)或 recompute(重跑 prefill 有时更快),配合连续批处理——序列一完就释放页。
- **改变注意力数学吗?** 没有。只改 KV **存放布局**,计算结果逐位相同,纯系统工程。

## 关键事实

- Woosuk Kwon 等,*Efficient Memory Management for LLM Serving with PagedAttention*,2023,**arXiv:2309.06180**(SOSP 2023),即 **vLLM** 核心算法。
- 机制:KV 切定长物理 block,block table 映射逻辑↔物理,允许 KV 存于**非连续**显存;按需分配 + copy-on-write 共享前缀。
- 碎片上界 = `block_size−1` 个 token,**与 max_len、序列长无关**;利用率 <40% → 90%+。
- 效果:吞吐较 FasterTransformer、Orca **提升 2–4×**(同延迟,vLLM 论文)。与 [[LLM/103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV 优化]] 互补:后者减总量,前者去碎片。
