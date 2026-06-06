[[026 PagedAttention 与 KV 分页|PagedAttention 与 KV 分页]]:借鉴操作系统的**虚拟内存/分页**,把 [[102 KV-Cache|KV-Cache]] 切成定长"页"、用块表把逻辑连续映射到物理非连续,消除碎片、利用率冲到 90%+,让批量更大、吞吐提升 2–4×(vLLM 的核心)。

## 直觉:KV-Cache 像在显存里反复"开大房间"
自回归解码时,每个请求的 [[102 KV-Cache|KV-Cache]] 会**随生成长度动态变长**。老式服务系统为了让一条序列的 KV 在显存里**连续存放**,必须按 `max_len` 预先**整块预留**一大段显存。问题:
- **内部碎片**:实际只生成 100 token,却按 2048 预留 → 大半空着浪费。
- **外部碎片**:显存被一块块占着,剩下的零碎空间凑不出一段连续大块 → 明明有空闲却分不出来。

结果:真实系统 KV 显存利用率常常**不到 40%**,batch 开不大,GPU 闲着。

PagedAttention 的招数照搬 OS:**别要求连续**。把 KV 切成固定大小的**页(block)**(例如每页放 16 个 token 的 K/V),物理上散落在显存各处;每条序列维护一张**块表(block table)**,记录"逻辑第几页 → 物理哪一页"。注意力计算时按块表去取页即可。

类比:OS 给进程的是连续的虚拟地址,背后映射到散落的物理页帧;进程以为自己有一整段连续内存,其实是页表在变魔术。PagedAttention 就是把这套搬到 KV-Cache 上。

## 例子:碎片去哪了
假设每页放 4 个 token,某请求生成了 10 个 token:
- **老式**:按 max_len=2048 预留连续显存,实际用 10/2048 ≈ **0.5%**,其余全浪费(内部碎片)。
- **PagedAttention**:分配 $\lceil 10/4\rceil=3$ 页;前 2 页满(8 token),第 3 页放 2 个、半满。浪费 = 末页的 2 个空位 = **总量的 2/12 ≈ 17%,且与 max_len 无关**。

把"按最坏情况预留"换成"按需一页页给",浪费从"整段空间"缩成"最多不到一页"。利用率 40% → 90%+,**同样显存能塞下 2–4 倍的并发请求** → 吞吐 2–4×。

**碎片上界的精确刻画(为什么浪费"与序列长无关")。** 每序列的浪费 = 末页未填满的槽位,上界恒为"页大小 − 1"个 token。设页大小 16:无论序列是 100 还是 100000 token,浪费最多 15 个 token 的空间 → 长序列里 15/100000 可忽略,短序列里 15/100 也才 ~13%。对比老式"按 max_len 预留",浪费 $\propto(\text{max\_len}-\text{实际长度})$,可能高达 99%。**PagedAttention 把浪费从"正比于预留量"压成"小于一页的常数"**,这是利用率从 <40% 跃到 90%+ 的数学根源。

**页大小怎么选(权衡)。** 页太小(如 1):块表很长、每序列要管很多页、kernel 取数索引开销大;页太大(如 256):末页浪费上界变大、共享前缀的粒度变粗。vLLM 默认 16,是"块表开销"与"末页浪费"的折中。

## 原理:块表 + 按需分配 + 写时复制
**① 分页存储。** 把每个序列的 KV 切成固定大小的 KV block,每块存若干 token 的 K、V。物理上这些块在显存的统一**页池**里任意位置,不要求相邻。

**② 块表映射。** 每序列一张块表,逻辑块号 $\to$ 物理块号。注意力 kernel 计算 $q_i\cdot k_j$ 时,按块表定位 $k_j$ 所在物理块再取数。逻辑上"连续",物理上"随便放"。

![[attn-PagedAttention分页表.svg]]

**③ 按需分配。** 序列每多生成一块的 token 才申请一个新页,从不预留 `max_len`。释放时整页归还页池。碎片只可能出现在**每序列的最后一页**(末页半满),上界是"页大小",与序列长度、与 `max_len` 都无关。

**④ 写时复制(copy-on-write)共享前缀。** 多个请求若共享同一段前缀(同一 system prompt、或并行采样 / beam search 从同一前缀分叉),它们的公共前缀 KV **只存一份**,多张块表指向同一批物理页;只有在分叉、需要写入不同内容时才**复制那一页**。这既省显存又省去重复计算前缀 KV——与 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|前缀共享]] 是同一思想的系统实现。

**⑤ 抢占与换出(显存不够时怎么办)。** 并发太多、页池耗尽时,vLLM 会**抢占**(preempt)部分序列:要么把它们的 KV 页**换出(swap)到 CPU 内存**、腾出 GPU 页给优先序列,稍后再换回;要么直接**丢弃并重算**(recompute,因为重跑 prefill 有时比换入换出还快)。这又是照搬 OS——进程内存不够就换页到磁盘。结合**连续批处理(continuous batching)**:一条序列生成完就立即释放其页、让新请求填进 batch,GPU 不空转。分页 + 连续批处理 + 抢占,三者共同把吞吐顶上去。

**和别的 KV 优化什么关系?** [[019 GQA 分组查询注意力|GQA]]/[[018 MQA 多查询注意力|MQA]]/量化 减的是 **KV 的总量**(每 token 占多少字节);PagedAttention 减的是**管理 KV 的碎片浪费**(同样总量能塞更多)。两类正交,叠加用。它不改 [[014 注意力复杂度 O(n²) 与瓶颈|算力复杂度]],纯属**显存管理**层面的工程突破。

## 代码:块表把逻辑映射到物理
```python
import torch

# ❌ 老式:每序列预留连续大块,按 max_len,浪费严重
class ContiguousKV:
    def __init__(self, max_len, d, n_seq):
        # 即使序列很短,也整块占着 max_len —— 内部碎片
        self.cache = torch.zeros(n_seq, max_len, d)   # 大量空间永远用不到

# ✅ PagedAttention:统一页池 + 每序列一张块表
class PagedKV:
    def __init__(self, total_pages, page_size, d):
        self.pool = torch.zeros(total_pages, page_size, d)  # 物理页池(共享)
        self.free = list(range(total_pages))                # 空闲物理页号
        self.block_tables = {}                              # seq_id -> [物理页号...]

    def append(self, seq_id, k, v_unused):                  # 写入一个 token 的 K
        tbl = self.block_tables.setdefault(seq_id, [])
        last = tbl[-1] if tbl else None
        # 只有当末页写满,才申请新页 —— 按需分配,不预留
        if last is None or self._page_full(seq_id):
            tbl.append(self.free.pop(0))                    # 物理页可乱序、可来自任意位置
        # ... 把 k 写进 tbl[-1] 这一物理页的下一个槽位

    def share_prefix(self, src, dst, n_pages):
        # 写时复制:dst 直接指向 src 的前缀物理页,不复制数据
        self.block_tables[dst] = self.block_tables[src][:n_pages].copy()

    def _page_full(self, seq_id): ...
# ❌ 易错:以为 PagedAttention 减少了 KV 总量 —— 不,它减少的是碎片浪费。减总量要靠 GQA/量化。
```

## 面试高频
- **PagedAttention 解决什么问题?** [[102 KV-Cache|KV-Cache]] 的**显存碎片**(内部 + 外部):老式按 max_len 预留连续块,利用率常 <40%。它按页管理,碎片只剩末页,利用率 90%+。
- **灵感来源?** 操作系统的**虚拟内存与分页**——逻辑连续、物理非连续,用页表(此处块表)映射。
- **它减少 KV 的总量吗?** **不**。减的是碎片浪费(同显存塞更多请求 → 吞吐 2–4×)。减总量要靠 [[019 GQA 分组查询注意力|GQA]]/[[018 MQA 多查询注意力|MQA]]/量化,二者正交可叠加。
- **为什么吞吐能涨这么多?** 显存省下来 → batch 能开更大 → GPU 利用率高 → 同延迟下吞吐 2–4×(vLLM 论文对比 FasterTransformer、Orca)。
- **写时复制省在哪?** 多请求共享 system prompt、并行采样共享前缀时,公共 KV 页只存一份;分叉才复制 → 省显存又免重算前缀,即 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|前缀共享]]。
- **它和 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 冲突吗?** 不冲突。FlashAttention 优化单次注意力的 IO;PagedAttention 优化 KV 的显存布局。工程上常一起用(vLLM 内核兼容分页 KV)。
- **页大小怎么权衡?** 太小 → 块表长、索引开销大;太大 → 末页浪费上界变大、前缀共享粒度粗。vLLM 默认 16。
- **显存还是不够时怎么办?** 抢占 + 换出(swap KV 到 CPU)或重算(recompute prefill),配合连续批处理(序列一完就释放页),与 OS 的换页同思想。
- **PagedAttention 改变了注意力的数学吗?** 没有。它只改 KV 在显存里的**存放布局**(逻辑连续、物理分散),注意力计算结果与连续存储**逐位相同**;纯系统/显存工程,不是近似。
- **前缀缓存(prefix caching)和它什么关系?** 写时复制让多请求共享同一段前缀 KV 物理页,正是前缀缓存的实现:相同 system prompt 只算一次、存一份;这也是多轮对话/批量采样省显存省算力的来源。

## 关键事实
- Woosuk Kwon、Zhuohan Li、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica,*Efficient Memory Management for Large Language Model Serving with PagedAttention*,2023,arXiv:2309.06180(SOSP 2023)。即 **vLLM** 系统的核心算法。
- 机制:KV-Cache 分定长页,块表映射逻辑↔物理,允许 KV 存于**非连续**显存;按需分配 + 写时复制共享前缀。
- 效果:KV 显存浪费从 60–80% 降到近乎只剩末页;吞吐较 FasterTransformer、Orca **提升 2–4×**(同延迟)。
- 定位:[[102 KV-Cache|KV-Cache]] 的系统级管理方案,与 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]] 互补——后者减总量,前者去碎片;纯显存工程,不改注意力数学。
