[[026 PagedAttention 与 KV 分页|PagedAttention 与 KV 分页]]:借鉴操作系统的**虚拟内存/分页**,把 [[102 KV-Cache|KV-Cache]] 切成定长块、用块表把逻辑连续映射到物理非连续,从而把按需增长的 KV 管理从“整段预留”改为“按块分配”。vLLM 论文在其模型、长度分布与延迟约束下报告了接近零浪费和 2–4× 吞吐；部署收益仍取决于块大小、请求分布与调度器。

## 直觉:KV-Cache 像在显存里反复"开大房间"
自回归解码时,每个请求的 [[102 KV-Cache|KV-Cache]] 会**随生成长度动态变长**。老式服务系统为了让一条序列的 KV 在显存里**连续存放**,必须按 `max_len` 预先**整块预留**一大段显存。问题:
- **内部碎片**:实际只生成 100 token,却按 2048 预留 → 大半空着浪费。
- **外部碎片**:显存被一块块占着,剩下的零碎空间凑不出一段连续大块 → 明明有空闲却分不出来。

结果:如果服务实现为每条请求预留连续的最大长度,内部碎片会限制可并发的 batch；实际浪费比例由请求长度分布、预留策略与实现决定，不能把某一论文实验的比例当作通用常数。

PagedAttention 的招数照搬 OS:**别要求连续**。把 KV 切成固定大小的**页(block)**(例如每页放 16 个 token 的 K/V),物理上散落在显存各处;每条序列维护一张**块表(block table)**,记录"逻辑第几页 → 物理哪一页"。注意力计算时按块表去取页即可。

类比:OS 给进程的是连续的虚拟地址,背后映射到散落的物理页帧;进程以为自己有一整段连续内存,其实是页表在变魔术。PagedAttention 就是把这套搬到 KV-Cache 上。

## 例子:碎片去哪了
假设每页放 4 个 token,某请求生成了 10 个 token:
- **老式**:按 max_len=2048 预留连续显存,实际用 10/2048 ≈ **0.5%**,其余全浪费(内部碎片)。
- **PagedAttention**:分配 $\lceil 10/4\rceil=3$ 页;前 2 页满(8 token),第 3 页放 2 个、半满。浪费 = 末页的 2 个空位 = **总量的 2/12 ≈ 17%,且与 max_len 无关**。

把"按最坏情况预留"换成"按需一页页给",由末页造成的**每序列**内部浪费最多小于一页。可并发数是否明显增加，仍受模型权重、KV 每 token 字节数、页池余量和调度策略共同约束。

**碎片上界的精确刻画(为什么浪费"与序列长无关")。** 每序列的浪费 = 末页未填满的槽位,上界恒为"页大小 − 1"个 token。设页大小 16:无论序列是 100 还是 100000 token,浪费最多 15 个 token 的空间；长序列里 $15/100000$ 可忽略，100-token 序列的粗上界是 $15/100=15\%$。对比老式"按 max_len 预留",浪费 $\propto(\text{max\_len}-\text{实际长度})$,可能接近预留量。**PagedAttention 把每序列末页的内部浪费压成“小于一页”的常数上界**；总利用率仍须看页池、请求分布和共享情况。

**页大小怎么选(权衡)。** 页太小(如 1):块表很长、每序列要管很多页、kernel 取数索引开销大；页太大(如 256):末页浪费上界变大、共享前缀的粒度变粗。块大小是引擎、硬件与版本相关的运行参数，应以目标模型和真实长度分布压测后确定，不把某个版本的默认值当作算法结论。

## 原理:块表 + 按需分配 + 写时复制
**① 分页存储。** 把每个序列的 KV 切成固定大小的 KV block,每块存若干 token 的 K、V。物理上这些块在显存的统一**页池**里任意位置,不要求相邻。

**② 块表映射。** 每序列一张块表,逻辑块号 $\to$ 物理块号。注意力 kernel 计算 $q_i\cdot k_j$ 时,按块表定位 $k_j$ 所在物理块再取数。逻辑上"连续",物理上"随便放"。

![[attn-PagedAttention分页表.png]]

**③ 按需分配。** 序列每多生成一块的 token 才申请一个新页,从不预留 `max_len`。释放时整页归还页池。碎片只可能出现在**每序列的最后一页**(末页半满),上界是"页大小",与序列长度、与 `max_len` 都无关。

**④ 写时复制(copy-on-write)共享前缀。** 多个请求若共享同一段前缀(同一 system prompt、或并行采样 / beam search 从同一前缀分叉),它们的公共前缀 KV **只存一份**,多张块表指向同一批物理页;只有在分叉、需要写入不同内容时才**复制那一页**。这既省显存又省去重复计算前缀 KV——与 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|前缀共享]] 是同一思想的系统实现。

**⑤ 抢占与恢复(显存不够时怎么办)。** 页池耗尽时，服务调度器可抢占部分序列，并依实现与版本选择重算、CPU/offload 或拒绝新请求等恢复路径。这里的关键不在于某个固定模式，而在于：被抢占序列恢复后必须重新拥有同一条**逻辑 token 顺序**的 K/V；调度、换入/重算与连续批处理共同决定端到端吞吐和尾延迟。

![[attn-PagedKV-COW.png]]

**和别的 KV 优化什么关系?** [[019 GQA 分组查询注意力|GQA]]/[[018 MQA 多查询注意力|MQA]]/量化 减的是 **KV 的总量**(每 token 占多少字节);PagedAttention 减的是**管理 KV 的碎片浪费**(同样总量能塞更多)。两类正交,叠加用。它不改 [[014 注意力复杂度 O(n²) 与瓶颈|算力复杂度]],纯属**显存管理**层面的工程突破。

## 代码:块表把逻辑映射到物理
```python
import torch

# ❌ 老式:每序列预留连续大块,按 max_len,浪费严重
class ContiguousKV:
    def __init__(self, max_len, d, n_seq):
        # 即使序列很短,也整块占着 max_len —— 内部碎片
        self.cache = torch.zeros(n_seq, max_len, d)   # 大量空间永远用不到

# ✅ 教学安全版:统一页池 + 块表 + 逻辑长度 + K/V + 引用计数/COW。
# 它只演示元数据不变量；真实内核还会按层、头和 dtype 组织张量。
class PagedKV:
    def __init__(self, total_pages, page_size, d):
        self.k_pool = torch.zeros(total_pages, page_size, d)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free = list(range(total_pages))
        self.refcount = [0] * total_pages
        self.block_tables, self.lengths = {}, {}             # seq_id -> 物理页；seq_id -> 逻辑长度

    def _alloc(self):
        page = self.free.pop()
        self.refcount[page] = 1
        self.k_pool[page].zero_(); self.v_pool[page].zero_()
        return page

    def _cow_tail(self, seq_id):
        """只在将要写共享的末页时复制；写入 offset 由逻辑长度决定。"""
        old = self.block_tables[seq_id][-1]
        if self.refcount[old] == 1:
            return old
        new = self._alloc()
        self.k_pool[new].copy_(self.k_pool[old]); self.v_pool[new].copy_(self.v_pool[old])
        self.refcount[old] -= 1
        self.block_tables[seq_id][-1] = new
        return new

    def append(self, seq_id, k, v):                           # 同时写入一对 K/V
        table = self.block_tables.setdefault(seq_id, [])
        logical_len = self.lengths.setdefault(seq_id, 0)
        offset = logical_len % self.page_size
        if offset == 0:
            table.append(self._alloc())                       # 新逻辑块映射到任意物理页
        else:
            self._cow_tail(seq_id)                            # 共享的部分页写前必须 COW
        page = table[-1]
        self.k_pool[page, offset].copy_(k); self.v_pool[page, offset].copy_(v)
        self.lengths[seq_id] += 1

    def fork_prefix(self, src, dst, n_tokens):
        assert 0 <= n_tokens <= self.lengths[src]
        n_pages = (n_tokens + self.page_size - 1) // self.page_size
        table = self.block_tables[src][:n_pages].copy()
        for page in table: self.refcount[page] += 1
        self.block_tables[dst], self.lengths[dst] = table, n_tokens

    def logical_kv(self, seq_id):
        out_k, out_v = [], []                                 # 按块表恢复逻辑顺序，不能按物理页号排序
        for pos in range(self.lengths[seq_id]):
            page = self.block_tables[seq_id][pos // self.page_size]
            off = pos % self.page_size
            out_k.append(self.k_pool[page, off]); out_v.append(self.v_pool[page, off])
        return torch.stack(out_k), torch.stack(out_v)

    def release(self, seq_id):
        for page in self.block_tables.pop(seq_id):            # 共享页要等最后一个引用释放
            self.refcount[page] -= 1
            if self.refcount[page] == 0:
                self.free.append(page)
        self.lengths.pop(seq_id)

    @property
    def page_size(self): return self.k_pool.shape[1]

cache = PagedKV(total_pages=8, page_size=2, d=1)
for x in (1., 2., 3.): cache.append("parent", torch.tensor([x]), torch.tensor([10*x]))
cache.fork_prefix("parent", "child", n_tokens=3)
cache.append("child", torch.tensor([9.]), torch.tensor([90.]))  # 第 2 页共享，先 COW 再写 offset=1
pk, pv = cache.logical_kv("parent"); ck, cv = cache.logical_kv("child")
assert pk[:, 0].tolist() == [1., 2., 3.] and pv[:, 0].tolist() == [10., 20., 30.]
assert ck[:, 0].tolist() == [1., 2., 3., 9.] and cv[:, 0].tolist() == [10., 20., 30., 90.]
cache.release("child"); cache.release("parent")
# ❌ 易错:以为 PagedAttention 减少了 KV 总量 —— 不,它减少的是碎片浪费。减总量要靠 GQA/量化。
```

## 面试高频
- **PagedAttention 解决什么问题?** [[102 KV-Cache|KV-Cache]] 的按需增长与显存碎片：按块分配后，每条序列的内部浪费上界小于一块，并消除“必须找连续大块”的限制；实际利用率要以工作负载测量。
- **灵感来源?** 操作系统的**虚拟内存与分页**——逻辑连续、物理非连续,用页表(此处块表)映射。
- **它减少 KV 的总量吗?** **不**。它减少的是碎片浪费，因而在受内存约束的工作负载中可能提高并发。减每 token 的 KV 字节数要靠 [[019 GQA 分组查询注意力|GQA]]/[[018 MQA 多查询注意力|MQA]]/量化；vLLM 论文的 2–4× 是特定实验结果。
- **为什么可能增吞吐?** 显存碎片下降后可容纳更大 batch，GPU 利用率可能提高。vLLM 论文在其测试模型、长度分布和同延迟约束下报告相对 FasterTransformer/Orca 的 2–4×；迁移到别的模型和调度器前应复测。
- **写时复制省在哪?** 多请求共享 system prompt、并行采样共享前缀时,公共 KV 页只存一份;分叉才复制 → 省显存又免重算前缀,即 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|前缀共享]]。
- **它和 [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]] 冲突吗?** 不冲突。FlashAttention 优化单次注意力的 IO;PagedAttention 优化 KV 的显存布局。工程上常一起用(vLLM 内核兼容分页 KV)。
- **页大小怎么权衡?** 太小 → 块表长、索引开销大；太大 → 末页浪费上界变大、前缀共享粒度粗。它是实现与硬件相关的参数，应按真实请求分布压测。
- **实现 COW 时最容易错什么?** 共享部分末页时，子序列追加必须先复制该页；还要维护 `logical_length`、K 和 V 的同一 offset、物理页引用计数与释放。按物理页号遍历会打乱逻辑 token 顺序。
- **PagedAttention 改变了注意力的数学吗?** 没有。它只改 KV 在显存里的**存放布局**(逻辑连续、物理分散)，数学上等价且不引入近似；但这不保证逐位相同，具体内核仍可能因浮点归约或执行顺序不同出现低位差异。它是系统/显存工程，不是近似注意力。
- **前缀缓存(prefix caching)和它什么关系?** 写时复制让多请求共享同一段前缀 KV 物理页,正是前缀缓存的实现:相同 system prompt 只算一次、存一份;这也是多轮对话/批量采样省显存省算力的来源。

## 关键事实
- Woosuk Kwon、Zhuohan Li、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica,*Efficient Memory Management for Large Language Model Serving with PagedAttention*,2023,arXiv:2309.06180(SOSP 2023)。即 **vLLM** 系统的核心算法。
- 机制:KV-Cache 分定长页,块表映射逻辑↔物理,允许 KV 存于**非连续**显存;按需分配 + 写时复制共享前缀。
- 效果(论文实验，2023):在该论文的模型、请求长度与同延迟设置中，vLLM 相对 FasterTransformer、Orca 报告 **2–4×** 吞吐提升；不要外推为所有模型或所有版本的固定收益。
- 定位:[[102 KV-Cache|KV-Cache]] 的系统级管理方案,与 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]] 互补——后者减总量,前者去碎片;纯显存工程,不改注意力数学。
