这是 [[LLM Infra]] 域对 [[066 NCCL：集合通信库与原语|NCCL]] all-reduce **内部算法**的拆解,直接决定 [[060 数据并行与副本扩展|数据并行]] 梯度同步和 [[058 TP 的通信：每层 all-reduce 与 NVLink 依赖|TP 通信]] 的快慢,参见 [[LLM/069 数据并行与 AllReduce|AllReduce]] 与 [[LLM/074 通信原语与计算通信重叠|通信原语]]。一句话:**同一个 all-reduce 有三种实现——ring 抢带宽(适合大消息/机内)、tree 抢延迟(适合小消息/几千卡)、SHARP 把求和搬进交换机(数据不来回搬、还省 GPU 算力);NCCL 按消息大小和规模自动选,选错算法能差几倍。**

类比:全班交作业求平均分。**ring**=作业沿一圈课桌传一遍累加、再传一圈把结果发回,每人手里过的纸最少(带宽省)但要绕整整两圈(人多就慢);**tree**=分组组长先汇总、组长再汇总到班长,层层上去 $\log N$ 步就到顶,人再多也快(延迟省);**SHARP**=讲台上的计算器(交换机)边收边加,作业只往上交一次,讲台直接报平均——纸不来回飞,学生(GPU)腾出手干别的。

**生活类比**:全班要凑出总账/平均分,三种凑法。**ring**(传话凑账):账本沿一圈课桌传一遍累加、再传一圈把结果发回,每人手里过的纸最少(带宽省),但要绕整整两圈——1024 人就得传 ~2046 步,人多就慢。**tree**(锦标赛淘汰式汇总):小组组长先各自汇总,再上交给班长,层层往上 log N 步就到顶——1024 人只要 ~20 步,小消息上比 ring **快约 100 倍**。**SHARP**(讲台计算器当裁判):交换机就是讲台上的计算器,边收边当场加总,作业只往上交一次、不来回飞,还把 all-reduce 占用的 SM 从 16 个降到 ≤6 个,学生(GPU)腾出手算别的。一句话:大消息/机内用 ring 抢带宽,小消息/几千卡用 tree 抢延迟,有硬件就上 SHARP 网内归约。技术对应:课桌一圈=ring 拓扑、组长层级=double binary tree、讲台=交换机网内计算。

![[net-067类比传话凑账.png]]

为什么 ring 带宽最优但延迟差?ring all-reduce = reduce-scatter($N-1$ 步)+ all-gather($N-1$ 步),共 $2(N-1)$ 步,每步只搬 $M/N$。每卡总收发 $\tfrac{2(N-1)}{N}M\to 2M$,与 $N$ 几乎无关(带宽最优);但**步数随 $N$ 线性增长**,延迟 $\propto N$。tree 反过来:延迟 $\propto \log N$,小消息(延迟主导)就赢。

小数字:用 Hockney 代价模型 $t \approx (\text{步数})\cdot\alpha + (\text{数据量})\cdot\beta$,$\alpha$=每步延迟(几 μs),$\beta$=1/带宽。1024 卡、消息 1 KB:ring 要 $2\times1023\approx2046$ 步,纯延迟就 $2046\alpha$;tree 只要 $\sim2\log_2 1024=20$ 步——**小消息上 tree 快 100 倍**。换成 1 GB 大消息:$\beta$ 项主导,ring 的带宽优势反超。SHARP 再省:NVLink+IB SHARP 把 all-reduce 占用的 SM 从 16 个降到 ≤6 个,算力还给计算。

$$
t_{\text{ring}}\approx 2(N-1)\alpha+\frac{2(N-1)}{N}M\beta,\qquad
t_{\text{tree}}\approx 2\log_2 N\cdot(\alpha+ M\beta)
$$

(注意 tree 每步沿树边搬的是**整段 $M$**,不像 ring 每步只搬 $M/N$;所以 tree 带宽项 $2\log_2 N\cdot M$ 反而 $\ge$ ring 的 $2M$——步数少但每步搬得多,大消息上 tree 吃亏。消息越大 $\beta$ 项主导→偏 ring;越小 $\alpha$ 项主导→偏 tree。NCCL 以 `NCCL_TREE_THRESHOLD` 为界切换。)

![[net-ring-tree-SHARP对比.png]]

![[net-067三算法拓扑.png]]

![[net-067消息大小交叉.png]]

```bash
# 让 NCCL 自动选(默认,推荐):什么都不设
# 调试时强制锁算法对比 busbw:
NCCL_ALGO=Ring  ./all_reduce_perf -b 1K -e 1G -f 2 -g 8   # 大消息这条快
NCCL_ALGO=Tree  ./all_reduce_perf -b 1K -e 1G -f 2 -g 8   # 小消息这条快
NCCL_ALGO=NVLS  ./all_reduce_perf -b 1K -e 1G -f 2 -g 8   # NVLink SHARP(需硬件)

# 看启动日志确认是否启用了 SHARP / NVLS
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING ./all_reduce_perf -b 1G -e 1G -g 8
```

```text
❌ 全程 NCCL_ALGO=Ring 锁死,以为带宽最优就永远最快
   → 几千卡上的小消息(优化器状态、小张量)被 ring 的线性延迟拖垮
❌ 买了带 SHARP 的 Quantum IB 交换机却没开 SHARP
   → all-reduce 仍占 16 个 SM,白白浪费算力和网内归约能力
✅ 默认让 NCCL 自动选;有 SHARP 硬件就确认日志里 NVLS/CollNet 已启用
```

## 面试高频
- **ring 和 tree all-reduce 各自最优在哪一维?** ring 带宽最优(每卡收发 ~2M,与 N 无关)、延迟差($\propto N$);tree 延迟最优($\propto \log N$)、适合小消息和大规模。
- **为什么 ring 是「带宽最优」?** 它=reduce-scatter+all-gather,每步只搬 $M/N$,总搬运量逼近理论下界 $2M$,与卡数几乎无关。
- **什么时候用 tree?** 消息小(延迟主导)或卡极多(几千上万卡),ring 的线性步数变成瓶颈时。
- **SHARP 解决什么?** 把归约下沉到交换机(网内计算):数据只上行一次不来回搬,降延迟、还把 all-reduce 占用的 SM 从 ~16 降到 ≤6,释放算力。
- **NCCL 怎么在 ring/tree 间切?** 按消息大小,`NCCL_TREE_THRESHOLD` 以下用 tree、以上用 ring;通常交给 NCCL 自动选。

## 关键事实
- ring all-reduce 带宽最优但延迟随 $N$ 线性;tree(NCCL 用 double binary tree)延迟 $\propto \log N$,小消息更快(NCCL 官方文档/社区共识,**2025**)。
- NVLink + IB SHARP 把 all-reduce 占用的 SM 从 16+ 降到 ≤6,提升 1000+ GPU 规模的训练效率(NVIDIA NCCL 2.27 博客,**2025**)。
- NCCL 2.27(**2025**)将 SHARP 从 all-reduce 扩展到 all-gather / reduce-scatter,覆盖 NVLink 与 IB 两种 fabric。
- PAT(Parallel Aggregated Trees,arXiv 2506.20252,**2025**):面向超大规模的新 all-gather/reduce-scatter 算法,性能介于 ring 与 tree 之间。
