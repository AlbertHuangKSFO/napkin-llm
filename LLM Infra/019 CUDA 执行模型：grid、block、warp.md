[[019 CUDA 执行模型：grid、block、warp]] 是 GPU 跑 kernel 的并行组织方式:一次 kernel 启动叫 **grid**,grid 切成若干 **block**,block 内线程每 **32 个** 绑成一个 **warp** 锁步执行,warp 里的单个执行流是 **thread**。理解这套层级是写出快 kernel 的前提,也是 [[021 kernel 融合：为何能省带宽|kernel 融合]]、[[024 FlashAttention 1、2：IO 感知精确注意力|FlashAttention]] 等优化的地基。

## 直觉类比
把 GPU 当一座大工厂:**grid** 是整张订单,**block** 是一个车间(被整体分配到一台机器 SM,车间之间不直接对话),**warp** 是车间里 32 个动作完全同步的工人小组——工头喊一个口令,32 人同时做同一动作(SIMT),只是各自手里料不同。某组等原料(等 [[003 GPU 内存层级：HBM、L2、SRAM、寄存器|内存层级]] 里的 HBM)时,工头立刻让另一组干活,机器不空转——这就是 GPU 靠"超额线程"隐藏延迟的核心。

## 小数字例子
处理 100 万元素、每 block 256 线程:
- block 数 = ⌈1,000,000 / 256⌉ = **3907 个 block**,grid 维度 `(3907,1,1)`,block 维度 `(256,1,1)`。
- 每 block 256 / 32 = **8 个 warp**。
- 若 block 设 250(非 32 倍数):最后一个 warp 只有 250−7×32=26 个有效 lane,**6 个 lane 空转**——所以 block 尺寸几乎总选 32 的倍数。
- 一台 SM 假设最多容纳 64 个活跃 warp,而当前每 block 占 8 warp、寄存器只够同时驻 4 个 block,则活跃 32 warp → occupancy = 32/64 = **50%**。

## 原理:SIMT 与 occupancy
warp 内 32 thread 共用一个程序计数器(SIMT);遇到分支,两条路径被**串行**执行、各自 mask 掉无关 lane,即 **warp divergence**,有效吞吐按分支比例下降。

占用率定义:

$$\text{occupancy} = \frac{\text{活跃 warp 数}}{\text{SM 支持的最大活跃 warp 数}}$$

它受三道配额的最小值卡住:每线程**寄存器数**、每 block **SRAM** 用量、SM 的 block/warp 上限。占用率高 → 可调度的 ready warp 多 → 越能用计算掩盖访存延迟;但**并非越高越快**(寄存器够用时低占用也可能更快,即 Volkov 的"低占用高性能")。

## 图
![[cuda-执行层级grid-block-warp.png]]

![[cuda-019occupancy与warp分化.png]]

## 代码:启动配置与边界检查
```cuda
// ✅ block 取 32 的倍数;用 grid-stride 兜住不整除
__global__ void scale(float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = a * x[i];          // 边界检查,避免越界
}
int threads = 256;                        // 32 的倍数
int blocks  = (n + threads - 1) / threads;
scale<<<blocks, threads>>>(x, 2.0f, n);   // <<<grid, block>>>
```
```cuda
// ❌ 反例:block 选 250(非 32 倍数)→ 每 block 末尾 warp 有空 lane
scale<<<blocks, 250>>>(x, 2.0f, n);       // 利用率天然有损
```

## 工厂类比:工人、车间、工厂

四个层级名字绕,但映射到一座工厂就全通了。**从小到大:thread → warp → block → grid。**

- **thread = 工人。** 一个工人干 1 份活,比如算 1 个 token、处理 1 个元素。它有自己的双手(寄存器),是最小的执行流。
- **warp = 32 人小组,锁步干活。** 这是最反直觉、也最关键的一层:**同一时刻,32 个人必须做完全一样的动作**(工头喊一个口令,32 人同手同脚),只是各自手里的料不同。这就是 SIMT(单指令多线程)。为什么是 32?因为这是 NVIDIA 硬件调度的固定粒度,雷打不动。
- **block = 车间。** 一个车间整体被分配到一台机器(SM)上。同车间的工人**共用一张公共工作台**(shared memory,又快又近),还能互相喊话、约定「都做完这步再一起下一步」(`__syncthreads()` 同步)。不同车间之间**不能直接对话**——这也是很多算法要拆成多个 kernel 的原因。
- **grid = 整个工厂。** 一次 kernel 启动就是一张订单,工厂里所有车间一起干这同一批订单。

![[cuda-019类比工厂车间工人.png]]

**关键难点:SIMT 锁步,以及「罚站」。** 既然一个 warp 的 32 人必须同手同脚,那遇到 `if/else` 怎么办?硬件的做法是:**先让走 then 分支的人干活,走 else 的人原地罚站等待;然后反过来,让走 else 的人干活,走 then 的人罚站。** 两条分支被串行执行,总时长翻倍,有效产能减半——这就是 **warp divergence(warp 分化)**。所以分支会让一半人闲着;优化的目标是让分支「按 warp 对齐」(同一组 32 人尽量走同一支),把分化关在组与组之间,而不是组内部。

![[cuda-019SIMT锁步与罚站.png]]

**用具体数字走一遍。** 假设要处理 **1000 个 token**,设每个车间(block)= **256** 个工人(线程):
- 需要车间数 = ⌈1000 / 256⌉ = **4 个 block**。前 3 个车间各满载 256 人(覆盖 768 个 token),**最后一个车间只用到 232 人**(1000 − 768),多出来的工人没活干——所以 kernel 里必须写**边界检查** `if (i < n)`,否则空闲工人会去碰不存在的数据,越界。
- 每个车间的 warp 数 = 256 / 32 = **8 组**。一共 4 车间 × 8 = 32 个 warp 在排队等机器调度。
- 顺带说明:block 尺寸几乎总取 32 的倍数。若设成 250(非 32 倍数),最后一个 warp 只有 26 个有效工人,**6 个 lane 天然空转**,白白浪费。

## 面试高频
- **warp 多少线程?为什么是 32?** 32,是 NVIDIA SIMT 调度的硬件粒度;block 尺寸应取 32 的倍数,否则尾 warp 有空 lane。
- **occupancy 越高越好吗?** 不一定。它只是"能否隐藏延迟"的上限指标;寄存器充裕时低占用反而可能因每线程资源多而更快。
- **block 之间能同步吗?** 不能(除非用 cooperative groups 的网格级同步);常规 block 间只能通过 global memory 通信,这也是很多算法要拆成多个 kernel 的原因。
- **warp divergence 的代价?** 同一 warp 内不同分支被串行执行,理想下应让分支按 warp 对齐。

## 关键事实
- warp = 32 thread,SIMT 锁步执行;block 整体调度到单个 SM(NVIDIA CUDA Programming Guide)。
- occupancy = 活跃 warp / SM 最大活跃 warp,受寄存器、SRAM、block 上限三者最小值约束。
- "低占用也能高性能"由 Vasily Volkov 提出(2010),反直觉但面试常考。
