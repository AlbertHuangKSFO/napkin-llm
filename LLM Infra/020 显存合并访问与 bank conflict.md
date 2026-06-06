[[020 显存合并访问与 bank conflict]] 讲的是两类"访存模式决定速度"的现象:**合并访问(coalesced access)** 关乎 global memory(HBM),**bank conflict** 关乎 shared memory(SRAM)。对绝大多数 memory-bound 的 kernel,真正决定快慢的不是算了多少 FLOP,而是这两件事是否做对——这也是 [[019 CUDA 执行模型：grid、block、warp|CUDA 执行模型]] 与 [[003 GPU 内存层级：HBM、L2、SRAM、寄存器|内存层级]] 落到性能上的关键一环。

## 直觉类比
**合并访问**像快递车装货:一个 warp 的 32 个线程要取数据,如果它们要的地址连成一片,快递车(一笔内存事务)一趟拉满;如果地址东一个西一个,得跑好几趟,带宽白白浪费。
**bank conflict** 像 32 个收银台:shared memory 切成 32 条 bank,一个 warp 同一拍最多能同时刷 32 个不同 bank。若两个线程挤同一个 bank,只能排队——一拍变两拍。

## 小数字例子
- 一个 warp 读 32 个 `float`(4B),合并时 = 1 笔 **128B** 事务,带宽利用 ≈100%。
- 若按 stride=32 跨步读(每个线程隔 128B 取一个),同样 32 个 float 可能拆成 **32 笔事务**,有效带宽降到约 1/32。
- shared memory 声明 `__shared__ float tile[32][32]`,按列访问 `tile[t][col]` 时 32 个线程全落在同一 bank → **32-way conflict**,串行 32 拍;改成 `tile[32][33]`(加 1 列 padding)后列地址错开,**冲突消失**。

## 原理:事务粒度与 bank 映射
global memory 以固定大小段(如 32B/128B)为单位读取。warp 的 32 个 4B 请求落入的不同段数 = 事务数:

$$\text{有效带宽} \approx \frac{\text{真正需要的字节}}{\text{实际搬运的段总字节}}$$

shared memory bank 号:`bank = (地址 / 4) mod 32`。一个 warp 内若 $k$ 个线程命中同一 bank(且非同址广播),则该访问串行化为 $k$ 拍,吞吐降为 $1/k$。**例外**:全 warp 读同一地址触发广播,仅 1 拍。

## 图
![[cuda-合并访问与bank-conflict.svg]]

下面把两类访存的"罚款"逐笔算给你看——合并与否差 32 笔事务,bank conflict 差 32 拍:

![[cuda-020合并访问事务数手算.svg]]

## 代码:转置 kernel 的访存修复
```cuda
// ❌ 朴素转置:写 global 时跨步 → 非合并写,带宽腰斩
out[x*height + y] = in[y*width + x];      // 相邻线程写的地址相隔 height

// ✅ 经 shared memory 中转 + padding 消 bank conflict
__shared__ float tile[32][33];            // +1 列 padding,关键
tile[threadIdx.y][threadIdx.x] = in[...]; // 合并读
__syncthreads();
out[...] = tile[threadIdx.x][threadIdx.y];// 合并写,且转置取列无 bank conflict
```

## 面试高频
- **什么是合并访问?** 一个 warp 的访问落入连续地址段,合成尽量少的内存事务;反例是大 stride / 随机访问。
- **bank conflict 怎么消?** 给 shared tile 加 padding(`[N][N+1]`)错开 bank 号;或重排数据布局;广播访问不算冲突。
- **为什么访存模式比算力更决定速度?** LLM 推理里大量算子(softmax、LayerNorm、逐元素、GEMV)是 memory-bound,瓶颈在 HBM 带宽,合并与否直接决定有效带宽。
- **AoS vs SoA?** SoA(结构体数组拆成数组的结构体)让同字段连续,利于合并访问。

## 关键事实
- warp 连续地址访问可合成一笔 128B 事务,跨步/随机访问拆成多笔,有效带宽骤降(NVIDIA CUDA Best Practices Guide)。
- shared memory = 32 个 bank,`bank=(addr/4) mod 32`;$k$-way conflict 串行成 $k$ 拍,广播例外。
- padding 到 `[32][33]` 是消列向 bank conflict 的经典手法。
