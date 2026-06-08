[[102 GPU 调度：Kueue、MIG、分时与碎片]]:GPU 是集群里最贵的资源,调度就是「怎么把有限的卡公平、不浪费地分给一堆作业」。这一篇把三件事讲清:**Kueue**(批作业排队 + 配额 + gang scheduling)、**怎么把一张卡分给多个负载**(MIG 硬切 vs 分时软分)、以及绕不开的**GPU 碎片**。它是 [[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|自动扩缩]] 的底座——扩缩想加副本,得先有卡可调度;也呼应 [[060 数据并行与副本扩展|副本扩展]] 与 [[113 容量规划：从 QPS 反推 GPU 数|容量规划]]:规划出的 GPU 数最终要落到调度器手里去摆放。多机大作业还牵涉 [[070 GPU 集群拓扑：fat-tree、rail-optimized、dragonfly|集群拓扑]] 的放置约束。

## ① 类比:停车场调度

把 GPU 集群想成停车场。**Kueue** 是闸机 + 排队系统:每个部门有车位配额(ClusterQueue),车位不够就在门外排队(而非进来乱占);**gang scheduling** 是「旅游大巴要么整车 8 个车位一起停,要么别进」——训练作业的 N 个 worker 必须同时拿到卡,否则半截卡死。**MIG** 是把一个大车位画线切成 7 个小车位(物理隔离,各停各的);**分时** 是同一个车位轮流停不同的车(没隔离,会互相挡)。**碎片** 就是满地零散小车位,凑不出一个能停大巴的连续大位。

## ② 小数字例子:MIG 切片与 gang 的整组性

**MIG 切片。** 一张 A100 80GB 可切成最多 **7 个 1g.10gb** 实例(各 ~10GB 显存 + 1/7 算力),或 3g.40gb × 2、4g.40gb × 1 + 3g.20gb 等组合。一个 1.5B 小模型只要 ~4GB → 用一个 MIG 切片就够,**7 个小模型挤一张卡**,GPU 利用率从 15% 拉到 ~90%。

**gang 的整组性。** 一个 TP=8 的推理副本要 8 张卡同时就绪(见 [[057 张量并行推理：延迟换显存|TP]]);没 gang scheduling 时,调度器可能先给 5 张、另 3 张被别的作业抢走 → 5 张卡白占着等,**死锁 + 浪费**。gang 保证「8 张一起拿到才入场,否则一张不占继续排队」。

## ③ 原理:三层分别管什么

**1. Kueue = 批作业队列层(配额 + gang)。** Kueue 是 K8s 原生的 job 队列,它不自己绑 pod 到节点,而是**决定一个作业何时被"准入"**,再交给 kube-scheduler 落盘。核心能力:
- **ClusterQueue / LocalQueue + 配额**:给团队/项目分 GPU 额度,超额作业排队等额度释放,避免一个团队吃光集群。
- **gang scheduling**:用 Workload 抽象,整组 pod all-or-nothing 准入,杜绝半截占卡。
- **优先级 / 抢占 / 公平分享**:高优作业可抢占低优;空闲配额可借用。
2025 年 Kueue 走向生产成熟,是 AI 批作业的事实标准之一(另有 Volcano、NVIDIA KAI Scheduler)。

**2. 一张卡分给多负载:MIG vs 分时。**
- **MIG(Multi-Instance GPU)= 硬切片**:在 A100 / H100 / H200 / B 系列上,把单卡物理切成多个独立实例,各有独立 SM + 显存 + 缓存路径,**硬隔离、无 noisy neighbor**。适合生产推理、强 SLA。代价:切片粒度固定,可能浪费、且只限这几代卡。
- **分时(time-slicing)= 软分时**:任意 CUDA GPU 都支持,多个负载**轮流**用同一张卡,无隔离 → 有延迟抖动、互相干扰。适合开发、实验、能容忍抖动的批处理。
- **新机制**:K8s **DRA(Dynamic Resource Allocation)** 在 1.34 GA,允许更细粒度地声明/分配 GPU(含分数卡、拓扑感知);NVIDIA **KAI Scheduler** 2025 年开源(Apache 2.0),支持分数 GPU、拓扑感知、分层队列。

**分数卡 vs MIG vs 分时,一个小例子。** 同一张 A100 80GB,塞 4 个各需 ~18GB 的 1.5B 推理副本(共 72GB)。① **DRA/KAI 分数卡**:每个 pod 声明 `gpu: 0.25`(或显式要 20GB 显存),调度器据剩余显存动态拼放,放下第 4 个时若只剩 8GB 就直接拒绝(不超卖)——粒度连续、按需,但软隔离仍可能抢算力。② **MIG**:得先把卡静态切成固定档(如 4×2g.20gb),切片边界写死,18GB 的负载占 20GB 档、浪费 2GB,且切完想改档要重置整卡——硬隔离但不灵活。③ **分时**:4 个 pod 都写 `nvidia.com/gpu: 1` 轮流跑,K8s 以为「4 张卡」其实超卖 1 张,无显存上限保护,撞一起就 OOM。所以**分数卡 = MIG 的灵活 + 比分时安全(带显存配额)**,代价是隔离弱于 MIG。

**3. GPU 碎片(fragmentation)= 核心痛点。** 当集群里散落着一堆小 MIG 切片或被分时占用的零碎卡,新来的大作业(如 8×H100 的 gang)凑不齐连续完整的卡 → 明明总量够,却调度不上。解法:bin-packing 紧凑放置、按作业类型分池(推理池 vs 训练池)、拓扑感知放置(同作业的卡尽量同 NVLink 域/同 rail,见 [[058 TP 的通信：每层 all-reduce 与 NVLink 依赖|TP 的 NVLink 依赖]])、定期重整(reschedule)。

![[orch-GPU调度全景.png]]

![[orch-102共卡与碎片.png]]

## ④ 配置:Kueue 配额 + gang,与 MIG 资源声明

```yaml
# Kueue:给团队定 GPU 配额,作业排队等额度
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata: { name: team-a-gpu }
spec:
  resourceGroups:
    - coveredResources: ["nvidia.com/gpu"]
      flavors:
        - name: h100
          resources:
            - name: "nvidia.com/gpu"
              nominalQuota: 16          # 该团队最多 16 张 H100,超额排队
---
# 作业声明属于某队列,Kueue 做 gang 准入(整组 all-or-nothing)
apiVersion: batch/v1
kind: Job
metadata:
  labels: { kueue.x-k8s.io/queue-name: team-a-local }
spec:
  parallelism: 8                        # TP=8:8 个 pod 必须一起入场
  template:
    spec:
      containers:
        - name: vllm
          resources:
            limits: { nvidia.com/gpu: 1 }
```

```yaml
# MIG:pod 请求一个 1g.10gb 切片而非整卡(硬隔离、多模型共卡)
resources:
  limits:
    nvidia.com/mig-1g.10gb: 1           # 拿一个 MIG 切片;✅ 强隔离
# 对照 ❌ 分时:多 pod 共享 nvidia.com/gpu: 1,无隔离、互扰抖动
```

❌ 反模式:① 不用队列/配额,所有作业裸抢 `nvidia.com/gpu`——一个大作业拿一半、卡死另一半,或一个团队吃光集群。② 给强 SLA 的生产推理用分时共卡——被同卡的邻居拖出 P99 抖动。③ 不管碎片,小推理把卡切得稀碎,大训练永远 gang 不上来。

✅ 正解:Kueue 做**配额 + gang scheduling**,生产推理用 **MIG 硬隔离**、开发实验用**分时**,并按作业类型分池 + 拓扑感知放置来控碎片;大集群上 DRA / KAI Scheduler 做更细的分数卡与拓扑调度。

## 面试高频

- **「Kueue 解决什么,和 kube-scheduler 什么关系?」** Kueue 是批作业队列层,管配额、排队、gang 准入、优先级抢占;它决定作业「何时被准入」,真正绑 pod 到节点交给 kube-scheduler。
- **「gang scheduling 为什么对 LLM 必要?」** TP/PP 多卡作业要 N 张卡同时就绪,否则半截占卡死锁、浪费;gang 保证 all-or-nothing 入场。
- **「MIG 和 time-slicing 区别、各用在哪?」** MIG 是硬切片(独立 SM+显存,硬隔离,限 A100/H100 等),用于生产强 SLA;time-slicing 是软分时(轮流用、无隔离、有抖动,任意卡),用于开发/实验/批处理。
- **「GPU 碎片是什么,怎么缓解?」** 集群里散落零碎切片/分时卡,总量够却凑不出大作业要的连续完整卡。缓解:bin-packing、分池、拓扑感知放置、定期重整。
- **「2025/2026 有什么新调度机制?」** K8s DRA(1.34 GA)细粒度声明 GPU、分数卡、拓扑感知;NVIDIA KAI Scheduler 开源,支持分数 GPU + 分层队列。

## 关键事实

- **Kueue**:K8s 原生 job 队列,ClusterQueue 配额 + gang scheduling + 抢占;2025 走向生产成熟。同类:Volcano、NVIDIA KAI Scheduler(2025 开源,Apache 2.0)。
- **MIG**:仅 A100 / H100 / H200 / B 系列支持;A100 80GB 最多切 7 个 1g.10gb;硬隔离、无 noisy neighbor。
- **time-slicing**:任意 CUDA GPU 可用;轮流共享、无隔离、有延迟抖动。
- **DRA(Dynamic Resource Allocation)**:K8s 1.34 GA,细粒度/拓扑感知 GPU 分配。
- **GPU 碎片**:零散 MIG/分时切片使大 gang 作业无法调度;靠 bin-packing + 分池 + 拓扑感知放置缓解。
