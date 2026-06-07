[[096 Kubernetes 上部署 LLM：基本盘]]讲的是:在 K8s 上把一个 LLM 引擎跑成生产服务所需的最小一套对象——Pod / Deployment / Service、GPU 资源请求、device plugin、模型权重挂载、就绪/启动探针——以及为什么 LLM 部署跟普通无状态 Web 服务有本质不同。它是 [[095 从单进程到集群：服务架构演进|架构演进]]第 ②③ 阶段的落地基础,也是 [[097 KServe：模型生命周期与 LLMInferenceService|KServe]] / [[098 llm-d：K8s 原生分布式推理|llm-d]] 这些上层抽象底下真正在管的东西。

## 类比
普通无状态 Web Pod 像便利店店员:换班(重启)只要几秒,谁来都一样,随时能扩。LLM Pod 像一位需要"开机仪式"的米其林主厨:上岗前要把几十上百 GB 的食材(权重)搬进后厨(VRAM)、预热烤箱(编译 CUDA graph)、摆好备餐台([[030 PagedAttention 深入：KV 当虚拟内存|预分配 KV-Cache]]),整个过程要几分钟;一旦上岗会一直占着整间厨房(整张 GPU),你不能像换便利店店员那样随手重启他。探针就是"别在主厨还没准备好时就放客人进来"的门禁。

## 小数字例子
部署 Llama-3-70B(BF16),约 140GB 权重:
- 显存:单卡 80GB 放不下 → 需 [[057 张量并行推理：延迟换显存|TP]]=2,请求 `nvidia.com/gpu: 2`,且两卡最好同节点走 [[008 片内互联：NVLink、NVSwitch、NVLink-C2C|NVLink]]。
- 启动:从网络存储拉 140GB 权重 ~ 几分钟;加载进 VRAM + 编译 + 预分配 KV 再加 1–3 分钟。
- 探针:readiness `initialDelaySeconds: 120`、startupProbe `failureThreshold: 60 × periodSeconds: 10`(10 分钟预算)。若用默认 Web 探针(几秒),Pod 会在加载途中被反复杀掉,永远 CrashLoop。

![[orch-096探针时间线.png]]

## 原理:最小对象与节点层
**工作负载三件套**:
- **Pod**:跑引擎容器 + initContainer(预拉权重到 PVC)。
- **Deployment**:管副本数、滚动更新、把"就绪"的 Pod 才接进 Service。
- **Service**:稳定 VIP/DNS,对副本做四层均衡(对 LLM 通常还要叠 [[100 推理网关与智能路由(cache-aware)|cache-aware 网关]],见 095③)。

**GPU 怎么变成可调度资源**(普通 CPU/内存是内建的,GPU 不是):
1. NFD + GPU Operator 给节点打 GPU 型号/显存标签;
2. **NVIDIA device plugin**(DaemonSet)向 kubelet 汇报 GPU,暴露为扩展资源 `nvidia.com/gpu`;
3. kube-scheduler 按 Pod 的 GPU `request/limit` 选有空卡的节点;
4. DCGM exporter 把 GPU 利用率/显存喂给 Prometheus 供监控与 [[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|扩缩]]。
注意:GPU 不可超卖,`request` 必须等于 `limit`,一块卡默认整卡独占(MIG/MPS 才能切分)。

**模型权重挂载**:几十 GB 权重不该烤进镜像。常用 initContainer 预下载到 PVC(或 hostPath/对象存储缓存),主容器只读挂载,加快重启与多副本共享。

**为何不同于无状态服务**:启动以分钟计、权重巨大、显存常驻、整卡独占、重启代价高、且天然有可复用状态([[015 KV-Cache 的显存账(逐层手算)|KV-Cache]])。这逼出"长探针预算 + 权重预热 + 谨慎扩缩"的部署范式。

![[orch-K8s部署LLM.png]]

## 代码:GPU LLM Deployment + Service
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: llama3-70b }
spec:
  replicas: 2
  selector: { matchLabels: { app: llama3-70b } }
  template:
    metadata: { labels: { app: llama3-70b } }
    spec:
      initContainers:
      - name: fetch-weights                 # 权重不进镜像,预拉到 PVC
        image: amazon/aws-cli:latest
        command: ["sh","-c","aws s3 sync s3://models/llama3-70b /models"]
        volumeMounts: [{ name: model, mountPath: /models }]
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args: ["--model","/models","--tensor-parallel-size","2"]
        resources:
          limits:   { nvidia.com/gpu: 2 }   # GPU:request 必须 = limit,整卡独占
        volumeMounts: [{ name: model, mountPath: /models, readOnly: true }]
        startupProbe:                        # 给加载留 10 分钟,期间不判失败
          httpGet: { path: /health, port: 8000 }
          failureThreshold: 60
          periodSeconds: 10
        readinessProbe:                      # 就绪才进 Service 的 LB
          httpGet: { path: /health, port: 8000 }
          initialDelaySeconds: 120
          periodSeconds: 10
      volumes:
      - name: model
        persistentVolumeClaim: { claimName: llama3-70b-pvc }
---
apiVersion: v1
kind: Service
metadata: { name: llama3-70b }
spec:
  selector: { app: llama3-70b }
  ports: [{ port: 80, targetPort: 8000 }]
```
```yaml
# ❌ 反例:把 LLM 当无状态 Web 服务
# resources: { limits: { cpu: "4" } }      # 漏了 GPU,Pod 永远调度不到卡
# readinessProbe: { initialDelaySeconds: 5 } # 几秒就判就绪/失败 → CrashLoop
# COPY weights/ into image                   # 140GB 烤进镜像,拉取/重启灾难
```

## 面试高频
- **K8s 上部署 LLM 跟无状态服务最大的区别?** 分钟级启动、巨大权重、显存常驻、整卡独占、重启贵;因此要长探针预算、权重预热、谨慎扩缩,且有可复用 KV 状态。
- **GPU 怎么成为可调度资源?** device plugin(DaemonSet)向 kubelet 注册并暴露 `nvidia.com/gpu`,scheduler 按 request 调度;GPU 不可超卖,request=limit。
- **为什么权重不放镜像里?** 镜像几十上百 GB → 拉取慢、节点磁盘爆、滚更慢;用 initContainer/PVC 预下载并多副本共享。
- **startup vs readiness vs liveness?** startup 兜"首次加载"窗口避免被早杀;readiness 控流量进出;liveness 阈值要更宽防误杀正在长加载的 Pod。
- **单卡放不下 70B 怎么办?** 请求多卡 + TP,且尽量同节点 NVLink;超单节点则进入分布式编排(098/099)。

## 关键事实
- `nvidia.com/gpu` 是扩展资源,由 NVIDIA device plugin / GPU Operator 暴露;K8s 本身不认识 GPU。GPU 资源不可超卖,request 必须等于 limit。
- 现代栈(2025–2026)标准组件:NFD + GPU Operator(含 device plugin、GFD)+ DCGM exporter;K8s 1.35 引入 In-Place Pod Resource Update、Node Topology 等利于 GPU 调度的能力。
- LLM 容器多阶段启动(权重落盘→载入 VRAM→编译 CUDA graph→预分配 KV)常以分钟计,探针需分钟级预算,这是与普通服务最实操性的差异。
- 生产里几乎不用裸 Deployment 手搓这些,而是交给 KServe/llm-d/Ray Serve 等上层封装——但它们底下管的正是这套对象。
