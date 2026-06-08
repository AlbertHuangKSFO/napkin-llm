[[10 SVD 奇异值分解|SVD]](奇异值分解)把**任意**矩阵 $A$ 拆成 $A=U\Sigma V^T$,几何上是"旋转 → 沿坐标轴缩放 → 再旋转"三步,其中缩放倍数 $\sigma_i$ 叫**奇异值**。它是 [[09 特征值与特征向量|特征分解]] 对非方阵的推广,是 [[11 主成分分析 PCA|PCA]]、低秩压缩、推荐系统、降维 [[04 Embedding 与向量数据库|Embedding]] 的统一数学引擎。

## 直觉

特征分解只对方阵、还得能对角化才行;现实里的数据矩阵大多是长方形(行=样本、列=特征),压根没有特征值。SVD 是"万能版":**任何形状的矩阵都能分解**。

它的几何故事最美:把一个**单位圆**喂给任意 $2\times2$ 矩阵,输出永远是一个**椭圆**。SVD 说,从圆到椭圆这件事,一定能拆成三步:

1. $V^T$:先**旋转**一下(把圆转个角度,还是圆);
2. $\Sigma$:沿坐标轴**拉伸**成椭圆(短轴/长轴 = 奇异值 $\sigma_2,\sigma_1$);
3. $U$:再**旋转**到最终朝向。

旋转不改变形状(正交矩阵),所有"形变"信息都装在中间那个对角的 $\Sigma$ 里——而 $\sigma_1\ge\sigma_2\ge\cdots\ge0$ 告诉你"哪个方向被拉得最狠、最重要"。

## 例子

**手算一个 $2\times2$ 的奇异值**。取 $A=\begin{bmatrix}3&0\\4&5\end{bmatrix}$。奇异值是 $A^TA$ 的特征值开根号。

$$A^TA=\begin{bmatrix}3&4\\0&5\end{bmatrix}\begin{bmatrix}3&0\\4&5\end{bmatrix}=\begin{bmatrix}25&20\\20&25\end{bmatrix}$$

解 $\det(A^TA-\lambda I)=0$:$(25-\lambda)^2-400=0\Rightarrow 25-\lambda=\pm20\Rightarrow\lambda=45\ \text{或}\ 5$。

于是奇异值 $\sigma_1=\sqrt{45}=3\sqrt5\approx6.71$,$\sigma_2=\sqrt5\approx2.24$。

**交叉验证**:奇异值之积 $\sigma_1\sigma_2=\sqrt{45\cdot5}=\sqrt{225}=15=|\det A|=|3\cdot5-0\cdot4|=15$。对上了(这对应 [[08 行列式与空间缩放|行列式]] = 面积缩放,而 SVD 把面积缩放拆成各轴 $\sigma$ 之积)。

**凑齐 $V$ 与 $U$**。$V$ 的列是 $A^TA$ 的单位特征向量。$\lambda=45$:$(25-45)v_1+20v_2=0\Rightarrow v_1=v_2$,得 $\mathbf{v}_1=\tfrac{1}{\sqrt2}(1,1)^T$;$\lambda=5$:$v_1=-v_2$,得 $\mathbf{v}_2=\tfrac{1}{\sqrt2}(1,-1)^T$。$U$ 不用再解 $AA^T$,直接用 $\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$:
$$A\mathbf{v}_1=\tfrac{1}{\sqrt2}(3,9)^T,\quad \mathbf{u}_1=\frac{A\mathbf{v}_1}{3\sqrt5}=\frac{1}{\sqrt{10}}(1,3)^T;$$
$$A\mathbf{v}_2=\tfrac{1}{\sqrt2}(3,-1)^T,\quad \mathbf{u}_2=\frac{A\mathbf{v}_2}{\sqrt5}=\frac{1}{\sqrt{10}}(3,-1)^T.$$
合起来 $A=U\Sigma V^T$,其中
$$U=\frac{1}{\sqrt{10}}\begin{bmatrix}1&3\\3&-1\end{bmatrix},\ \Sigma=\begin{bmatrix}3\sqrt5&0\\0&\sqrt5\end{bmatrix},\ V=\frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}.$$
关键是 **$U$ 由 $\mathbf{u}_i=A\mathbf{v}_i/\sigma_i$ 一步算出**,省掉解第二个特征问题,也自动保证 $U,V$ 的奇异向量按 $\sigma$ 配对。

![[la-SVD三步几何.png]]

## 原理

**定义**:任意 $m\times n$ 实矩阵都可写成

$$A=U\Sigma V^T$$

- $U$:$m\times m$ 正交矩阵($U^TU=I$),列叫**左奇异向量**;
- $V$:$n\times n$ 正交矩阵,列叫**右奇异向量**;
- $\Sigma$:$m\times n$ "对角"矩阵,对角元 $\sigma_1\ge\sigma_2\ge\cdots\ge0$ 是**奇异值**。

**和特征分解的精确关系**(面试常考):

$$A^TA=V\Sigma^2 V^T,\qquad AA^T=U\Sigma^2 U^T$$

即:右奇异向量 $V$ = $A^TA$ 的特征向量,左奇异向量 $U$ = $AA^T$ 的特征向量,而 $\sigma_i^2$ = 这两个(对称半正定)矩阵的特征值。所以 $\sigma_i=\sqrt{\lambda_i(A^TA)}\ge0$ 恒非负——这就是为什么奇异值永远 $\ge0$,而特征值可正可负可复。

**完整版 vs 经济版(thin SVD)**。$m\times n$、$m>n$ 时,完整 $U$ 是 $m\times m$ 很浪费;**经济版**只保留前 $\min(m,n)$ 列:$U$ 取 $m\times n$、$\Sigma$ 取 $n\times n$、$V$ 取 $n\times n$,重建结果一样。numpy 用 `full_matrices=False` 取经济版——大矩阵务必用它,否则 $U$ 撑爆内存。

**奇异值 = 各方向的拉伸倍数,与范数/条件数挂钩**:

- **谱范数**(矩阵的"最大拉伸"):$\|A\|_2=\sigma_1$(最大奇异值)。
- **Frobenius 范数**:$\|A\|_F=\sqrt{\sum_i\sigma_i^2}=\sqrt{\sum_{ij}A_{ij}^2}$。
- **条件数** $\kappa(A)=\sigma_1/\sigma_r$(最大/最小奇异值之比):衡量矩阵"病态"程度。$\kappa$ 大 = 又长又窄的椭圆 = 解线性方程、训练时数值不稳、梯度下降在"峡谷"里震荡(见 [[17 雅可比与 Hessian|条件数]])。

**外积展开(SVD 的灵魂)**:

$$A=\sum_{i=1}^{r}\sigma_i\, u_i v_i^T,\qquad r=\text{rank}(A)$$

把矩阵写成 $r$ 个"秩 1 块"的加权和,权重就是奇异值。$\sigma$ 大的块重要,小的块是细节/噪声。每个 $u_iv_i^\top$ 是一个秩 1 矩阵(见 [[03 点积、范数与相似度|外积]]),所以 SVD 把任意矩阵拆成 $r$ 个秩 1 积木的叠加。

**低秩近似 = 压缩(Eckart–Young 定理)**。只保留前 $k$ 个最大奇异值:

$$A_k=\sum_{i=1}^{k}\sigma_i u_i v_i^T$$

定理保证:在所有秩 $\le k$ 的矩阵里,$A_k$ 是对 $A$ 误差最小的最佳近似(Frobenius / 谱范数下都最优),丢掉的误差正好由被砍掉的 $\sigma_{k+1},\sigma_{k+2},\dots$ 决定:

$$\|A-A_k\|_2=\sigma_{k+1},\qquad \|A-A_k\|_F=\sqrt{\sigma_{k+1}^2+\cdots+\sigma_r^2}$$

存储从 $m\times n$ 降到 $k(m+n+1)$。**怎么选 $k$**:看奇异值衰减,保到累计能量 $\frac{\sum_{i\le k}\sigma_i^2}{\sum_i\sigma_i^2}$ 达 90–99%,或看 $\sigma$ 谱的"肘部"(和 [[11 主成分分析 PCA|PCA]] 选维度同理)。

**LoRA 的数学背景**:大模型微调时,权重更新 $\Delta W$ 经验上是**低秩**的,于是用 $\Delta W=BA$($B$ 是 $d\times r$、$A$ 是 $r\times d$,$r\ll d$)只训这两个瘦矩阵——本质就是"假设更新矩阵秩低、用秩 $r$ 近似",和截断 SVD 一脉相承(Hu et al., LoRA, 2021)。

![[la-SVD低秩近似.png]]

应用:图像压缩、潜在语义分析(LSA)、协同过滤推荐、降维得到的低维 [[04 Embedding 与向量数据库|Embedding]],本质都是"截断 SVD 保留主要方向"。

## 代码

```python
import numpy as np

A = np.array([[3., 0.],
              [4., 5.]])
U, S, Vt = np.linalg.svd(A)          # 注意返回的是 Vt = V^T
print("奇异值:", S)                   # [6.708.. 2.236..] = [3√5, √5]

# 验证重建 A = U Σ V^T
Sigma = np.diag(S)
print("重建:\n", U @ Sigma @ Vt)      # ≈ 原 A

# 手算对照:σ 应是 sqrt(eig(A^T A))
eigvals = np.linalg.eigvalsh(A.T @ A)         # [5, 45]
print("sqrt(eig(A^T A)):", np.sqrt(eigvals[::-1]))  # [6.708.. 2.236..]

# 低秩近似:一个低秩矩阵只留前 1 个奇异值
B = np.array([[1., 2., 3.],
              [2., 4., 6.],         # = 2×第一行
              [3., 6., 9.]])        # = 3×第一行,真实 rank=1
Ub, Sb, Vtb = np.linalg.svd(B)
print("B 的奇异值:", Sb.round(3))    # 只有第一个非零 → rank 1
B1 = Sb[0] * np.outer(Ub[:, 0], Vtb[0])
print("秩1 重建 ≈ B:\n", B1.round(3))  # 几乎等于 B
```

手算对照:`S=[3√5, √5]≈[6.708, 2.236]` 与手算奇异值一致;`sqrt(eig(A^T A))` 给出同样的值,印证 $\sigma_i=\sqrt{\lambda_i(A^TA)}$;`B` 的第 2、3 个奇异值约为 0,说明它真实秩为 1。

```python
# 经济版 SVD + 范数/条件数 + 截断误差验证
A = np.random.randn(6, 4)
U, S, Vt = np.linalg.svd(A, full_matrices=False)  # 经济版,U:6x4
print("谱范数 σ1:", S[0].round(4), np.linalg.norm(A, 2).round(4))  # 相等
print("Frobenius:", np.sqrt((S**2).sum()).round(4), np.linalg.norm(A,'fro').round(4))
print("条件数 σ1/σr:", (S[0]/S[-1]).round(3), np.linalg.cond(A).round(3))

# Eckart-Young:秩 k 截断误差 == σ_{k+1}(谱范数)
k = 2
Ak = (U[:, :k] * S[:k]) @ Vt[:k]
print("‖A-Ak‖2 =", np.linalg.norm(A - Ak, 2).round(4), " σ3 =", S[k].round(4))  # 相等
print("‖A-Ak‖F =", np.linalg.norm(A-Ak,'fro').round(4),
      " sqrt(σ3²+σ4²) =", np.sqrt(S[k:]**2 @ np.ones(len(S)-k)).round(4))
```

## 面试高频

- **"SVD 和特征分解的区别?"** 标准答:特征分解只对(可对角化的)方阵;SVD 对任意矩阵都成立。$U,V$ 正交、$\sigma\ge0$;特征值可负/复。对称半正定矩阵两者一致($\sigma_i=\lambda_i$)。几乎必考。
- **"奇异值和特征值什么关系?"** $\sigma_i=\sqrt{\lambda_i(A^TA)}$,$V$ 是 $A^TA$ 的特征向量,$U$ 是 $AA^T$ 的特征向量。
- **"SVD 怎么做降维/压缩?"** 截断 SVD 保留前 $k$ 个奇异值,Eckart–Young 保证是最优低秩近似。能点出"最优"二字加分。
- **"SVD 和 PCA 关系?"** 对**中心化**后的数据矩阵做 SVD,右奇异向量就是主成分方向,$\sigma_i^2$ 正比于方差。工程上 PCA 常用 SVD 实现(数值更稳,不必显式算协方差)。见 [[11 主成分分析 PCA|PCA]]。
- **陷阱**:numpy 的 `svd` 返回的是 $V^T$ 不是 $V$;且 `S` 是一维向量不是对角阵,要 `np.diag` 复原。这俩坑很多人栽。
- **"奇异值和矩阵范数/条件数的关系?"** $\|A\|_2=\sigma_1$,$\|A\|_F=\sqrt{\sum\sigma_i^2}$,条件数 $\kappa=\sigma_1/\sigma_r$。条件数大 = 病态 = 训练/求解不稳。
- **"完整版和经济版 SVD?"** 大矩阵用 `full_matrices=False`(经济版),否则 $U$ 是 $m\times m$ 撑爆内存;重建结果相同。
- **"LoRA 和 SVD 什么关系?"** LoRA 假设权重更新低秩,用 $BA$(秩 $r$)近似 $\Delta W$,思想等同截断 SVD。
- **"SVD 复杂度?"** 稠密 $m\times n$ 约 $O(mn\min(m,n))$;只要前 $k$ 个奇异值时用随机化 SVD / Lanczos 更快。

## 关键事实

- "每个矩阵都有奇异值分解 $A=U\Sigma V^T$,$U,V$ 正交,$\Sigma$ 对角非负且 $\sigma_1\ge\cdots\ge0$",见 Strang《Introduction to Linear Algebra》(第 6 版,2023)第 7 章,及 Trefethen & Bau《Numerical Linear Algebra》(1997)Lecture 4–5。
- 最优低秩近似由 Eckart 与 Young 于 1936 年证明(Eckart, C. & Young, G., "The approximation of one matrix by another of lower rank", *Psychometrika*, 1936)。
- 奇异值与 $A^TA$ 特征值的关系、$\sigma_i\ge0$ 恒成立,见 Golub & Van Loan《Matrix Computations》(第 4 版,2013)§2.4。
- 谱范数/Frobenius 范数/条件数与奇异值的关系见 Trefethen & Bau《Numerical Linear Algebra》(1997)Lecture 4–5。
- 低秩适配 LoRA(假设权重更新低秩,用秩 $r$ 的 $BA$ 近似)见 Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"(ICLR, 2022;arXiv 2021)。
