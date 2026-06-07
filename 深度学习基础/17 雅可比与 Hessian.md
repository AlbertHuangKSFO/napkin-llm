[[17 雅可比与 Hessian|雅可比与 Hessian]]是 [[14 导数与偏导数|导数]] 在多变量下的两层升级:**雅可比**是"向量对向量"的全部一阶偏导排成的矩阵(描述局部线性变换);**Hessian**是"标量对向量"的全部二阶偏导排成的方阵(描述曲面的弯曲)。雅可比是 [[16 链式法则|链式法则]] 与 [[20 反向传播的数学推导|反向传播]] 的搬运工,Hessian 则刻画损失曲面的形状,决定优化好不好走。

## 直觉

先把三个东西摆在一起,核心区别只有两点:**一阶还是二阶,输出是标量还是向量**。

- [[15 梯度、方向导数与等高线|梯度]]:**标量** $f$ 对向量 $x$ 的一阶导 → 一个向量。
- **雅可比** $J$:**向量** $y$ 对向量 $x$ 的一阶导 → 一个矩阵(每行是一个输出分量的梯度)。它告诉你"输入动一个小向量 $\Delta x$,输出近似动 $J\Delta x$"——也就是**该点把函数当成线性变换的那个矩阵**(见 [[05 矩阵与线性变换(可视化)|线性变换]])。
- **Hessian** $H$:**标量** $f$ 的**二阶**导 → 一个对称方阵。一阶导(梯度)说"坡往哪倾",二阶导(Hessian)说"坡在变陡还是变缓、碗是朝上还是朝下"。

生活类比:开车。位置→速度是一阶(像梯度/雅可比),速度→加速度是二阶(像 Hessian)。只知道速度你能猜下一秒位置;还知道加速度你能猜得更准——这正是牛顿法用 Hessian 比只用梯度收敛快的原因。

Hessian 的几何意义:它的特征值(见 [[09 特征值与特征向量|特征值]])描述各方向的弯曲。全正 → 碗朝上(局部极小);全负 → 碗朝下(极大);有正有负 → 鞍点(一头上翘一头下凹)。

## 例子

**雅可比**。设 $y=f(x)$,$y_1=x_1^2+x_2$,$y_2=3x_1+\sin x_2$。逐个偏导:

$$J=\frac{\partial y}{\partial x}=\begin{bmatrix}\dfrac{\partial y_1}{\partial x_1}&\dfrac{\partial y_1}{\partial x_2}\\[4pt]\dfrac{\partial y_2}{\partial x_1}&\dfrac{\partial y_2}{\partial x_2}\end{bmatrix}=\begin{bmatrix}2x_1&1\\3&\cos x_2\end{bmatrix}$$

在点 $(1,0)$:$J=\begin{bmatrix}2&1\\3&1\end{bmatrix}$。含义:输入扰动 $\Delta x$,输出近似变 $J\Delta x$。

**Hessian**。设标量 $f(x,y)=x^2+3xy+y^2$。先一阶(梯度)$\nabla f=(2x+3y,\ 3x+2y)$,再对每个分量求一次偏导:

$$H=\begin{bmatrix}\dfrac{\partial^2 f}{\partial x^2}&\dfrac{\partial^2 f}{\partial x\partial y}\\[4pt]\dfrac{\partial^2 f}{\partial y\partial x}&\dfrac{\partial^2 f}{\partial y^2}\end{bmatrix}=\begin{bmatrix}2&3\\3&2\end{bmatrix}$$

注意 $H$ 对称($\partial^2 f/\partial x\partial y=\partial^2 f/\partial y\partial x=3$),这是**克莱罗定理**保证的。它的特征值是 $2\pm3$,即 $5$ 和 $-1$,一正一负 → 这个点是**鞍点**(不是极小也不是极大)。

**再算一个"极小值"的 Hessian**。$f(x,y)=x^2+y^2$(碗朝上),$\nabla f=(2x,2y)$,$H=\begin{bmatrix}2&0\\0&2\end{bmatrix}$。特征值 $2,2$ 全正 → **正定** → 该临界点(原点)是局部极小。**$2\times2$ 不算特征值的快速判据**:看 $\det H$ 和 $H_{11}$——$\det H>0$ 且 $H_{11}>0$ ⇒ 正定(极小);$\det H>0$ 且 $H_{11}<0$ ⇒ 负定(极大);$\det H<0$ ⇒ 鞍点。这里 $\det=4>0,H_{11}=2>0$ ⇒ 极小。前面鞍点例 $\det\begin{bmatrix}2&3\\3&2\end{bmatrix}=4-9=-5<0$ ⇒ 鞍点,与特征值结论一致。

**牛顿法一步(用 Hessian 加速)**。梯度下降 $\mathbf{x}\leftarrow\mathbf{x}-\eta\nabla f$ 只用一阶;牛顿法用二阶:$\mathbf{x}\leftarrow\mathbf{x}-H^{-1}\nabla f$。对二次函数**一步到位**(因为二次泰勒展开是精确的)。例 $f=x^2+y^2$ 在 $(3,4)$:$\nabla f=(6,8)$,$H^{-1}=\tfrac12 I$,$\mathbf{x}_{\text{new}}=(3,4)-\tfrac12(6,8)=(0,0)$ —— 直达最小值。但 $H^{-1}$ 在百万参数下算不动,所以深度学习不用纯牛顿法。

![[calc-雅可比Hessian形状.png]]

## 原理

**雅可比矩阵**。$f:\mathbb R^n\to\mathbb R^m$,雅可比是 $m\times n$ 矩阵:

$$J\in\mathbb R^{m\times n},\qquad J_{ij}=\frac{\partial y_i}{\partial x_j}$$

第 $i$ 行就是输出分量 $y_i$ 的梯度。一阶泰勒近似:$f(x+\Delta x)\approx f(x)+J\,\Delta x$——所以雅可比是"函数在该点的最佳线性近似矩阵"。**梯度是雅可比在 $m=1$ 时的(转置)特例**。

链式法则的复合就是雅可比相乘:$z=g(y),y=f(x)$,则 $J_{z\to x}=J_{z\to y}\,J_{y\to x}$。这正是 [[16 链式法则|链式法则]] 的矩阵版,也是 [[20 反向传播的数学推导|反向传播]] 逐层传梯度($J^T$ 左乘上游梯度)的依据。

**Hessian 矩阵**。$f:\mathbb R^n\to\mathbb R$(标量),Hessian 是 $n\times n$ 方阵:

$$H\in\mathbb R^{n\times n},\qquad H_{ij}=\frac{\partial^2 f}{\partial x_i\partial x_j}$$

二阶泰勒近似:$f(x+\Delta)\approx f(x)+\nabla f^T\Delta+\tfrac12\Delta^T H\Delta$。二次项里的 $H$ 决定曲面弯曲。

**克莱罗(Schwarz)定理**:若二阶偏导连续,则 $\partial^2 f/\partial x_i\partial x_j=\partial^2 f/\partial x_j\partial x_i$,**Hessian 对称**。

**用特征值判极值**(二阶判据):在梯度为 0 的临界点,看 $H$ 的特征值——

- 全 $>0$(正定):局部**极小**(碗朝上)。
- 全 $<0$(负定):局部**极大**。
- 有正有负(不定):**鞍点**——高维损失曲面上鞍点远多于局部极小,这是深度学习优化的关键事实。

![[calc-雅可比Hessian形状.png]]

**雅可比行列式 = 体积/密度的缩放因子**。当 $m=n$(方阵雅可比),$|\det J|$ 是该点局部体积的缩放倍数(见 [[08 行列式与空间缩放|行列式]]);概率密度做变量替换时要除以 $|\det J|$。这是归一化流(normalizing flow)的核心——设计 $\det J$ 易算的变换(如三角雅可比),就能精确算变换后的密度。

**为什么深度学习很少直接用 Hessian**:参数 $n\sim10^6\!-\!10^9$,$H$ 是 $n\times n$,存不下也算不动($O(n^2)$ 存储)。所以实践用只需梯度的一阶方法(SGD、[[39 优化器(Momentum、RMSProp、Adam、AdamW)|Adam]]),或用 Hessian-向量积、对角近似等省事手段。Hessian 主要用于理论分析(收敛性、损失曲面几何、condition number 解释为何要归一化/调学习率)。

**实践中的二阶近似(面试可加分)**:

- **Hessian-向量积(HVP)**:不显式建 $H$,用两次自动微分直接算 $H\mathbf{v}$,$O(n)$ 而非 $O(n^2)$;用于共轭梯度、信赖域。
- **Gauss-Newton / Fisher 近似**:用 $J^\top J$ 近似 Hessian(丢掉二阶残差项),正定且只需一阶雅可比;自然梯度、K-FAC、二阶优化器(如 Shampoo)的基础。
- **对角近似**:Adam 用梯度平方的滑动平均近似 Hessian 对角线,实现"每参数自适应学习率"——这是一阶方法偷偷借了二阶信息。

## 代码

```python
import numpy as np

# 雅可比:y1=x1^2+x2, y2=3x1+sin(x2)
def fvec(x): return np.array([x[0]**2 + x[1], 3*x[0] + np.sin(x[1])])
def jac_analytic(x): return np.array([[2*x[0], 1.0],
                                      [3.0, np.cos(x[1])]])

# ✅ 用数值雅可比验证(对每个输入分量做中心差分)
def jac_numeric(f, x, eps=1e-6):
    n = len(x); fx = f(x); m = len(fx)
    J = np.zeros((m, n))
    for j in range(n):
        e = np.zeros(n); e[j] = eps
        J[:, j] = (f(x + e) - f(x - e)) / (2 * eps)   # 第 j 列 = 对 x_j 的偏导
    return J

x = np.array([1.0, 0.0])
print("解析雅可比:\n", jac_analytic(x))     # [[2,1],[3,1]]
print("数值雅可比:\n", jac_numeric(fvec, x)) # ≈ 同上

# Hessian:f=x^2+3xy+y^2,二阶偏导对称
def f(p): return p[0]**2 + 3*p[0]*p[1] + p[1]**2
def hessian_numeric(f, p, eps=1e-4):
    n = len(p); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pp = p.astype(float).copy(); pm = pp.copy()
            ei = np.zeros(n); ei[i]=eps
            ej = np.zeros(n); ej[j]=eps
            # 二阶混合差分
            H[i, j] = (f(p+ei+ej) - f(p+ei-ej) - f(p-ei+ej) + f(p-ei-ej)) / (4*eps*eps)
    return H

H = hessian_numeric(f, np.array([1.0, 1.0]))
print("数值 Hessian:\n", np.round(H, 3))      # ≈ [[2,3],[3,2]] 对称
print("特征值:", np.round(np.linalg.eigvalsh(H), 3))  # [-1, 5] 一正一负 → 鞍点
```

```python
# 2x2 快速判据:det 和 H11 定极小/极大/鞍点(不必算特征值)
def classify(H):
    det = H[0,0]*H[1,1] - H[0,1]*H[1,0]
    if det < 0: return "鞍点"
    return "极小(碗朝上)" if H[0,0] > 0 else "极大(碗朝下)"
print(classify(np.array([[2.,0.],[0.,2.]])))   # 极小  (x²+y²)
print(classify(np.array([[2.,3.],[3.,2.]])))   # 鞍点  (det=-5)
print(classify(np.array([[-2.,0.],[0.,-1.]]))) # 极大

# 牛顿法对二次函数一步到位 x ← x - H^-1 ∇f
f_grad = lambda p: np.array([2*p[0], 2*p[1]])   # ∇(x²+y²)
H2 = np.array([[2.,0.],[0.,2.]])
x = np.array([3., 4.])
x_newton = x - np.linalg.inv(H2) @ f_grad(x)
print("牛顿一步:", x_newton)   # [0. 0.] 直达最小值
```

手算对照:解析雅可比 $\begin{bmatrix}2&1\\3&1\end{bmatrix}$ 与数值一致;Hessian 数值结果 $\begin{bmatrix}2&3\\3&2\end{bmatrix}$ 对称,特征值 $\{-1,5\}$ 一正一负,印证该点是鞍点;$2\times2$ 判据 $\det<0$ 同样判出鞍点;牛顿法对二次函数一步到原点。

## 面试高频

- **"雅可比和梯度什么关系?"** 梯度是雅可比在输出为标量($m=1$)时的特例;雅可比是向量值函数的全部一阶偏导矩阵($m\times n$)。
- **"Hessian 是什么、有什么用?"** 标量函数的二阶偏导方阵,描述曲面弯曲;特征值全正=极小、全负=极大、有正有负=鞍点。牛顿法用它二阶收敛。
- **"为什么深度学习不直接用 Hessian/牛顿法?"** 参数上百万,$H$ 是 $n\times n$,存储 $O(n^2)$、求逆 $O(n^3)$,算不动;改用一阶方法或其廉价近似。
- **"Hessian 一定对称吗?"** 二阶偏导连续时对称(克莱罗定理),实践中几乎总成立。
- **"鞍点为什么重要?"** 高维损失面鞍点远多于局部极小,SGD 的噪声反而帮助逃离鞍点——这是深度网络能训起来的原因之一。
- **陷阱:雅可比的形状是 $m\times n$ 还是 $n\times m$?** 取决于布局约定。回答时声明"$J_{ij}=\partial y_i/\partial x_j$,即 $m\times n$",避免转置歧义(参见 [[12 矩阵求导|矩阵求导]])。
- **"condition number(Hessian 最大/最小特征值之比)和训练有什么关系?"** 比值大 = 损失面"又长又窄的峡谷",梯度下降会震荡变慢,这正是要做特征缩放/归一化的原因。
- **"$2\times2$ Hessian 怎么不算特征值判极值?"** 看 $\det H$ 和 $H_{11}$:$\det>0\,\&\,H_{11}>0$ 极小、$\det>0\,\&\,H_{11}<0$ 极大、$\det<0$ 鞍点。
- **"牛顿法 vs 梯度下降?"** 牛顿用 $H^{-1}\nabla f$,二阶、对二次函数一步到位、不挑学习率;但 $H^{-1}$ 是 $O(n^3)$,深度学习用一阶或 Gauss-Newton/对角近似。
- **"怎么不建 Hessian 也用上二阶信息?"** Hessian-向量积(两次 AD,$O(n)$)、Gauss-Newton 用 $J^\top J$、Adam 用梯度平方近似对角线。
- **"雅可比行列式在 ML 哪出现?"** 归一化流的密度变量替换因子 $|\det J|$;设计三角雅可比让它易算。

## 关键事实

- 雅可比矩阵作为向量值函数的一阶导、Hessian 作为标量函数的二阶导及二阶极值判据,见 James Stewart《Calculus》(8th ed., 2015)第 14 章与多元微积分标准教材。
- 克莱罗(Schwarz)定理:二阶偏导连续则混合偏导可交换,Hessian 对称。
- 高维非凸损失面以鞍点为主、二阶方法在深度学习中难以直接使用,见 Dauphin et al., 《Identifying and attacking the saddle point problem in high-dimensional non-convex optimization》(NeurIPS, 2014)及 Goodfellow, Bengio & Courville《Deep Learning》(2016)第 4.3、8 章。
