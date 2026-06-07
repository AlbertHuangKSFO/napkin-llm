[[114 手写自动微分引擎(micrograd 级)|手写自动微分引擎]]:用约 100 行 Python 实现一个 micrograd 级的**标量反向模式自动微分**——`Value` 类在前向时建计算图、记录每个运算的「本地导数」,`backward()` 用**拓扑排序逆序 + 链式法则**把梯度一路回传到所有叶子。这是 PyTorch `.backward()` 的去魅版,理解它就理解了整个深度学习训练的引擎。

## 直觉

PyTorch 里 `loss.backward()` 像魔法:调一下,所有参数的 `.grad` 就有了。这魔法其实只有两步:

1. **前向时偷偷建图**:每做一次运算(`a*b`、`x.tanh()`),不仅算出结果,还记下「我是谁算出来的」(父节点)和「对每个父节点的本地导数怎么算」(一个闭包 `_backward`)。
2. **反向时按序回传**:从最终的 loss 出发,把它的梯度设为 1,然后**逆着拓扑序**走,每个节点把自己的 `grad` 乘上本地导数,**累加**给父节点——这就是链式法则(见 [[20 反向传播的数学推导|反向传播]])。

核心数据结构是 `Value`:它包一个标量 `data`、一个 `grad`(对最终 loss 的偏导)、一组父节点 `_prev`、和一个 `_backward` 闭包。**整个自动微分库就是这一个类。**

为什么用标量(micrograd 级)而不是张量?因为标量版最透明——你能在纸上画出每个节点、手验每个梯度。PyTorch 把同一套规则搬到张量上(本地导数变成雅可比向量积),数学一模一样,只是高效。

![[impl-Value计算图前向反向.png]]

## 例子

**手算一遍 $L=(a\cdot b+c).\tanh()$,$a{=}2,b{=}{-}3,c{=}10$。**

前向(蓝):$e=a\cdot b=-6$ → $n=e+c=4$ → $L=\tanh(4)\approx0.9993$。

反向(红),从 $L.\text{grad}=1$ 开始,逆拓扑序:

| 节点 | 本地导数 | 回传计算 | grad |
|---|---|---|---|
| $L=\tanh(n)$ | $\frac{dL}{dn}=1-\tanh^2(4)\approx0.00134$ | $n.\text{grad}{+}{=}1\cdot0.00134$ | $n:0.00134$ |
| $n=e+c$ | 加法原样分发 $\frac{\partial n}{\partial e}{=}\frac{\partial n}{\partial c}{=}1$ | $e,c$ 各 $+0.00134$ | $e,c:0.00134$ |
| $e=a\cdot b$ | 乘法交换对方 data:$\frac{\partial e}{\partial a}{=}b,\frac{\partial e}{\partial b}{=}a$ | $a{+}{=}(-3)(0.00134)$,$b{+}{=}(2)(0.00134)$ | $a:-0.00402,\ b:0.00268$ |

代码跑出来一字不差(见 `## 代码` 的输出注释)。

**为什么必须拓扑排序。** 若一个变量被用了两次(如 $y=x^2+3x$,$x$ 出现两次),它的总梯度是两条路径之和 $\frac{dy}{dx}=2x+3$。**必须等所有下游节点都把贡献回传完,才能确定 $x$ 的 grad**——所以反向要按「前向拓扑序的逆序」走,且 grad 用 `+=` 累加(不是 `=`,否则后一条路径覆盖前一条)。这是手写 autodiff 最容易踩的两个坑:**忘了拓扑序、用了 `=`**。

![[impl-topo反向排序.png]]

**数值梯度检验(必做的自检)。** 解析梯度对不对?用有限差分 $\frac{\partial f}{\partial x}\approx\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$ 比对(见 [[19 自动微分：前向 vs 反向模式|数值梯度]]):一个含 `tanh/relu/exp/除法/幂` 的复杂表达式,解析与数值的最大误差 $\sim10^{-11}$,过关。

**再手算一个「分叉」例子,看清 `+=` 为什么必要。** $y=x^2+3x$,$x=2$。前向:$x^2=4$、$3x=6$、$y=10$。$x$ 被用了**两次**(平方支路 + 线性支路)。反向($y.\text{grad}=1$):
- 经 $x^2$ 支路回传:$x.\text{grad}\mathrel{+}=2x\cdot1=4$。
- 经 $3x$ 支路回传:$x.\text{grad}\mathrel{+}=3\cdot1=3$。
- 合计 $x.\text{grad}=4+3=7$,正好等于 $\frac{dy}{dx}=2x+3=7$。✅

若用 `=` 而非 `+=`,后回传的支路会**覆盖**前一条,$x.\text{grad}$ 只剩 3(或 4),梯度错。这就是「分叉变量梯度是各路径之和」的代码体现——也是手写 autodiff 最隐蔽的 bug:小网络没分叉时用 `=` 也能跑对,一上残差/权重共享就错,且不报错、只是悄悄训不好。

**$\epsilon$ 怎么选(数值检验的坑)。** 太大($10^{-2}$)→ 截断误差(差分近似不准);太小($10^{-10}$)→ 浮点舍入误差(两个相近数相减损失有效位)。甜点约 $10^{-6}\sim10^{-4}$(双精度)。中心差分 $\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$ 误差 $O(\epsilon^2)$,比单边 $\frac{f(x+\epsilon)-f(x)}{\epsilon}$ 的 $O(\epsilon)$ 精度高一阶——所以一律用中心差分。

## 原理

**1. 计算图与链式法则。** 任何复合函数都能拆成基本运算的 DAG(见 [[18 计算图(前向)|计算图]])。设最终输出 $L$,对中间节点 $v$ 的梯度记 $\bar v=\frac{\partial L}{\partial v}$。若 $v$ 的所有「下游」(直接用到 $v$ 的节点)是 $\{u_i\}$,则多元链式法则给出:

$$\bar v=\sum_i \bar u_i\,\frac{\partial u_i}{\partial v}$$

$\frac{\partial u_i}{\partial v}$ 就是**本地导数**(只依赖该运算自身),$\bar u_i$ 是下游已算好的梯度。反向模式的精髓:**先算靠近输出的梯度,再往输入推**,一次反向就拿到所有叶子的梯度——比前向模式对多输入单输出的场景高效得多(见 [[19 自动微分：前向 vs 反向模式|前向 vs 反向]])。

**2. 每个运算的本地导数(就这几条,够搭整个网络)。**

$$\begin{aligned}
&\text{加法 } c=a+b:&&\bar a\mathrel{+}=\bar c,\quad \bar b\mathrel{+}=\bar c\\
&\text{乘法 } c=a\cdot b:&&\bar a\mathrel{+}=b\,\bar c,\quad \bar b\mathrel{+}=a\,\bar c\\
&\text{幂 } c=a^p:&&\bar a\mathrel{+}=p\,a^{p-1}\,\bar c\\
&\tanh:\ c=\tanh a:&&\bar a\mathrel{+}=(1-c^2)\,\bar c\\
&\text{ReLU}:\ c=\max(0,a):&&\bar a\mathrel{+}=[a>0]\,\bar c\\
&\exp:\ c=e^a:&&\bar a\mathrel{+}=c\,\bar c
\end{aligned}$$

减法 = 加负、除法 = 乘幂 $-1$,都能复用。有了这几条,加上嵌入/矩阵乘的张量版,就能搭出任意神经网络(见 [[38 反向传播在网络中的实现|反向传播实现]])。

**3. 拓扑排序保证依赖正确。** 反向时,算 $\bar v$ 前必须所有 $\bar u_i$ 已就绪。用 DFS 后序得到拓扑序(父在子后),**逆序遍历**即满足此条件。grad 必须 `+=` 累加,因为一个节点可能被多个下游共享(分叉)。

**3'. 为什么反向模式对神经网络是「天选」(成本对比)。** 设网络有 $n$ 个输入参数、$1$ 个标量输出(loss)。**反向模式**:一次反向遍历,成本约等于一次前向(常数倍),就拿到**全部 $n$ 个参数的梯度**——与参数量无关的「一遍搞定」。**前向模式**:每次只能算「所有输出对一个输入」的导数,要拿全部 $n$ 个参数梯度得跑 $n$ 遍——参数上亿时完全不可行。神经网络恰好是「极多输入、单标量输出」,所以**反向模式是唯一可行解**。反过来,若是「单输入、多输出」(如雅可比的某些科学计算场景),前向模式才更划算(见 [[19 自动微分：前向 vs 反向模式|前向 vs 反向]])。

**4. 与 PyTorch 的对应。** PyTorch 的 `Tensor` 就是张量版 `Value`:`requires_grad=True` 的张量记录 `grad_fn`(= 我们的 `_backward`),`.backward()` 做的就是拓扑逆序回传,只不过本地导数是**雅可比向量积(VJP)**而非标量乘。`optimizer.zero_grad()` 对应「把 grad 清零」(因为 `+=` 会累加,不清就把上一步的梯度叠进来——这是新手最常见 bug,见 [[117 训练一个 tiny GPT(PyTorch,可跑)|训练循环]])。

为什么张量版叫 **VJP** 而非雅可比矩阵?设运算 $y=f(x)$,$x\in\mathbb{R}^n,y\in\mathbb{R}^m$,雅可比 $J\in\mathbb{R}^{m\times n}$。反向模式拿到的是上游梯度 $\bar y\in\mathbb{R}^m$,要算 $\bar x=J^\top\bar y$——这是**雅可比的转置乘一个向量**,无需显式构造 $J$(那会很大)。每个算子只要实现「给 $\bar y$ 返回 $\bar x$」即可,正是我们标量版 `_backward` 的张量推广。这就是为什么 autodiff 能在巨大网络上高效跑:从不显式建雅可比,只做 VJP。

**5'. softmax + 交叉熵的梯度(GPT 的输出层,值得手推一次)。** 输出 logits $z$,softmax 得 $p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$,真标签 $y$(one-hot,正确类 $k$),交叉熵 $L=-\log p_k$。对 logits 的梯度极其干净:

$$\frac{\partial L}{\partial z_i}=p_i-y_i=\begin{cases}p_k-1&i=k\\ p_i&i\ne k\end{cases}$$

「预测概率减去真值」——这就是为什么深度学习框架把 softmax 和 cross-entropy **融合成一个算子**(`F.cross_entropy`):分开写不仅数值不稳(见 softmax 减最大值),梯度也得分两步;合起来梯度直接是 $p-y$,又稳又快。tiny GPT 的反传链条最末端就是这个 $p-y$,一路按链式法则往前乘,经过每个 Block,最终到嵌入和所有权重。理解了这一项 + 上面几条本地导数,你就理解了整个 GPT 的反传。

## 代码

完整可运行(纯标准库,无依赖)。已通过数值梯度检验、并能训练一个微型神经元。直接贴进 `engine.py` 跑。

```python
# engine.py —— micrograd 级标量自动微分（约 100 行，纯 Python）
import math

class Value:
    """包一个标量 data，记录梯度 grad、父节点 _prev 和反传闭包 _backward。"""
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0                       # 对最终输出的偏导，初始 0
        self._backward = lambda: None         # 本地反传规则（叶子为空）
        self._prev = set(_children)           # 父节点（建图用）
        self._op = _op                        # 调试用：这个节点由什么运算产生

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():                      # 加法：梯度原样分发
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():                      # 乘法：交换对方的 data
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Value(self.data ** p, (self,), f'**{p}')
        def _backward():
            self.grad += p * (self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):                            # 自然对数，d/dx ln x = 1/x（softmax+CE 要用)
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # 由上面几个原语派生
    def __neg__(self):          return self * -1
    def __sub__(self, other):   return self + (-other)
    def __radd__(self, other):  return self + other         # 支持 2 + Value
    def __rmul__(self, other):  return self * other          # 支持 2 * Value
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def backward(self):
        # 1) 拓扑排序（DFS 后序：父在子之后）
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        # 2) 逆序回传：从自己 grad=1 开始，每个节点执行本地反传
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

```python
# —— 验证 1：图示例 L=(a*b+c).tanh()，梯度应与手算一致 ——
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c).tanh()
L.backward()
print(L)                         # Value(data=0.9993, grad=1.0000)
print(a.grad, b.grad, c.grad)    # -0.004023  0.002682  0.001341
# 对照手算：dL/dn=1-tanh^2(4)≈0.001341；a.grad=b*dLdn, b.grad=a*dLdn, c.grad=dLdn ✅
```

```python
# —— 验证 2：数值梯度检验（解析 vs 有限差分），最大误差应 < 1e-4 ——
def f(x0, x1, x2):
    return (x0 * x1 + x2.tanh()).relu() * (x0 ** 2) + (x1 / x2).exp()

xs = [Value(1.5), Value(-2.0), Value(3.0)]
out = f(*xs); out.backward()
ana = [v.grad for v in xs]

eps = 1e-6
def fval(vals): return f(*[Value(v) for v in vals]).data
base = [1.5, -2.0, 3.0]
num = [(fval([*base[:i], base[i]+eps, *base[i+1:]])
        - fval([*base[:i], base[i]-eps, *base[i+1:]])) / (2*eps)
       for i in range(3)]
print("analytic:", [round(x, 5) for x in ana])   # [0.0, 0.17114, 0.11409]
print("numeric :", [round(x, 5) for x in num])    # [0.0, 0.17114, 0.11409]
print("max err  :", max(abs(a-b) for a, b in zip(ana, num)))   # ~4e-11 ✅ 过关
```

```python
# —— 验证 3：用这个引擎训练一个神经元，能学会非线性目标 ——
import random; random.seed(42)
w = [Value(random.uniform(-1, 1)) for _ in range(3)]; bias = Value(0.0)
def neuron(x):                       # tanh(w·x + b)
    return (sum((wi*xi for wi, xi in zip(w, x)), bias)).tanh()

X = [[2.,3.,-1.], [3.,-1.,.5], [.5,1.,1.], [1.,1.,-1.]]
Y = [1., -1., -1., 1.]
for step in range(100):
    preds = [neuron([Value(v) for v in x]) for x in X]
    loss = sum(((p - y) ** 2 for p, y in zip(preds, Y)), Value(0.0))  # MSE
    for p in w: p.grad = 0.0          # ❗ 必须清零，否则 += 把上一步梯度叠进来
    bias.grad = 0.0
    loss.backward()                   # 反传
    for p in w: p.data -= 0.05 * p.grad   # 手写 SGD 一步
    bias.data -= 0.05 * bias.grad
print("final loss:", round(loss.data, 4))   # ≈ 0.0135，从 ~4 降到 <0.05，学会了 ✅
```

```python
# —— 验证 4：手算分叉，确认 += 累加（用 = 会错）——
x = Value(2.0)
y = x * x + 3 * x            # y = x^2 + 3x，x 被用两次（分叉）
y.backward()
print("x.grad =", x.grad)    # 7.0  = 2x+3 = 2*2+3 ✅（两条支路 4 和 3 累加）
# 若把 Value 里的 self.grad += ... 改成 self.grad = ...，这里会得到 3 或 4（错）

# —— 验证 5：softmax + 交叉熵的梯度应等于 p - y_onehot（GPT 输出层反传末端）——
import math
zs = [Value(2.0), Value(1.0), Value(0.1)]; k = 0          # logits，正确类 = 0
m = max(v.data for v in zs)                               # 数值稳定：减最大值
exps = [(v + (-m)).exp() for v in zs]
Z = sum(exps, Value(0.0))
p = [e / Z for e in exps]                                 # softmax 概率
L = (p[k]).log() * Value(-1.0)                            # L = -log p_k（交叉熵）
L.backward()
y_onehot = [1.0, 0.0, 0.0]
print("p      =", [round(pi.data, 4) for pi in p])        # [0.659, 0.2424, 0.0986] 和为 1
print("dL/dz  =", [round(z.grad, 4) for z in zs])         # [-0.341, 0.2424, 0.0986]
print("p - y  =", [round(p[i].data - y_onehot[i], 4) for i in range(3)])  # [-0.341, 0.2424, 0.0986] 完全一致 ✅
# 结论：dL/dz_i = p_i - [i==k]，所以框架把 softmax 与 CE 融合成一个算子（又稳又快）
```

```python
# ============ 扩展练习 ============
# 1. 给 Value 加 log / sin / sigmoid 算子（各写本地导数 + 数值检验）。
# 2. 实现一个 MLP 类（多个 neuron 串联），用本引擎训练一个二分类玩具数据集。
# 3. 把标量 Value 换成 numpy 数组版（本地导数变 VJP），对比与 PyTorch 的一致性。
# 4. 故意把某算子的 += 改成 =，构造一个分叉网络复现「梯度错但不报错」的 bug。
```

## 面试高频

- **Q:`loss.backward()` 内部到底做了什么?** A:两步——前向时已建好计算图(每个运算记下父节点和本地导数闭包);backward 先对图做拓扑排序,从 loss 设 grad=1,**逆拓扑序**遍历,每个节点把自己的 grad 乘本地导数**累加**给父节点(链式法则)。
- **Q:为什么反向要拓扑排序?** A:算某节点梯度前,它所有下游的梯度必须已就绪(否则少加贡献)。拓扑逆序保证父节点在所有子节点之后处理。
- **Q:grad 为什么用 `+=` 而不是 `=`?** A:一个变量可能被多条路径用到(分叉),总梯度是各路径贡献之和。用 `=` 会让后一条覆盖前一条,梯度算错。也正因累加,每步训练前必须 `zero_grad()`。
- **Q:`tanh` / 乘法的本地导数怎么来?** A:乘法 $c=ab$ 对 $a$ 的本地导是 $b$(交换对方);$\tanh$ 的导数是 $1-\tanh^2$,所以 $\bar a{+}{=}(1-c^2)\bar c$。这些只依赖运算本身,与下游无关。
- **Q:怎么验证自动微分写对了?** A:数值梯度检验——用 $\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$(中心差分)和解析梯度比对,误差应在 $10^{-6}$ 量级以下。
- **Q:前向模式和反向模式区别?** A:反向模式一次得到「单输出对所有输入」的梯度,适合神经网络(loss 是标量,参数极多);前向模式一次得到「所有输出对单输入」的梯度,适合输入少输出多的场景(见 [[19 自动微分：前向 vs 反向模式|前向 vs 反向]])。
- **Q:softmax+交叉熵对 logits 的梯度是什么?** A:$\frac{\partial L}{\partial z_i}=p_i-y_i$(预测概率减真值 one-hot),极其干净。所以框架把 softmax 与 CE 融合成一个算子(`F.cross_entropy`):又数值稳定、梯度又直接是 $p-y$。这是 GPT 反传链条的末端。
- **Q:张量版 autodiff 为什么叫 VJP、不显式建雅可比?** A:反向模式每个算子算的是 $\bar x=J^\top\bar y$(雅可比转置乘上游梯度向量),无需构造完整雅可比 $J$(那太大)。只实现「给上游梯度返回下游梯度」即可,这是标量 `_backward` 的张量推广,也是 autodiff 能在巨网络上高效跑的原因。
- **Q:数值梯度检验 $\epsilon$ 怎么选?** A:太大有截断误差、太小有浮点舍入误差,甜点约 $10^{-6}\sim10^{-4}$;用中心差分(误差 $O(\epsilon^2)$,优于单边 $O(\epsilon)$)。

## 关键事实

- 本实现对应 Karpathy 的 **micrograd**(github.com/karpathy/micrograd,约 150 行标量 autodiff + 一个 MLP),配套讲解见 "The spelled-out intro to neural networks and backpropagation: building micrograd"(2022)。
- 反向模式自动微分 = 计算图上的反向传播,理论见 [[20 反向传播的数学推导|反向传播推导]];计算图概念见 [[18 计算图(前向)|计算图]];前向/反向模式对比见 [[19 自动微分：前向 vs 反向模式|自动微分]]。
- PyTorch `autograd` 是本类的张量版:`grad_fn` ≈ `_backward`,`.backward()` 做拓扑逆序回传 + 雅可比向量积;`zero_grad()` 对应 grad 累加语义下的清零(见 [[117 训练一个 tiny GPT(PyTorch,可跑)|训练循环]])。
- 三个原子坑:忘拓扑排序(梯度算错)、grad 用 `=` 而非 `+=`(分叉变量梯度错)、训练步间忘 zero_grad(梯度越叠越大)。
- 数值梯度检验是验证自动微分实现的金标准,中心差分 $O(\epsilon^2)$ 比单边差分 $O(\epsilon)$ 精度高;深度学习里只在调试自定义算子时用,正式训练全靠反向模式。
- softmax+交叉熵对 logits 的梯度恒为 $p-y$(预测概率减 one-hot 真值),故框架融合两者为单算子(数值稳 + 梯度直接),这是 GPT 反传链条的最末端。
- 张量版 autodiff 用雅可比向量积(VJP):$\bar x=J^\top\bar y$,从不显式构造雅可比 $J$,只实现「上游梯度→下游梯度」,这是 autodiff 在巨网络上高效的关键。
- 关联:此引擎是 [[113 从零实现总览：课程地图到代码|从零实现总览]] 的 `engine.py`,理解后过渡到 numpy 注意力 [[115 手写多头注意力与 Transformer Block(numpy)|numpy 多头注意力]] 与 PyTorch 训练 [[117 训练一个 tiny GPT(PyTorch,可跑)|tiny GPT]];优化器更新规则见 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|优化器]]。
