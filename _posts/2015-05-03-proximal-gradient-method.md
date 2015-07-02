---
layout: post
title: Proximal Gradient Methods
comments: True
---

## Proximal Operator
对于凸函数 $$ f(x)$$， proximal operator 的定义为

$$
\textrm{prox}_{f}(v) = \textrm{argmin}_{x} \left( f(x) + \frac{1}{2} \|x-v\|_2^2\right).
$$

我们会遇到针对函数 $$ \lambda f$$ 的 proximal operator。相似的，它的定义为

$$
\textrm{prox}_{\lambda f}(v) = \textrm{argmin}_{x} \left( f(x) + \frac{1}{2\lambda} \|x-v\|_2^2\right).
$$

通常 $$ \textrm{prox}_{f}(v)$$ 也称作 $$v$$ 的对于函数 $$f$$ 的 proximal point。如果 $$x^*$$ 最小化函数 $$ f$$， 等价于 $$ x^* = \textrm{prox}_{f}(x^*)$$。这样的点叫做 $$ \textrm{prox}_{f}$$  的 fixed points。因此找函数的 minimizer 就和 operator  的 fixed points 相关联了。以此需要了解一下 operator 的东西。


## Proximal Operator and Fixed Points
这一节简要介绍  proximal operator 的 fixed points， 主要整理自 [Stanford EE364b 的 leture notes](http://stanford.edu/class/ee364b/lectures/monotone_slides.pdf)，目的是从 operator 的角度来解释为什么迭代地应用 proximal operator 能收敛到最优解。

首先定义 relation R 是一个 $$ R^n \times R^n$$ 上的子集，然后定义

$$ R(x) = \{ y \mid (x,y) \in R\}.$$

一个 $$ R^n $$ 上的 relation F 是 monotone 如果对于对所有的 $$ (x,u), (y,v) \in F$$ 都满足如下的条件：

$$
(u-v)^T(x-y) \geq 0 .
$$

我们称 Relation $$ F$$ 有 Lipschitz 常数 $$ L$$， 如果对所有的 $$ x,y \in \textrm{dom} F$$ 满足如下的条件

$$
\|F(x) - F(y)\|_2 \leq L\|x-y\|_2.
$$

如果 $$ L=1$$, $$F$$ 称作 nonexpansive，如果 $$ L < 1$$ 那么 $$F$$ 是一个 contraction。如果 $$F$$ 是 nonexpansive 的，那么它的 fixed point 集合是凸集。如果 $$F$$ 是 contraction， 那么它有一个 fixed points， 并且可以由如下的迭代来找到：

$$
x^{k+1} = F(x^k).
$$

但是如果 $$F$$ 是 nonexpansive， 此过程不一定收敛（即使 $$F$$ 的 fixed point set 是空集也不一定收敛）。但是如果我们定义另一个 operator $$ T = (1-\alpha) I + \alpha F$$（$$T$$ 和 $$F$$ 有相同的 fixed points），其中 $$ \alpha \in (0,1)，$$  $$T$$ 会收敛到 $$T$$ 的 fixed point，其实也就是 $$F$$ 的 fixed point。这个迭代过程可以表示为：

$$
x^{k+1} = (1-\alpha) x^k + \alpha F(x^k).
$$

这称作 $$F$$ 的 damped iteration。$$T$$ 称作 $$F$$的 $$ \alpha$$-averageed operators。


由于 proximal operator 满足如下被称为 firmly nonexpensiveness 的条件：

$$
\|\textrm{prox}_{f}(x) - \textrm{prox}_{f}(y)\|_{2}^2 \leq (x-y)^T(\textrm{prox}_{f}(x) - \textrm{prox}_{f}(y)).
$$

满足这个条件的 operator 被称为 firm nonexpensive operator。这样的 operator 是$$ \frac{1}{2}$$-averaged operator。总之，contraction 和 firmly nonexpensiveness 的 operator 是 average operator 的子集。 因此迭代地使用 proximal operator 会收敛到 fixed point。 


## Inperpretations of Proximal oOperator

从 proximal operator 的定义可以看出， $$ \textrm{prox}_{\lambda f}(v)$$ 也是这个问题的解：

$$x = v - \lambda \nabla f(x) ,$$

这并不是一个梯度的迭代步骤。可以从以下的角度理解 proximal operator。

### Moreau Envelope
函数 $$ \lambda f$$ 的 Moreau envelope 定义如下：

$$
M_{\lambda f}(v) = \inf_x \left( f(x) + \frac{1}{2\lambda }\|x-v\|_2^2 \right).
$$

二维的 Moreau envelope 图形可以看[这里](http://oldweb.cecm.sfu.ca/projects/CCA/FCT/demo/HTML/)。Moreau envelope 其实是 $$ f$$ 的光滑的近似版本。他们有着相同的最优解。根据 proximal operator 的定义，我们可以把 Moreau envelope 表示为：

$$
M_{\lambda f}(v) =  f(\textrm{prox}_{\lambda f}(v)) + \frac{1}{2\lambda }\|\textrm{prox}_{\lambda f}(v)-v\|_2^2.
$$

Moreau envelope 的导数可以表示为：

$$
\nabla M_{\lambda f}(v) = \frac{1}{\lambda } (v - \textrm{prox}_{\lambda f}(v)).
$$

因此， 

$$\textrm{prox}_{\lambda f}(v) = v - \lambda   \nabla M_{\lambda f}(v) .$$

所以 proximal operator 可以看做是在其 Moreau envelope 上的梯度下降，步长为 $$ \lambda $$。

### Modified Gradient Descent 
我们考察 proximal operator 在函数 $$ f$$ 的一阶和二阶近似上的行为。我们记其在 $$v$$ 附近的 一阶和二阶的近似为

$$
\begin{align}
f_v^{(1)} &= f(v) + \nabla f(v)^T (x-v) \\
f_v^{(2)} &= f(v) + \nabla f(v)^T (x-v) + \frac{1}{2}(x-v)^T \nabla^2 f(v) (x-v).
\end{align}
$$

我们可以求出 proximal operator 的解析解如下

$$
\begin{align}
\textrm{prox}_{\lambda f_v^{(1)}} &= v - \lambda \nabla f(v) \\
\textrm{prox}_{\lambda f_v^{(2)}} &= v - (\nabla^2 f(v) + \frac{1}{\lambda} I )^{-1} \nabla f(v) .
\end{align}
$$

因此，如果每一个步骤都用 $$ f $$ 的一阶的近似，那么 proximal operator 就是一个标准的梯度下降，如果每一个步骤用 $$ f$$ 的二阶近似，那么 proximal operator 是加了正则的牛顿迭代。如果 $$\lambda  \to 0 $$， $$ \textrm{prox}_{\lambda f_v^{(2)}} \approx v - \lambda \nabla f(v)  $$，此时相当于没有用到二阶的信息。

另外，从问题 $$ x = v - \lambda \nabla f(x) $$ 考虑，如果 $$\lambda  \to 0 $$， $$ x \approx v - \lambda \nabla f(v) $$，此时 $$ \textrm{prox}_{\lambda f} $$ 近似为 $$ f$$ 上的一个梯度下降的步骤。



## Proximal Gradient Method
对于凸函数 $$ f$$ 可以直接用如下的迭代

$$
x^{k+1} = \textrm{prox}_{\lambda f} (x^k)
$$

来找到最优解。但是极少被使用，因为 $$ f$$ 加上二次的项这个目标函数有可能不那么容易优化。但是如果目标函数可以分解成两个项的和，比如我们的问题可以表示为

$$
\min f(x) + g(x),
$$

通常 $$ f(x)$$是光滑可微的，但 $$ g(x)$$ 不一定可微， 例如 $$ g(x)$$可以是 $$ \ell_1$$-norm。 我们可以用如下的迭代来解

$$
x^{k+1} = \textrm{prox}_{\lambda^k g} (x^k - \lambda^k \nabla f(x^k)).
$$

这称为 proximal gradient method。 当 $$ f $$ 是 $$ L$$-smooth 的，并且用固定的步长 $$ \lambda^k \in [0,L]$$ 的时候有 convergence rate $$ O(1/k)$$。如果 $$ L $$ 不知道，可以用 line search 来找到合适的步长。


这个过程可以用 majorization minimization (MM) 来理解。 用 MM 算法来找函数 $$ h(x)$$ 的最优解可以表示为如下的迭代过程

$$
x^{k+1} = \textrm{argmin}_{x} \hat{h}(x, x^k),
$$

其中凸函数 $$ \hat{h}(x, x^k) $$ 是 $$ h(x)$$ 的 tight upper bound，也就是有 $$ \hat{h}(x, x^k) \geq h(x) $$ 并且 $$ \hat{h}(x^k, x^k) = h(x^k) $$ 成立。这样的 upper bound 叫 matorization，它不一定唯一。MM 算法每一步都是在最小化 matorization。 对于我们关心的问题，我们考虑函数 $$ f(x)$$的一个 matorization ：

$$
\hat{f}_{\lambda}(x,y) = f(y) + \nabla f(y)^T(x-y) + \frac{1}{2\lambda} \|x-y\|_2^2.
$$

当 $$ \lambda \geq \frac{1}{L}$$ 的时候， $$ \hat{f}_{\lambda}(x,y) \geq f(x)$$ 并且 $$ \hat{f}_{\lambda}(x,x) = f(x)$$。 定义函数 $$ q_{\lambda} (x,y)$$ 为

$$
q_{\lambda} (x,y) = \hat{f}_{\lambda}(x,y)  + g(x).
$$

显然， $$ q_{\lambda} (x,y)$$  是 $$ f(x)+g(x)$$ 的 majorization。因此我们的问题可以用如下的迭代步骤来解

$$
x^{k+1} = \textrm{argmin}_{x} q_{\lambda}(x,x^k).
$$

这恰好就是如下 proximal operator的迭代步骤：

$$
x^{k+1} = \textrm{prox}_{\lambda^k g} (x^k - \lambda^k \nabla f(x^k)).
$$

上面的方法的 convergence rate 是 $$ O(1/k)$$，通过 Nesterov's method 来加速到 $$ O(\frac{1}{k^2})$$。步骤如下：

$$
\begin{align}
y^{k+1} &= x^k + \omega^k(x^k - x^{k-1}) ,\\
x^{k+1} &= \textrm{prox}_{\lambda^k g} (y^k - \lambda^k \nabla f(y^k)).
\end{align}
$$

其中 $$ \omega^k \in [0,1)$$，$$ \omega^k $$ 需要以特定的方法选出来。一个简单的方法是：$$ \omega^k = \frac{k}{k+3}$$。对于加速和非加速版本的 proximal gradient method 的 $$ \lambda $$ 都可以用 line search 的方法来找。

# Reference
1. Parikh, Neal, and Stephen Boyd. "Proximal algorithms." Foundations and Trends in optimization 1.3 (2013): 123-231.
