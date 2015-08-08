---
layout: post
title: Notes on Mirror Descent
comments: True
---

这是关于 mirror descent 的不严谨的介绍。严谨的推导可以参考文献[2]。

通常做迭代的空间是Hilber space。但是如果优化变量是在 Banach space $$ \mathcal{B}$$ 中就没有办法按照以前的步骤来了。这是因为 Banach space 的内积没有定义。这时候需要在其对偶空间 $$ \mathcal{B}^*$$ 上更新优化变量。对偶空间 $$ \mathcal{B}^*$$ 是线性函数（从 $$ \mathcal{B}$$ 到域 $$ F$$ ）的向量空间。可以参考维基百科的[定义](https://en.wikipedia.org/wiki/Dual_space)。目标函数的梯度 $$ \nabla f(x)$$ 是对偶空间 $$ \mathcal{B}^*$$ 中的，这时候可以在对偶空间上更新梯度，然后在投影回来就完成了对优化变量的更新。把 $$ \mathcal{B}$$ 中的元素映射成 $$ \mathcal{B}^*$$ 中的元素的函数 $$ \Phi$$ 叫做 mirror map，它是由 $$ \mathcal{B}$$ 上的一个凸开集到实数 $$ R$$ 上的一个函数 ，它需要符合以下三个条件：
- $$ \Phi$$ 是 strictly convex 以及 differentiable 的。
- $$ \Phi$$ 的梯度能去遍所有的 $$ \mathbb{R}^n$$。
- $$ \Phi$$ 的梯度在其定义域的凸开集的边界上是 diverge 的，也就是 $$\lim_{x \to \partial{\mathcal{D}} } \|\Phi\| = + \infty$$

通常称 $$\Phi $$ 把 primal space 的点投影到 dual space 中 ，其实是把  $$ \mathcal{B}$$ 中的点 $$ x$$ 映射成 $$ \nabla \Phi(x)$$ 。这样我们在 dual space 中更新梯度 $$ \nabla \Phi(y) = \nabla \Phi(x) - \eta \nabla f(x)$$，其中 $$ y$$ 是 primal space 中的点，这样我们就可以在 primal space 中更新优化变量了。当然，$$ y$$ 可能在目标函数的定义域 $$ \mathcal{X}$$ 之外，需要在 primal space 上投影回这个定义域，这个投影用的是 Bregman divergence 来做：

$$
\Pi_{\mathcal{X}}^{\Phi}(y)= argmin_{x \in \mathcal{X}\cap \mathcal{D} } D_{\Phi}(x,y)
$$

其中 $$ D_{\Phi}(x,y)$$ 是对于函数 $$ f$$ 的 Bregman divergence， 定义如下

$$
D_{f}(x,y) = f(x) - f(y) - \nabla f(y)^T(x-y)
$$

它是 $$ f(x)$$ 跟在 $$ y$$ 处的一阶近似做差得到的。因此，mirror descent 的步骤总结如下：

$$
\begin{align}
\nabla \Phi(y_{t+1}) &= \nabla \Phi(x_t) - \eta \nabla f(x_t) \\
x_{t+1} &=argmin_{x \in \mathcal{X}\cap \mathcal{D} } \ D_{\Phi}(x,y_{t+1})
\end{align}
$$

对于上面的这两个步骤可以有如下的另一种表示方法

$$
\begin{align}
x_{t+1} &= argmin_{x} \ D_{\Phi}(x,y_{t+1}) \\
&= argmin_{x} \ \Phi(x) - \Phi(y_{t+1}) - \nabla \Phi(y_{t+1})^T(x-y_{t+1}) \\
&= argmin_{x} \ \Phi(x)  - \nabla \Phi(y_{t+1})^T x \\
&= argmin_{x} \ \Phi(x)  -  (\nabla \Phi(x_t) - \eta \nabla f(x_t))^T x \\
&= argmin_{x} \ \Phi(x)  -  \nabla \Phi(x_t)^T x + \eta \nabla f(x_t)^T x \\
&= argmin_{x} \ \eta \nabla f(x_t)^T x + \Phi(x) -\Phi(x_t) -  \nabla \Phi(x_t)^T (x-x_t)   \\
&= argmin_{x} \ \eta \nabla f(x_t)^T x  + D_{\Phi}(x, x_t)
\end{align}
$$

最后这一步就是 proximal gradient descent  的迭代步骤了，因此 mirror descent 可以看作更general 的 proximal gradient descent。所有对于 proximal gradient descent 的方法都可以用到 MD 上，并且 MD 还好有个好处就是可以采用更好的 mirror map 函数来加速收敛。


Nesterov 把 MD 更新的第一个步骤改为

$$
\nabla \Phi(y_{t+1}) = \nabla \Phi(y_t) - \eta \nabla f(x_t)
$$

也就是把 $$ \Phi(x_t)$$ 改成了 $$ \Phi(y_t)$$，这相当与省略了一个投影的步骤，直接在上一次的 $$ y_t$$ 上继续更新。这称为 lazy mirror descent，也叫做 Nesterov's dual averaging。所以，这里的 dual 并不是指的目标问题的对偶问题。同样的，根据把 MD 化成 proximal gradient descent  的方法可以得到

$$
\begin{align}
x_{t+1} &= argmin_{x} \ D_{\Phi}(x,y_{t+1}) \\
&= argmin_{x} \ \Phi(x) - \Phi(y_{t+1}) - \nabla \Phi(y_{t+1})^T(x-y_{t+1}) \\
&= argmin_{x} \ \Phi(x)  - \nabla \Phi(y_{t+1})^T x \\
&= argmin_{x} \ \Phi(x)  -  (\nabla \Phi(y_t) - \eta \nabla f(x_t))^T x \\
&= argmin_{x} \ \Phi(x)  -  \nabla \Phi(y_t)^T x + \eta \nabla f(x_t)^T x \\
&= argmin_{x} \ \Phi(x)  -  (\nabla \Phi(y_{t-1}) - \eta \nabla f(x_{t-1}))^T x + \eta \nabla f(x_t)^T x \\
&= argmin_{x} \ \Phi(x)  -  \nabla \Phi(y_{t-1})^T x + \eta \nabla f(x_{t-1})^T x + \eta \nabla f(x_t)^T x \\
& \cdots \\
&= argmin_{x} \ \Phi(x)  -  \nabla \Phi(y_{1})^T x + \eta \sum_{i}^{t} \nabla f(x_{i})^T x \\
&= argmin_{x} \ \Phi(x)  + \eta \sum_{i}^{t} \nabla f(x_{i})^T x 
\end{align}
$$

这里通常 $$ \nabla \Phi(y_{1})=0$$。

MD 和 lazy MD 的 convergence rate 是 $$ O(1/\sqrt{t})$$。对于光滑函数可以加速达到 $$ O(1/t)$$，下面的 mirror prox 是其中的一种方法：
$$
\begin{align}
\nabla \Phi(y_{t+1}') &= \nabla \Phi(x_t) - \eta \nabla f(x_t) \\
y_{t+1} &=argmin_{x \in \mathcal{X}\cap \mathcal{D} } \ D_{\Phi}(x,y_{t+1}')\\
\nabla \Phi(x_{t+1}') &= \nabla \Phi(x_t) - \eta \nabla f(y_{t+1}) \\
x_{t+1} &=argmin_{x \in \mathcal{X}\cap \mathcal{D} } \ D_{\Phi}(x,x_{t+1}')
\end{align}
$$



# Reference
1. [Bregman divergence and mirror descent](http://users.cecs.anu.edu.au/~xzhang/teaching/bregman.pdf)
2. Bubeck, Sébastien. "Theory of convex optimization for machine learning." arXiv preprint arXiv:1405.4980 (2014).
