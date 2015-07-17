---
layout: post
title: Strong Convexity and Smoothness
comments: True
---

函数自身的性质对 convergence rate 有很大的影响，优化中主要用以下的两种度量来刻画函数的性质。

## $$ \alpha$$-strong Convexity

一个函数 $$ f: \mathcal{X} \mapsto \mathbb{R} $$ 是 $$ \alpha$$-strong convex 需要满足以下的条件：

$$ f(x) - f(y) \leq \nabla f(x)^T(x-y) - \frac{\alpha}{2}\|x-y\|_2^2,$$

其实此定义对不可微分的函数也成立，只要把导数换成 subgradient。$$f(y) $$在 $$ x$$出的二阶近似为：

$$ f(y) \approx f(x) + \nabla f(x)^T(y-x) + (y-x)^T\nabla^2f(x)(y-x).$$

$$ \alpha$$-strong convexity 的意思是要求 $$f(y)$$ 比当我们以 identity matrix 来近似做 Hessian matrix 的时候的近似值要大。也就是对于任意的$$y$$ 找到一个凸的二次的lower bound： 

$$ f(x)  + \nabla f(x)^T(y-x) + \frac{\alpha}{2}\|x-y\|_2^2 .$$

$$\alpha$$可以看做函数 curvature 的一种度量。对于线性函数，$$\alpha$$为0。在优化中，大的 $$ \alpha$$会有更好的 convergence rate 。 因为大的 $$ \alpha$$ 表示曲率大，因此表示梯度下降的步长大。 Strongly convex 的函数一定是 strictly convex 的，但是反之不成立。 $$f(x) = x^4$$  是 strictly convex 但不是 strongly convex。 $$ f(x) = |x |$$ 是 convex，不是 strongly convex，因此也不是 strictly convex。 因此 strongly convex 函数是凸函数中有更好性质、更便于处理的一部分。


$$\alpha$$-strong convexity 有如下的性质：

1. 如果 $$ f_1(x)$$ 和 $$ f_2(x)$$ 分别是 $$ \mu_{1}$$-strongly convex 和 $$ \mu_{2}$$-strongly convex，那么 $$ f(x) = \alpha f_1(x) + \beta f_2(x)$$ 是 $$ (\alpha \mu_1 + \beta \mu_2)$$-strongly convex 的。

2. 如果 $$ f(x) $$是 $$ \alpha$$-stongly convex 的，那么 
$$ f(y) \leq f(x) + \nabla f(x)^T(y-x) + \frac{1}{2\alpha} \|\nabla f(x) - \nabla f(y)\|^2 $$.

3. 如果 $$ f(x) $$是 $$ \alpha$$-stongly convex 的，那么 
$$\left(\nabla f(x) - \nabla f(y)\right)^T(x-y) \leq \frac{1}{\alpha} \|\nabla f(x) - \nabla f(y)\|^2$$.


$$\alpha$$-strong convexity 的函数可以推出以下等价的结论（其中 $$ \mu \in [0,1]$$）：

1. 函数 $$ x \mapsto f(x) - \frac{\alpha}{2}\|x\|_2^2$$  是凸函数。

2. $$ \left(\nabla f(x) - \nabla f(y)\right)^T(x-y) \geq \alpha \|x-y\|^2 $$.

3. $$ \mu f(x) + (1-\mu) f(y) \geq f(\mu x + (1-\mu)y) + \mu (1-\mu) \frac{\alpha}{2} \|x-y\|^2$$.

4. $$\nabla^2 f(x) \succeq \alpha I$$.

- - -

## $$ \beta$$-smoothness

如果一个连续可微的函数 $$ f(x)$$ 的梯度 $$ \nabla f(x)$$ 是 $$ \beta$$-Lipschitz 的，即：

$$ \|\nabla f(x) - \nabla f(y)\| \leq \beta \|x -y\|,$$

那么我们称 $$ f(x)$$  是 $$ \beta$$-smooth 的。给定一个$$ \beta$$-smooth 的函数可以推出以下的等价的结论（其中 $$ \mu \in [0, 1]$$ 并且 $$f$$ 是凸函数）：

1. $$ 0 \leq f(y) - f(x) - \nabla f(x)^T (y-x) \leq \frac{\beta}{2} \|x-y\|_2^2$$ .

2. $$ f(x) + \nabla f(x)^T (y-x) + \frac{1}{2\beta}\|\nabla f(x) - \nabla f(y)\|^2 \leq f(y)$$.

3. $$\frac{1}{\beta}  \|\nabla f(x) - \nabla f(y)\|^2 \leq \left(\nabla f(x) - \nabla f(y)\right)^T(x-y)$$.

4. $$\left(\nabla f(x) - \nabla f(y)\right)^T(x-y) \leq \beta \|x-y\|^2$$.

5. $$\mu f(x) + (1-\mu)f(y) \geq f(\mu x + (1-\mu)y) + \frac{\mu(1-\mu)}{2\beta} \| \nabla f(x) - \nabla f(x)\|^2 $$.

6. $$\mu f(x) + (1-\mu)f(y) \leq f(\mu x + (1-\mu)y) + \frac{\mu(1-\mu)}{2\beta} \| x-y\|^2  $$.

7. $$ 0 \preceq \nabla^2f(x)\preceq \beta I $$.


如果 $$f(x) $$ 是 $$ \alpha$$-stongly convex，那么有：

$$ \left(\nabla f(x) - \nabla f(y)\right)^T(x-y) \geq \frac{\alpha\beta}{\alpha + \beta} \|x-y\|^2 + \frac{1}{\alpha + \beta} \| \nabla f(x) - \nabla f(x)\|^2.$$

函数的  $$ \beta$$-smoothness 和  $$ \alpha$$-strong convexity 有对偶的关系。如果一个函数 $$f(x)$$是 $$ \beta$$-smooth 的，那么 $$f^*(x)$$ 是 $$\frac{1}{\beta}$$-strongly convex 的，其中 $$f^*(x) = \max_{y}(y^Tx - f(y)) $$ 是函数 $$f(x)$$ 的 convex conjugate 。

- - -

## condition number

把这两个度量放一起定义函数的条件数 (condition number)为： $$Q = \frac{\beta}{\alpha} $$。$$Q$$其实是函数 $$f(x) $$的 Hessian matrix 的条件数的上限。$$Q$$ 对在 $$ f(x)$$ 上的梯度下降的速度有很大影响， $$Q$$越小的目标函数越容易收敛。



- - -

## Reference
1. Nesterov, Y. "Introductory lectures on convex optimization: a basic course. 2004."
2. Bubeck, Sébastien. "Theory of convex optimization for machine learning." arXiv preprint arXiv:1405.4980 (2014).
