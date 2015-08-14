---
layout: post
title: Optimization Methods for Large Scale Machine Learning
comments: True
---

这是关于机器学习领域的大规模优化的论文列表，为了方便查阅和回忆。 这个列表不一定完整，仅仅包含我认为重要的论文。

在机器学习领域，我们关心的问题表示如下

$$
\min_{x} \frac{1}{n} f_i(x) + R(x)
$$

其中 $$ f(x)$$ 通常是光滑的，而  $$ R(x)$$ 通常不光滑。在这个note中为了方便用 $$ f(x)$$ 表示 $$ \frac{1}{n}\sum_{i}f_i(x)$$。 在机器学习中，这个问题的 $$ n$$ 和 $$ p$$ 通常都很大，我们主要关心它的基于梯度的 online learing 和 mini-batch 解法，以及 coordinate descent 方法。 full gradient 的方法通常慢，但是 online 方法的很多新思想是从 full gradient 的方法中来的，因此 full gradient 的方法也会在这个 note 中提到。

## 2000 之前

- Full Gradient (FG) 的方法可以追溯到 Cauchy 1847 年的论文[29]。对于凸函数，FG 的 convergence rate 是 $$O(1/t)$$， 强凸函数可以达到线性收敛 $$O(\rho^t) $$，其中 $$\rho < 1 $$。  

- Stochastic Gradient Descent (SGD) 于1951和1952年在文献[15，16]中提出。SGD 随机选择一个 $$ f_{i_{t}}$$  来作为梯度的近似 （$$ R(x)\equiv 0 $$ ），然后更新 $$ x_t = x_{t-1} - \eta_t \nabla f_{i_t}(x_{t-1})$$。 步长 $$ \eta_{t}$$ 需要在 步长$$t $$ 无穷大的时候趋近于0。 SGD 的 convergence rate 是  $$ O(1/\sqrt{t})$$。 [15] 中证明了只要 $$ \eta_t$$ 满足如下的条件：

$$
\begin{align}
\sum_{t} \eta_t = \infty \  \textrm{and } \ \sum_{t} \eta_t^2 < \infty
\end{align}
$$ 

SGD就可以收敛。SGD 每次迭代的时间和空间的复杂度都是 $$O(d) $$.

- Conjugate Gradient (CG)：共轭梯度是 Magnus Hestenes 和 Eduard Stiefel 在1952年提出的[31]。CG 用来解 $$ Ax = b $$ 的，不错的材料是 [25]，ML 中不是经常用到。

- Conditional Gradient Descent （CGD）由 Frank 和 Wolfe 在1956年提出的 [26]， 它用来解一个问题( $$ R(x)\equiv 0 $$)的步骤如下

$$
\begin{align}
y_t  &= argmin_{y \in \mathcal{X}} \nabla f(x_t)^T y \\
x_{t+1} &= (1-\gamma_t) x_t - \gamma_t y_t
\end{align}
$$

这个方法的好处是 projection-free、norm-free 以及 sparse iterates。

- Proximal Method [23]： Proximal operator 最早由 Moreau 于1960年代提出，本来是作为投影的扩展，[23]中可以找到相关的历史。我们通常用的一般都是 proximal gradient method 来处理带有不光滑项的目标函数， 它可以理解为将光滑的函数 $$ f(x) $$二阶近似，只是这个近似的 Hessian 矩阵是 $$ \frac{1}{2\eta}I $$， 然后解这个近似的问题，这个问题通常更简单，有的有解析解。它的步骤如下

$$
x_{t+1} = argmin_{x} \   \nabla f(x_t)^T x  + \frac{1}{2\eta} \|x- x_t\|^2+   R(x)
$$

这个步骤也有另一种用 proximal operator 的写法

$$
x_{t+1} = prox_{\eta R} (x_t - \eta \nabla f(x_t))
$$

- L-BFGS 于1980年在文献 [17] 中提出，一开始叫 SQN，在[18]中分析。BFGS 是一种  qusi-Newton 方法， BFGS 用近似的方法来计算 Hessian matrix 的逆。BFGS的原理是要使得二阶近似的导数要和原来的函数相匹配。比如在有了新的 $$ x_{t+1}$$ 时用到的二次近似 $$ m_{t+1}(p) = f(x_{t+1}) + \nabla f(x_{t+1})^T p + \frac{1}{2}p^TB_{t+1} p$$，其中 $$ B_{t+1}$$ 是 $$ \nabla^2 f(x_{t+1})$$ 的近似。BFGS要求

$$
\begin{align}
\nabla m (0) &= \nabla f(x_{t+1}) \\
\nabla m (x_{t} - x_{t+1}) &= \nabla f(x_{t}) \\
\end{align}
$$

其中 $$ \nabla m $$ 是对 $$ p$$ 的导数。因此 $$ B_{t+1} (x_{t+1} - x_t) = \nabla f(x_{t+1}) -\nabla f(x_{t}) $$ 。就有了 [19] 第八章的迭代算法。 BFGS 的 convergence rata  是 superlinear (Newton  method 是 quadratic )， 每个迭代的复杂度是 $$ O(d^2)$$。 LBFGS 不用存一个大的矩阵 $$ B$$， 它存前 $$ m$$  个迭代的 $$s_t = x_{t+1} - x_{t}$$ 和 $$y_t = \nabla f_{t+1} - \nabla f_{t} $$, lbfgs 的矩阵 $$ B$$ 满足 $$ B_{t}y_j = s_j$$， $$ j = t-1, \cdots, t-m$$。这称为 quasi-Newton equation (rule)。

- Nesterov’s Accelerated Gradient (NAG) Method：NAG 由 Nesterov 在1983年在文献[30]中提出。简单介绍见以前的[blog](https://cswhjiang.github.io/2015/07/31/nag-and-cm/)。

- Mirror Descent [21]： MD 是由Nemirovsky和Yudin 于1983年在文献 [22] 中提出的。2003年在[21]中被证明和用更一般形式的距离函数的 proximal gradient method 是等价的。它的步骤如下

$$
x_{t+1} = argmin_{x} \ \eta f(x_t)^T x + D_{\Phi} (x, x_t)
$$

其中 $$ D_{\Phi} $$ 是用函数 $$\Phi(x) $$ 定义的 Bregman divergence。关于 Mirror descent 总结的很好的材料是[张歆华](http://users.cecs.anu.edu.au/~xzhang/)的[note](http://users.cecs.anu.edu.au/~xzhang/teaching/bregman.pdf)。这个note的有些结论是来自 [24]. 

- SGD with momentum：1988年 Hinton在文献 [13] 用如下的步骤来求解。

$$
\begin{align}
\Delta x_t &= \gamma \Delta x_{t-1} - \eta_t \nabla f_{i_{t-1}}(x_{t-1}) \\
x_t &= x_{t-1} + \Delta x_t
\end{align}
$$

如果一个维度上的梯度的符号一直变化，那么在这个维度上的更新会被减慢；如果一个维度上的梯度符合很稳定，那么在这个维度上的更新会被加快。 Momentum 最早于1964年在 [20] 中提出。

- Averaged SGD [14] 据说由 Ruppert 和 Polyak 在 1980 年代后期独立发明，能找到的引用是 [14]。


## 2000-2010
- Nesterov's dual averaging，也称为 lazy mirror descent [27]， 这种方法是在 mirror descent 基础上改的： $$ \nabla \Phi(y_{t+1}) = \nabla \Phi(y_t) - \eta \nabla f(x_t) $$，因此它省了一个步骤，最终的迭代步骤如下：

$$
x_{t+1}= argmin_{x} \ \Phi(x)  + \eta \sum_{i}^{t} \nabla f(x_{i})^T x 
$$

- Natural gradient [44] 

- Forward-backward splitting （FOBOS） [28] 其实就是 proximal gradient method。


- FISTA [9] 其实是 proximal gradient method 在当正则项是 $$ \ell_1$$-norm的时候的 Nesterov 加速版本，convergence rate 是 $$ O(1/t^2)$$

- sub(L)BFGS [32] 将 (L)BFGS 的应用扩展到非光滑的函数。

- Regularized dual averging (RDA) [6] 是在2010年由 MSR 的 Lin Xiao 提出的。RDA 是一种改进的 lazy mirror descent 的方法， RDA 的步骤如下：

$$
x_{t+1} = argmin_{x} \   \frac{1}{t}\sum_{j=1}^t \nabla f(x_j)^T x  + R(x)+ \frac{\beta_t}{t}\Phi(x)
$$


- [37] SCD， ICML 2009年会议提出。

- Stochastic Accelerated GradiEnt (SAGE) [39] 是受 Nesterov 的加速方法启发的一种加速方法，它每个步骤维护三个变量（Nesterov加速维护两个）。

## 2011 
- ADAGRAD [11] 着眼于函数的 condition number， 让每个dimension有不同的 learning rate。更新过程为

$$
\Delta x_t = -\eta \   diag(G_t)^{-1/2} g_t 
$$

其中 $$ G_t = \sum_i^t g_ig_i^T $$。


- [38]

- [48] 是关于 mirror descent 的理论一点的分析。在[48] 中作者证明了只要 mirror map 的函数合适，mirror descent总会有 near-optimal regret，这样的函数一定是存在的，只是没那么容易找到。

## 2012 
- ADADELTA [12] 于2012年提出，是结合了 AdaGrad 和冲量的一种方法。据说在神经网络中效果不错。

- Stochastic Average Gradient（SAG）[4,10] 的前提是 $$ R(x)\equiv 0 $$。采用固定步长的SAG对于convex 函数的 convergence rate 是 $$ O(1/t)$$， strongly-convex 的 convergence rate 是线性收敛。SAG 记录历史的梯度，每次的梯度都是更新一个在随机选择的样本处的梯度然后求平均的梯度。作者提供了[代码](http://www.cs.ubc.ca/~schmidtm/Software/SAG.html)。

- [36] 

- feature clustering [35]

- Optimal Regularized Dual Averaging (ORDA) [49] 采用两种技术：1. 梯度的带权平均，越老的梯度权重越小。2.每次在平均的结果上又前进了一步。

## 2013
- Stochastic Dual Coordinate Ascent (SDCA) [8] 针对的是没有不光滑的正则项的目标函数，原问题定义如下：

$$
P(w) = \frac{1}{n} \sum_{i=1}^n \phi(w^T x_i) + \frac{\lambda}{2} \|w\|^2
$$

它的对偶问题为：

$$
D(\alpha) = \frac{1}{n} \sum_{i=1}^n -\phi^*(-\alpha_i) -\frac{\lambda}{2} \|\frac{1}{\lambda n} \sum_{\alpha_i x_i}\|^2
$$

用 Stochastic Coordinate Ascent 来解这个对偶问题。

- Stochastic Variance Reduced Gradient（SVRG） [1]: 一开始提出来的时候还是针对正则项是光滑函数的情况。 SVRG 用在所有数据上重新计算梯度的方法来减小梯度的方差。作者 Rie Johnson 的主页提供[代码](http://riejohnson.com/svrg_download.html) 

- [5] 是 SVRG 的扩展

- [40] 中分析了用新的框架分析 SCD 中的 non-uniform sampling 问题，并提出一个最有的对于坐标的采样分布。这篇文章有2015年的journal版。非均匀采样对于 SCD 和 SGD 都是很自然的扩展。

- Prox-full-gradient [3]: 分析了 proximal gradient method 的加速版本

- [41] 中分析了对于非光滑的目标函数的 $$\alpha$$-suffix averaging 和 polynomial-decay averaging 影响。 

- leaning rate [45]

- Stochastic Majorization-minimization [47] 是针对随机MM的优化算法。迭代用的 approximate surrogate  是当前函数和前一个迭代的加权和，可以扩展到非凸的目标函数。 

## 2014
- Prox-SDCA [7]

- Prox-SVRG [2] 是 SVRG 的扩展，它能处理带有 $$ R(x)$$ 的问题，以及用了 non-uniform 采样。

- mini-batch SGD 为了减小 SGD 的梯度的方差以及分布式 SGD 的通信代价，可以用多个样本来求梯度，但是mini-batch 的 size 如果太大会减小 convergence rate。

- Adaptive moment estimation (Adam) [33]

- SFO [34]

- Acc-Prox-SVRG [42] 是一种 mini-batch 的方法，它同时采用 Nesterov 加速和 SVRG 的 varicande reduction 的技术来加速。

- [46] 

## 2015

-  Probabilistic line search [43]：随机梯度下降中的梯度是有噪音的，因此作者 Bayesian Optimization 来解决，是当前方法的一个补充。






# Reference
1. Johnson, Rie, and Tong Zhang. "Accelerating stochastic gradient descent using predictive variance reduction." Advances in Neural Information Processing Systems. 2013.
2. Xiao, Lin, and Tong Zhang. "A proximal stochastic gradient method with progressive variance reduction." SIAM Journal on Optimization 24.4 (2014): 2057-2075.
3. Nesterov, Yu. "Gradient methods for minimizing composite functions." Mathematical Programming 140.1 (2013): 125-161.
4. Roux, Nicolas L., Mark Schmidt, and Francis R. Bach. "A stochastic gradient method with an exponential convergence rate for finite training sets." Advances in Neural Information Processing Systems. 2012.
5. Konečný, Jakub, and Peter Richtárik. "Semi-stochastic gradient descent methods." arXiv preprint arXiv:1312.1666 (2013).
6. Xiao, Lin. "Dual averaging method for regularized stochastic learning and online optimization." Advances in Neural Information Processing Systems. 2009.
7. Shalev-Shwartz, Shai, and Tong Zhang. "Accelerated proximal stochastic dual coordinate ascent for regularized loss minimization." Mathematical Programming (2014): 1-41.
8. Shalev-Shwartz, Shai, and Tong Zhang. "Stochastic dual coordinate ascent methods for regularized loss." The Journal of Machine Learning Research 14.1 (2013): 567-599.
9. Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." SIAM journal on imaging sciences 2.1 (2009): 183-202.
10. Schmidt, Mark, Nicolas Le Roux, and Francis Bach. "Minimizing finite sums with the stochastic average gradient." arXiv preprint arXiv:1309.2388 (2013).
11. Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic optimization." The Journal of Machine Learning Research 12 (2011): 2121-2159.
12. Zeiler, Matthew D. "ADADELTA: An adaptive learning rate method." arXiv preprint arXiv:1212.5701 (2012).
13. Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. "Learning representations by back-propagating errors." Cognitive modeling 5 (1988): 3.
14. Polyak, Boris T., and Anatoli B. Juditsky. "Acceleration of stochastic approximation by averaging." SIAM Journal on Control and Optimization 30.4 (1992): 838-855.
15. Robbins, Herbert, and Sutton Monro. "A stochastic approximation method." The annals of mathematical statistics (1951): 400-407.
16. Kiefer, Jack, and Jacob Wolfowitz. "Stochastic estimation of the maximum of a regression function." The Annals of Mathematical Statistics 23.3 (1952): 462-466.
17. Nocedal, Jorge. "Updating quasi-Newton matrices with limited storage." Mathematics of computation 35.151 (1980): 773-782.
18. Liu, Dong C., and Jorge Nocedal. "On the limited memory BFGS method for large scale optimization." Mathematical programming 45.1-3 (1989): 503-528.
19. Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer Science & Business Media, 2006.
20. Polyak, Boris Teodorovich. "Some methods of speeding up the convergence of iteration methods." USSR Computational Mathematics and Mathematical Physics 4.5 (1964): 1-17.
21. Beck, Amir, and Marc Teboulle. "Mirror descent and nonlinear projected subgradient methods for convex optimization." Operations Research Letters 31.3 (2003): 167-175.
22. Nemirovsky, Arkadiĭ Semenovich, and David Borisovich Yudin. "Problem complexity and method efficiency in optimization." (1983).
23. Parikh, Neal, and Stephen Boyd. "Proximal algorithms." Foundations and Trends in optimization 1.3 (2013): 123-231.
24. Bubeck, Sébastien. "Theory of convex optimization for machine learning." arXiv preprint arXiv:1405.4980 (2014).
25. Shewchuk, Jonathan Richard. "An introduction to the conjugate gradient method without the agonizing pain." (1994).
26. Frank, Marguerite, and Philip Wolfe. "An algorithm for quadratic programming." Naval research logistics quarterly 3.1‐2 (1956): 95-110.
27. Nesterov, Yurii. "Primal-dual subgradient methods for convex problems." Mathematical programming 120.1 (2009): 221-259.
28. Singer, Yoram, and John C. Duchi. "Efficient learning using forward-backward splitting." Advances in Neural Information Processing Systems. 2009.
29. Cauchy, Augustin. "Méthode générale pour la résolution des systemes d’équations simultanées." Comp. Rend. Sci. Paris 25.1847 (1847): 536-538.
30. Nesterov, Yurii. "A method of solving a convex programming problem with convergence rate O (1/k2)." Soviet Mathematics Doklady. Vol. 27. No. 2. 1983.
31. Hestenes, Magnus Rudolph, and Eduard Stiefel. "Methods of conjugate gradients for solving linear systems." (1952).
32. Yu, Jin, et al. "A quasi-Newton approach to non-smooth convex optimization." Proceedings of the 25th international conference on Machine learning. ACM, 2008.
33. Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
34. Sohl-dickstein, Jascha, Ben Poole, and Surya Ganguli. "Fast large-scale optimization by unifying stochastic gradient and quasi-Newton methods." Proceedings of the 31st International Conference on Machine Learning (ICML-14). 2014.
35. Scherrer, Chad, et al. "Feature clustering for accelerating parallel coordinate descent." Advances in Neural Information Processing Systems. 2012.
36. Scherrer, Chad, et al. "Scaling Up Coordinate Descent Algorithms for Large $\ ell_1 $ Regularization Problems." Proceedings of the 29th International Conference on Machine Learning (ICML-12). 2012.
37. Shalev-Shwartz, Shai, and Ambuj Tewari. "Stochastic methods for l1-regularized loss minimization." The Journal of Machine Learning Research 12 (2011): 1865-1892.
38. Moulines, Eric, and Francis R. Bach. "Non-asymptotic analysis of stochastic approximation algorithms for machine learning." Advances in Neural Information Processing Systems. 2011.
39. Hu, Chonghai, Weike Pan, and James T. Kwok. "Accelerated gradient methods for stochastic optimization and online learning." Advances in Neural Information Processing Systems. 2009.
40. Richtárik, Peter, and Martin Takáč. "On optimal probabilities in stochastic coordinate descent methods." arXiv preprint arXiv:1310.3438 (2013).
41. Shamir, Ohad, and Tong Zhang. "Stochastic Gradient Descent for Non-smooth Optimization: Convergence Results and Optimal Averaging Schemes." Proceedings of the 30th International Conference on Machine Learning (ICML-13). 2013.
42. Nitanda, Atsushi. "Stochastic proximal gradient descent with acceleration techniques." Advances in Neural Information Processing Systems. 2014.
43. Mahsereci, Maren, and Philipp Hennig. "Probabilistic Line Searches for Stochastic Optimization." arXiv preprint arXiv:1502.02846 (2015).
44. Amari, Shun-Ichi, Hyeyoung Park, and Kenji Fukumizu. "Adaptive method of realizing natural gradient learning for multilayer perceptrons." Neural Computation 12.6 (2000): 1399-1409.
45. Schaul, Tom, Sixin Zhang, and Yann Lecun. "No more pesky learning rates." Proceedings of the 30th International Conference on Machine Learning (ICML-13). 2013.
46. McMahan, Brendan, and Matthew Streeter. "Delay-Tolerant Algorithms for Asynchronous Distributed Online Learning." Advances in Neural Information Processing Systems. 2014.
47. Mairal, Julien. "Stochastic majorization-minimization algorithms for large-scale optimization." Advances in Neural Information Processing Systems. 2013.
48. Srebro, Nati, Karthik Sridharan, and Ambuj Tewari. "On the universality of online mirror descent." Advances in neural information processing systems. 2011.
49. Chen, Xi, Qihang Lin, and Javier Pena. "Optimal regularized dual averaging methods for stochastic optimization." Advances in Neural Information Processing Systems. 2012.
