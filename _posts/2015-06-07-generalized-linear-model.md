---
layout: post
title: Generalized linear models
comments: True
---


广义线性模型（Generalized linear models，不要和 General Linear Models 搞混）是 John Nelder 和 Robert Wedderburn 提出的 [1]。在给定一个训练数据集 $$D =\{(x_1,y_1), \cdots, (x_n,y_n) \}$$，要学习一个假设 $$h(x)$$ 能预测 $$y$$ 的值。在线性模型中 $$h(x) = x^T\beta$$。在广义线性模型中 $$h(x) = f(x^T \beta)$$，其中 $$f(\cdot)$$ 是非线性函数，称作激活函数（activation function），也称作响应函数（response function），它的反函数叫做连接函数（link function）。函数 $$f$$ 由给定 $$x$$ 时 $$y$$ 的分布决定。

- - -

## Exponential Family Distributions

### Formulations
指数族分布是指有着如下形式的分布

$$
\begin{align}\label{exponential_family_distributions}
 p(y;\eta) = b(y)\exp\{ \eta^T T(y) - a(\eta)\}.
\end{align}
$$

其中

- $$\eta$$ 叫做自然参数（也叫指数参数）
- $$T(y)$$ 是充分统计量 （很多时候 $$T(y) = y$$，这时候的分布叫做经典形式（canonical form），这时候的 $$\eta$$ 叫做经典参数）。
- $$a(\eta)$$ 是 log-partition function (也叫 normalization factor, cumulant generating function) 。它使得 $$p(y; \eta)$$ 是个分布。
- $$b(y)$$ non-negative base measure (很多时候为 1).


当给定 $$T, a, b$$ 的时候，参数 $$\eta$$ 就确定了一族的分布，不同的 $$\eta$$ 就定义了在这一族中不同的分布。 很多分布都是指数族分布，比如
Gaussian, Multionmial, Dirichlet, Poisson, Gamma 分布等。不是指数族分布的分布包括: Cauchy, uniform 等。


### Properties

- 充分统计量 $$T(y)$$ 的维度由自然参数 $$\eta$$ 的个数决定。
- 指数族分布的乘积依然是指数族分布，但是可能是没有归一化的。
- $$a(\eta)$$ 称作累积发生函数（cumulant generating function，注意不是 Moment-Generating Function），　$$a(\eta)$$　有如下的性质：

$$
\begin{align}
 E(T(y)) &= \frac{d a(\eta)}{ d \eta} \nonumber \\
 Var(T(y)) &= \frac{d^2 a(\eta)}{ d \eta^2} \nonumber
\end{align}
$$

- Log-partition function $$a(\eta)$$ 以及一阶导数都是凸函数。
-  能使得 $$a(\eta) < \infty $$ 成立的 $$\eta$$ 的集合称作 natural parameter space.
-  每个指数族分布都有共轭分布。



### Examples

#### Gaussian Distribution
高斯分布的概率密度函数为

$$
\begin{align}
 p(x) &= \frac{1}{\sqrt{2\pi \sigma^2}} \exp\{ -\frac{(x-\mu)^2}{2\sigma^2}\} \nonumber \\
 &= \frac{1}{\sqrt{2\pi \sigma^2}} \exp\{ -\frac{x^2}{2\sigma^2} + \frac{\mu x}{\sigma^2} - 
\frac{\mu^2}{2\sigma^2} \} \nonumber \\
 &= \frac{1}{\sqrt{2\pi}} \exp\{ \log(\sigma) -\frac{x^2}{2\sigma^2} + \frac{\mu x}{\sigma^2} - 
\frac{\mu^2}{2\sigma^2} \} \nonumber
\end{align}
$$

因此

$$
\begin{align}
 \eta &= \left( \begin{array}{c}
\mu/\sigma^2 \\
-1/(2\sigma^2)\\
\end{array} \right)  \nonumber \\
 T(x) &= \left( \begin{array}{c}
x \\
x^2\\
\end{array} \right)  \nonumber \\
 a(\eta) &=  -\log(\sigma) + \frac{\mu^2}{2\sigma^2} = 
-\frac{\eta_1^2}{4\eta_2}-\frac{1}{2} \log(-2 \eta_2)\nonumber \\
 b(x) &=   \frac{1}{\sqrt{2\pi}} \nonumber 
\end{align}
$$


#### Multivariate Gaussian Distribution
多变量高斯分布的概率密度函数为

$$
\begin{align}
 p(x) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\{-\frac{1}{2} (x-u) \Sigma^{-1} (x-u) \}
\end{align}
$$

$$
\begin{align}
 \eta &= \left( \begin{array}{c}
\Sigma^{-1}\mu \\
-\frac{1}{2} \Sigma^{-1}\\
\end{array} \right)  \nonumber \\
 T(x) &= \left( \begin{array}{c}
x \\
xx^T\\
\end{array} \right)  \nonumber \\
 b(x) &=   (2\pi)^{-\frac{D}{2}} \nonumber 
\end{align}
$$

#### Bernoulli Distribution
Bernoulli distribution 可以写作如下的形式：

$$
\begin{align}
 p(y;\phi) &= \phi^{y}(1-\phi)^{1-y} \nonumber \\
 &= \exp\{y \log(\phi) + (1-y)\log(1-\phi)\} \nonumber \\
 &= \exp\{ \log(\frac{\phi}{1-\phi})y + \log(1-\phi)\}
\end{align}
$$

根据此形式，我们可以得到：

$$
\begin{align}
\eta &= \log(\frac{\phi}{1-\phi}) \nonumber \\
 T(y) &= y \nonumber \\
 a(\eta) &= -\log(1-\phi) =\log(1+e^{\eta}) \nonumber \\
 b(y) &= 1 \nonumber 
\end{align}
$$

因此 $$ \phi = \frac{1}{1-e^{-\eta}}$$。

#### Multinomial Distribution

$$
 \begin{align}
  p(x ; \mu) &= \Pi_{k=1}^{M} \mu_{k}^{x_k} = \exp\{\sum_{k=1}^M x_k \log{\mu_k} \} \\
  & = \exp\{\eta^T x + \log(1+\sum_{k=1}^M \eta_k)^{-1} \}
 \end{align}
 $$

其中

$$
\begin{align}
 \mu_{k} = \frac{\exp(\eta_k)}{1+\sum_j \exp(\eta_j)}
\end{align}
$$

- - -

##  Generalized Linear Models

我们需要做下面假设（来自Andrew Ng的讲义）：

- $$y \mid x ; \theta \sim \textrm{ExponentialFamily}(\eta)$$。在给定 $$x$$ 和 $$\theta$$ 的时候，$$y$$ 服从指数族分布，这个指数族分布的参数是 $$\eta$$。
- 自然参数 $$\eta$$ 和输入是线性关系 $$\eta = x^T \beta$$（如果 $$\eta$$ 是向量，那么 $$\eta_i = x^T \beta_i$$）
- 输出 $$h(x) = E_{y \mid x}[y] = \frac{d a(\eta)}{d \eta}$$。



从第二个假设能看出来 $$\theta$$ 通过自然参数 $$\eta$$ 作用于这个分布，$$\theta$$ 被隐式地包含于函数 $$a(\eta)$$ 中。如果 $$y$$ 是标量，那么这个指数族分布可以表示为以下的形式：

$$
\begin{align}
  p_{\beta}(y\mid x) = b(y)\exp\{ x^T \beta y - a(x^T \beta)\}.
\end{align}
$$

如果 $$y$$ 是一个向量，那么可以表示为如下的形式

$$
\begin{align}
 p_{B}(y \mid x) = b(y)\exp\{ y^T(B x)- a(B x)\}.
\end{align}
$$


在第三个假设中 $$h(x)$$ 是 $$\eta$$ 的函数，其形式由具体的分布决定。如果定义 mean parameter $$\theta$$，它满足 $$\theta = E[T(y)]$$。也可以将指数族函数写成如下的形式

$$
\begin{align} 
 p(y;\eta) = b(y)\exp\{ \eta(\theta)^T T(y) - a(\eta(\theta))\}.
\end{align}
$$



GLM中包含一下三个部分：

- 随机部分（即响应变量 $$y$$），这是 GLM 里面唯一的含有随机的成分的地方。在给定 $$x$$ 后 $$y$$ 服从经典指数族分布（$$T(y) = y$$）。注意有些分布不能写成经典形式，比如：LogNormal 分布。因此不会有条件分布是 LogNormal 的 GLM。
- 系统部分。这一部分是所有的 GLM 共同的部分。这一部分定义了协变量 $$x$$ 通过将自然参数 $$\eta =x^T \beta$$ 来进入 GLM。
- 连接部分。连接部分将 $$x^T \beta$$ 和 mean parameter $$\theta$$ 通过一个单调可微函数 $$g(\cdot)$$ 连接起来：

$$
 \begin{align}
  g(\theta) &= x^T \beta \nonumber \\
 E[y] &= \theta = g^{-1}(x^T \beta) \nonumber 
 \end{align}
 $$

函数 $$g(\cdot)$$ 称作连接函数（也就是上面公式中的 $$\eta(\theta)$$）， $$g^{-1}(\cdot)$$ 称作响应函数（即一开始提到的　$$f(\cdot)$$， 也即是 $$\nabla_{\eta}a(\eta)$$）。连接函数是 $$\textrm{mean parameter} \mapsto \textrm{natural parameter}$$，而响应函数是 $$\textrm{natural parameter}　\mapsto  \textrm{mean parameter}$$。


### Training

在给定一个训练数据集 $$D = \{(x_1,y_1), \cdots, (x_n,y_n) \}$$，要计算在测试数据集上的 $$E_{y \mid x}[y]$$，我们需要估计 $$\beta$$。可以通过最大化 log-likelihood 来求解：

$$
\begin{align}
 l(\beta \mid D) &= \log\left(  \Pi_{i=1}^{n} b(y_i)\exp\{ \eta_i^T y_i - a(\eta_i)\} \right) 
\nonumber \\
&= \log\left(  \Pi_{i=1}^{n} b(y_i)\exp\{ y_i (x_i^T \beta) - a(x_i^T \beta)\}  \right) \nonumber \\
&= \sum \log (b(y_i)) + \beta^T \sum y_i x_i  - \sum a(\beta^T x_i) 
\end{align}
$$

由于函数 $$a(\cdot)$$  是凸函数，因此可以通过 Newton-Raphson 方法来找到最优解：

$$
\begin{align}
 \beta_{t+1} =\beta_{t} - [H(l(\beta))]^{-1} \nabla_{\beta} l(\beta)
\end{align}
$$


### Examples of GLM

#### Ordinary Least Squares

$$
\begin{align}
y \mid x \sim N(x^T \beta, \sigma^2)
\end{align}
$$

log-likelihood 表示如下

$$
\begin{align}
  l(\beta |D) &= \log \left( \Pi_{i} \frac{1} {\sqrt{2 \pi \sigma^2}} \exp \left\{-\frac{(y_i-  x_i^T \beta)^2}{2\sigma^2}\right\} \right) \nonumber \\
&= n \log(\frac{1}{\sqrt{2 \pi \sigma^2}})  - \sum_{i}  \frac{(y_i- x_i^T \beta)^2}{2\sigma^2} 
\end{align}
$$

#### Logistic Regression

$$
y \mid x \sim \textrm{Bernoulli}(\phi)
$$
根据　$$\eta =  \log(\frac{\phi}{1-\phi}) = \beta^T x $$，可以得到 $$\phi = \frac{1}{1+e^{-\beta^T x}}$$ 也就是　$$h(x)$$。

log-likelihood 表示如下

$$
\begin{align}
  l(\beta \mid　D)  &= \log \left( \Pi_{i}  \exp\left\{ y_i\cdot \beta^T x_i  + \log(\frac{e^{-\beta^T x}}{1+e^{-\beta^T 
x}})\right\}  \right) \\
&= \sum_i y_i\cdot \beta^T x_i  -  \log (1+e^{\beta^T 
x})
\end{align}
$$
这个目标函数和　$$y$$ 用　$${-1,+1}$$　做类标（即常见的 logistic loss）是等价的。
 
- - -

## Reference
1. J. A. Nelder and R. W. M. Wedderburn. Generalized linear models. Journal of the Royal Statistical Society. Series A (General), 135(3):pp. 370–384, 1972.
2. Stephen Senn and John Nelder. A conversation with john nelder. Statistical Science, 18(1):pp. 118–131, 2003.
