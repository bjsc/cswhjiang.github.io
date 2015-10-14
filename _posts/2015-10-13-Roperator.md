---
layout: post
title: R-operator in Neural Networks
comments: True
---

## What is R-operator?
The R-operator in neural network is used to compute the product of Hessian matrix $$H$$ and a vector $$v$$. The time complexity is $$O(n)$$, which is lower than ordinary matrix vector product. Because it use back-propagation without computing the results directly.


We use $$w$$ to denote the parameters and $$g$$ to denote gradient vector, and the Hessian matrix could be expressed as $$H = \nabla(g)$$. Therefore

$$
\begin{align}
\nabla f(w + \Delta w) &= \nabla f(w) + \nabla^2f(w) \Delta w + O(\|\Delta w\|^2) \\
&=\nabla f(w) + H \Delta w + O(\|\Delta w\|^2)
\end{align}
$$

If we define $$rv = \Delta w$$, we will get

$$
Hv = \frac{\nabla f(w + rv) - \nabla f(w)}{r} + O(r)
$$

It is an easy way compute $$Hv$$, but it is not stable. Hence R-operator is needed here. The R-operator [1] with respect to $$v$$ is defined as

$$
R_{v}\{f(w)\} = \frac{\partial{f(w+r v)}}{\partial{r}} \big|_{r=0}
$$

Hence $$Hv = R_{v} \{\nabla E(w)\}$$. It is obvious that $$R_v\{w\} = v$$.
Moreover, R-operator has following properties:

$$
\begin{align}
R\{ cf(w)\} &= c R\{ f(w)\} \\
R\{ f(w) + g(w)\} &= R\{ f(w)\} + R\{ g(w)\} \\
R\{f(w)g(w) \} &= R\{f(w) \} g(w) + g(w)R\{ f(w)\} \\
R\{f(g(w))\} &= f'(g(w)) R\{ g(w)\} \\
R\left\{ \frac{d f(w)}{dt} \right\} &= \frac{d R\{f(w)\}}{dt}
\end{align}
$$

Our goal is to use the above properties to compute $$R_{v} \{\nabla E(w)\}$$, the method use is back-propagation.


## How to compute $$Hv$$ using R-operator

### Back-propagation

We use the same notations with  [UFLDL](http://deeplearning.stanford.edu/wiki/index.php/Neural_Networks) . We denote  $$a^{(l)}$$ as the input of the $$l$$-th layer, $$h^{(l+1)}$$ as the activation function and $$s_l$$ as the number of nodes in layer $$l$$. Hence,  $$a^{(1)}$$ is the original input $$x$$. The forward step of the $$l$$-th layer can be expressed as 


$$
\begin{align}
z_j^{(l+1)} &= \sum_i w_{ji}^{(l)}a_i^{(l)} + b^{(l)}_j \\
a_j^{(l+1)} &= h^{(l+1)}(z_j^{(l+1)})
\end{align}
$$

Except the final layer, the error term $$\delta^{(l)}$$ can be computed as

$$
\begin{align}
\delta^{(l)}_{i} = {h^{(l)}}'(z^{(l)}_i)\sum_j^{s_{l+1}} w_{ji}^{(l)} \delta^{(l+1)}_{j}
\end{align}
$$

For the final layer, the error term is 

$$
\begin{align}
\delta^{(l+1)}_{i} = \frac{\partial{loss}}{\partial{a^{(l+1)}_i}} {h^{(l+1)}}'(z^{(l+1)}_i)
\end{align}
$$

With error terms, we can express the gradient by

$$
\begin{align}
\frac{\partial{E}}{\partial{W^{(l)}_{ij}}} &= a^{(l)}_j\delta^{(l+1)}_i \\
\frac{\partial{E}}{\partial{b^{(l)}_{i}}} &= \delta^{(l+1)}_i
\end{align}
$$

### R-operator

According the computation of gradients, we have 

$$
\begin{align}
R_{v}\left\{\frac{\partial{E}} {\partial{W^{(l)}_{ij}}} \right\}&＝R_v \{a^{(l)}_j\delta^{(l+1)}_i \} \\
&= R_v \{a^{(l)}_j \} \delta^{(l+1)}_i + a^{(l)}_j R_v \{\delta^{(l+1)}_i \} \\
R_{v} \left\{\frac{\partial{E}} {\partial{b^{(l)}_{i}}} \right\}&＝ R_v \{\delta^{(l+1)}_i \}
\end{align}
$$

For the first layer $$a^{(1)} = x$$ and $$R_v \{a^{(l)}_j \} =0$$. In order to compute $$Hv$$, we need to know $$R_{v}\{a^{(l)}_j \}$$  and $$R_v \{\delta^{(l)}_i \}$$. By applying R-operator to the terms in back-propagation, we have 

$$
\begin{align}
R_v\{z_j^{(l+1)}\} &= R_v\{  \sum_i w_{ji}^{(l)}a_i^{(l)} + b^{(l)}_j \}\\
&= \sum_i  R_v\{w_{ji}^{(l)} \}a_i^{(l)} + \sum_i w_{ji}^{(l)} R_v\{ a_i^{(l)}\}  + R_v\{ b^{(l)}_j \}\\
&= \sum_i v_{ji}^{(l)} a_i^{(l)} + \sum_i w_{ji}^{(l)} R_v\{ a_i^{(l)}\} + v^{(l)}_j\\
R_v\{a_j^{(l+1)}\} &= R_v\{ h^{(l+1)}(z_j^{(l+1)})\}  \\
&= {h^{(l+1)}}'(z_j^{(l+1)}) R_v\{  z_j^{(l+1)}\} \\
R_v\{\delta^{(l)}_i\} &= R_v \left\{ {h^{(l)}}'(z^{(l)}_i)\sum_j^{s_{l+1}} w_{ji}^{(l)} \delta^{(l+1)}_{j} \right\} \\
&= {h^{(l)}}''(z^{(l)}_i) R_v\{z^{(l)}_i\}   \sum_j^{s_{l+1}} w_{ji}^{(l)} \delta^{(l+1)}_{j} \\
& \quad + {h^{(l)}}'(z^{(l)}_i)\sum_j^{s_{l+1}} v_{ji}^{(l)} \delta^{(l+1)}_j  +  {h^{(l)}}'(z^{(l)}_i)\sum_j^{s_{l+1}} w_{ji}^{(l)} R_v \left\{\delta^{(l+1)}_j \right\} 
\end{align}
$$


For the final layer, 
$$
\begin{align}
& R_{v} \{ \delta^{(l+1)}_{i} \} \\
=& R_{v} \left\{ \frac{\partial{loss}}{\partial{a^{(l+1)}_i}} {h^{(l+1)}}'(z^{(l+1)}_i) \right\} \\
=& \frac{\partial{loss}}{\partial{a^{(l+1)}_i}} R_{v} \left\{ {h^{(l+1)}}'(z^{(l+1)}_i) \right\} 
+ R_{v} \left\{ \frac{\partial{loss}}{\partial{a^{(l+1)}_i}} \right\} {h^{(l+1)}}'(z^{(l+1)}_i) \\
=&  \frac{\partial{loss}}{\partial{a^{(l+1)}_i}} {h^{(l+1)}}''(z^{(l+1)}_i)  R_{v} \left\{ z^{(l+1)}_i \right\} + \frac{\partial^2{loss}}
{  \partial{ a_i^{(l+1)} }^2    } 
R_{v} \left\{   a^{(l+1)}_i   \right\} {h^{(l+1)}}'(z^{(l+1)}_i)
\end{align}
$$


This is a general form. There is no activation function in the last layer. If it is the case, just set $$h''=0, h'=I$$.


Therefore, each layer pass $$R_v\{ a^{(l)}\}$$ (for the first layer, it is just the zero vector) to next layer in the forward step, and pass $$R_v\{ \delta^{(l)}\}$$ to the previous layer in the backward step. They depend on $$R_v\{ z^{(l)}\}$$. Following the above procedure, we can compute $$Hv$$, which is similar to the computation of gradient.



### Example: Log-softmax layer
It is easy to get the steps for linear layer. Here, we show how to obtain the forward and backward steps of LogSoftMax layer. For the LogSoftMax layer,  the input is denoted as $$z_j$$ and the output $$a_i$$ is computed by 

$$
a_i = \log p_i =\log \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Therefore, 

$$
\begin{align}
R\{a_i\} &= R\left\{\log \frac{e^{z_i}}{\sum_j e^{z_j}}\right\}  \\\\
&= \frac{\sum_j e^{z_j}}{e^{z_i}} R\left\{ \frac{e^{z_i}}{\sum_j e^{z_j}} \right\} \\
R\left\{ \frac{e^{z_i}}{\sum_j e^{z_j}} \right\} &= e^{z_i} R\left\{\frac{1}{\sum_j e^{z_j}} \right\} + R\{e^{z_i}\}\frac{1}{\sum_j e^{z_j}} \\
&= -\frac{e^{z_i}}{\left(\sum_j e^{z_j}\right)^2} R\{\sum_j e^{z_j}\} + e^{z_i} R\{z_i\}\frac{1}{\sum_j e^{z_j}} \\
&=-\frac{e^{z_i}}{\left(\sum_j e^{z_j}\right)^2} \left(\sum_j e^{z_j} R\{z_j\} \right) +  R\{z_i\}\frac{e^{z_i}}{\sum_j e^{z_j}}
\end{align}
$$

Hence, we have 

$$
\begin{align}
R\{a_i\} &=-\frac{1}{ \sum_j e^{z_j} } \left(\sum_j e^{z_j} R\{z_j\} \right) +  R\{z_i\}
\end{align}
$$

For the backward step, 

$$
\delta_i^{(l)} = \sum_j\frac{\partial{a_j}}{\partial{z_i}}\delta_j^{(l+1)}
$$

Since we know 

$$
\begin{align}
\frac{\partial{a_j}}{\partial{z_i}} = -p_i \\
\frac{\partial{a_i}}{\partial{z_i}} = 1-p_i \\
\end{align}
$$

and 

$$
\begin{align}
R\{p_i\} &=R\{ \frac{e^{z_i}}{\sum_j e^{z_j}}\} \\
&=\frac{R\{e^{z_i}\}}{\sum_j e^{z_j}} + e^{z_i} R\{\frac{1}{\sum_j e^{z_j}}\} \\
&=e^{z_i} R\{z_i\} \frac{1}{\sum_j e^{z_j}} -\frac{ e^{z_i}}{(\sum_j e^{z_j})^2} R\{\sum_j e^{z_j}\}  \\
&=e^{z_i} R\{z_i\} \frac{1}{\sum_j e^{z_j}} -\frac{ e^{z_i}}{(\sum_j e^{z_j})^2} \sum_jR\{ e^{z_j}\}  \\
&=e^{z_i} R\{z_i\} \frac{1}{\sum_j e^{z_j}} -\frac{ e^{z_i}}{(\sum_j e^{z_j})^2} \sum_je^{z_j} R\{ z_j\}
\\
&=p_i R\{z_i\}  - p_i \sum_j p_j R\{ z_j\}  \\
\end{align}
$$


Finally, we have 

$$
\begin{align}
& R\{\delta_i^{(l)}\}\\
=&  R\left\{ \sum_j\frac{\partial{a_j}}{\partial{z_i}}\delta_j^{(l+1)} \right\} \\
=& \sum_j R\left\{\frac{\partial{a_j}}{\partial{z_i}}\delta_j^{(l+1)}  \right\} \\
=&\sum_j \left( R\left\{\frac{\partial{a_j}}{\partial{z_i}}  \right\}\delta_j^{(l+1)} + \frac{\partial{a_j}}{\partial{z_i}} R\left\{\delta_j^{(l+1)}  \right\} \right) \\
=&\sum_j \left( -R\{p_i\}\delta_j^{(l+1)} + \frac{\partial{a_j}}{\partial{z_i}} R\left\{\delta_j^{(l+1)}  \right\} \right) \\
=&-R\{p_i\}\sum_j  \delta_j^{(l+1)} + \sum_j \left(\frac{\partial{a_j}}{\partial{z_i}} R\left\{\delta_j^{(l+1)}  \right\} \right) \\
=&-R\{ \frac{e^{z_i}}{\sum_j e^{z_j}}\}\sum_j  \delta_j^{(l+1)} + \sum_j \left(\frac{\partial{a_j}}{\partial{z_i}} R\left\{\delta_j^{(l+1)}  \right\} \right) \\
=&- p_i\left( R\{z_i\}  -  \sum_k p_k R\{ z_k\}\right)\sum_j  \delta_j^{(l+1)} + \sum_j \left(\frac{\partial{a_j}}{\partial{z_i}} R\left\{\delta_j^{(l+1)}  \right\} \right) \\
=& -p_i\left( R\{z_i\}  -  \sum_k p_k R\{ z_k\}\right)\sum_j  \delta_j^{(l+1)} - p_i \sum_j  R\left\{ \delta_j^{(l+1)} \right\} + R\left\{\delta_i^{(l+1)} \right\} 
\end{align}
$$


You can find torch implementation in the `rop` branch of [nn](https://github.com/cswhjiang/nn) on my github.

## Reference
1. Pearlmutter, Barak A. "Fast exact multiplication by the Hessian." Neural computation 6.1 (1994): 147-160.
2. Bishop, Christopher M. Pattern recognition and machine learning. springer, 2006. (Chapter 5.4.6)
