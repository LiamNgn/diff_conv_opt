This repository implement a differentiable convex optimization layer. [Diffcp](https://github.com/cvxgrp/diffcp) will be used to solve for the optimality of cone program and [cvxpy](https://github.com/cvxpy/cvxpy) will be used to declare the form of the convex optimization problem.

## Some basic concept

**Parametrized convex optimization**

$$\begin{aligned}
\min_{} \quad & f_0(x;\theta)\\
\textrm{s.t.} \quad & f_i(x;\theta) \leq 0, i = 1,2,\ldots,m\\
  &Ax = B    \\
\end{aligned}$$

with variables $x \in R^n$. 

- Objective and inequality constraint $f_0,\ldots,f_m$ are convex
- Equality constraint are linear
- We want to find a value for $x$ that minimize the objective function while satisfying constraints

# Forward pass, backward pass

**Forward pass**

In order to do a forward pass, we aim to solve the optimization problems:

$$\begin{aligned}
x^\ast(\theta) = argmin \quad &f_0(x;\theta)\\
\textrm{s.t.} \quad & f_i(x;\theta) \leq 0, i = 1,2,\ldots,m\\
  &A(\theta)x = b(\theta)    \\
\end{aligned}$$ 

for every instances in the input batch. 

This parametrized convex optimization problem is in fact a mapping from the parameters to the solution. Thus with different parameters we will have different solutions. 

**Backward Pass**

Even though convex optimization problems do not have a general closed-form solution, it is possible to differentiate through convex optimization problems by implicitly differentiating their optimality conditions.[[1]](#1) 

**Implicit function theorem**

Let $f: R^p \times R^n \to R^n$ and $a_0 \in R^p,z_0 \in R^n$ be such that

- $f(a_0,z_0) = 0$, and
- $f$ is continuously differentiable with non-singular Jacobian $\partial_1 f(a_0,z_0) \in R^{n \times n}$

Then there exist open set $S_{a_0} \subset R^p$ and $S_{z_0} \subset R^n$ containing $a_0$ and $z_0$, respectively, and a unique continuous function $z^\ast:S_{a_0} \to S_{z_0}$ such that

1. $z_0 = z^\ast(a_0)$
2. $f(a,z^\ast(a)) = 0 \forall a \in S_{a_0}$
3. $z^\ast$ is differentiable on $S_{a_0}$

Using the Implicit function theorem, we get:

$$D_x f(x) = D_y g(x,f(x))^{-1}g(x,f(x))$$

which gives us an implicitly defined function $f(x) = g(x,y')$ where $y' \in \{y: g(x,y)\} = 0$

Since every convex program can be turned into a cone program and there are good methodology to differentiate through convex cone program, in this program, inspired by [[1]](#1), the original convex optimization problem will be casted into a convex cone program and differentiate to get the optimality condition. For more detail on how to differentiate a cone program, check the appendix in [[1]](#1).

**Example**:

For some example, check the [example](https://github.com/LiamNgn/diff_conv_opt/blob/main/Example.ipynb)



# References

<a id="1">[1]</a> Agrawal, Akshay, Brandon Amos, Shane T. Barratt, Stephen P. Boyd, Steven Diamond, and J. Zico Kolter. 2019. “Differentiable Convex Optimization Layers.” CoRR abs/1910.12430. arXiv: 1910.12430. http://arxiv.org/abs/1910.12430.

<a id="2">[2]</a> [Differntiable Convex Optimization Layers](https://locuslab.github.io/2019-10-28-cvxpylayers/)







## 
