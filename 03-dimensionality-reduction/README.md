# Dimensionality Reduction

Dimensionality reduction serves as the means to take a dataset, $XY$ and reduce it's dimensionality down to a point where it is more maneagble and computationally efficient.

If $XY$ is shape $(n, p)$, where $n$ is the numbr of samples and $p$ is the number of features within a given sample (row) vector, the goal of dimensionality reduction is to reduce $XY$ to size, $(n, k)$ such that $k$ features are < $p$ and become more manageable for a computer to work with.

You'll always lose some data when performing dimensionality reduction, but in the right situation, it can be invaluable. The goal is to reduce a given dataset $XY$ to the smallest dimensionality possible, the smallest size $k$ while minimizing the loss of quality of data in $XY$.

### Vector Projections

Vector projection is a technique where we take a vector, say $\vec{v}$, and project it onto another vector $\vec{u}$. It measures the amount by which a given vector goes in direction of another vector. When $\vec{v}$ is smaller than $\vec{u}$, the projection of $\vec{v}$ onto $\vec{u}$ is a portion of $\vec{u}$, representing the component of $\vec{v}$ that goes in direction of $\vec{u}$.

- If $\vec{v}$ and $\vec{u}$ are orthogonal, the projection of $\vec{v}$ onto $\vec{u}$ is essentially $0$.

Say $\vec{p}$ is the projected vector from $\vec{v}$ to $\vec{u}$ where $\vec{p} = k\vec{u}$ and $k$ is a scalar multiple of $\vec{u}$.

- The $\vec{p}$ can be described this way as the projected vector must be going in the same direction as $\vec{u}$

Finding $\vec{p}$ is then a matter of finding the scalar $k$.

This can be done as:

$k = \frac{\vec{v} \cdot \vec{u}}{||\vec{u}||^2}$

> *Note that a dot product tells us how much a vector aligns with another.*
> 
> *In the above, v • u tells us how much v aligns with u*

and to get the final projection, we multiply it by $\vec{u}$:

$p = k\vec{u}$

When computing the projection of a vector onto another vector, the magnitude of each vector doesn't matter, you'll get the same projection either or.

Then, once we project a vector, $\vec{v}$ onto another vector $\vec{u}$, all we need to know are the multipliers that project the original vector, $\vec{v}$ onto $\vec{u}$, and the values for the vector being projected onto, which is $\vec{u}$.

Doing so reduces the dimensionality of the datapoints, which can prove to be useful computationally if dealing with complex datasets.

Since we're getting estimates, of near equivalent norms, taking the vector projection of near similar yet different datapoints, can significantly decrease the magnitude, as the dimensionality of the given vectors increase.

### Eigenvalues & Eigenvectors

An Eigenvector of a matrix $A$, is any vector $\vec{x} \in \mathbb{R}$ where $\vec{x} ≠ \vec{0}$  that when multiplying a matrix $A$ by $\vec{x}$ you get back a scalar multiple of vector $\vec{x}$ where the multiplier of $x$ is $\lambda$, which $\in \mathbb{R}$.

*The scalar multiplier, $\lambda$, is called the eigenvalue*

Again -- eigenvalues $(\lambda)$ are the scalar multipliers of a vector $\vec{x}$ that give back the corresponding multiple of the eigenvector.

To find the eigenvector:

1. $A\vec{x} = \lambda\vec{x}$
2. $A\vec{x} = \lambda I\vec{x}$
3. $A\vec{x} - \lambda I \vec{x} = 0$
4. $\vec{x}(A - \lambda I) = 0$
5. $det(A - \lambda I ) \rightarrow polynomial$
6. $polynomial_{root} \rightarrow eigenvalue \hspace{1mm} (\lambda)$
7. Plug $\lambda$ back into $A\vec{x} = \lambda \vec{x}$ and solve the system of equations to get the eigenvectors
8. Verify answer.

$A\vec{x} = \lambda\vec{x}$

where:

- $\vec{x} ≠ \vec{0}$
- $\lambda \in \mathbb{R}$

Note that for an Eigenvector to exist, $det(A - \lambda I)$ must equal $0$ 

### Lagrange Multipliers

Given a function we want to maximize or minimize, Lagrange multipliers are the scalar multipliers that denote the maximum or minimum of a given function, under a specified constraint.

$L(X, \lambda) = f(x) + \lambda(g(x))$

where $\lambda$ is the Lagrange Multipliers, $f(X)$ is the original function we're trying to maximize or minimize, and $g(x)$ is our constraint.

**In Linear Algebra**

1. Take the derivative of $L(x, \lambda)$, see that you can compute the Lagrange as the maximum or minimum Eigenvalue
2. Find the Eigenvalue
3. The largest Eigenvalue is the Lagrange multiplier which corresponds to the maximum possible value of $f(x)$. The inverse is true for the minimum.

### Covariance and the Covariance Matrix

Covariance measures how much 2 variables change together, quantifying the degree to which changes in 1 variable, $X$, are correlated with the same changes in $Y$.

$Cov_{population}(X, Y) = \frac{1}{N} \sum_{i=1}^N (X_i - \mu_X)(Y_i - \mu_Y)$
$Cov_{sample}(X, Y) = \frac{1}{n-1} \sum_{i=1}^n (X_i - \mu_X)(Y_i - \mu_Y)$

If $X$ and $Y$ are both extremely positive or extremely negative, then they have a high degree of covariance. Otherwise, if $X$ is extremely positive and $Y$ is extremely negative, they have a very low covariance.

The covariance of a variable with itself, say $X$ as $Cov(X, X)$ is simply $Var(X)$.

A Covariance matrix for given vectors, $X$ and $Y$:

$X = \begin{bmatrix} 2, 4, 6, 8 \end{bmatrix}$
$Y = [5, 9, 7, 10]$ 

is found as:

$$
\bar{X} = \frac{2 + 4 + 6 + 8}{4} = 5, \quad \bar{Y} = \frac{5 + 9 + 7 + 10}{4} = 7.75
$$
$$
\text{Cov}(X, X) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})^2 
= \frac{1}{3} \left[(2 - 5)^2 + (4 - 5)^2 + (6 - 5)^2 + (8 - 5)^2 \right]
= \frac{1}{3} \left[9 + 1 + 1 + 9 \right] = \frac{20}{3} \approx 6.67
$$
$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y}) 
= \frac{1}{3} \left[(2 - 5)(5 - 7.75) + (4 - 5)(9 - 7.75) + (6 - 5)(7 - 7.75) + (8 - 5)(10 - 7.75) \right]
= \frac{13}{3} \approx 4.33
$$
$$
\text{Cov}(Y, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (Y_i - \bar{Y})^2 
= \frac{1}{3} \left[(5 - 7.75)^2 + (9 - 7.75)^2 + (7 - 7.75)^2 + (10 - 7.75)^2 \right]
= \frac{14.75}{3} \approx 4.92
$$
$$
\text{Covariance Matrix} = \begin{bmatrix} 
\text{Cov}(X, X) & \text{Cov}(X, Y) \\
\text{Cov}(Y, X) & \text{Cov}(Y, Y) 
\end{bmatrix} = \begin{bmatrix} 
6.67 & 4.33 \\
4.33 & 4.92 
\end{bmatrix}
$$
Covariance Matrix is symmetric, it's transpose is equal to the original matrix

The closed form of the Covariance matrix is the form which can be determined by a deterministic formula:

$Cov(X, Y) = \frac{1}{N-1} \sum_{k=1}^N(X_k - \bar{X})(Y_k - \bar{Y})^T$ 

where $X_k$ and $Y_k$ are the individual $k$-th samples of $X$ and $Y$.

You can also take the covariance of a matrix $X$ with itself with the same formula as:

$Cov(X, Y) = \frac{1}{N-1} \sum_{k=1}^N(X_k - \bar{X})(X_k - \bar{X})^T$ 

which will be important for Principal Component Analysis.

## Principal Component Analysis

Principal Component Analysis is the means for reducing the dimensionality of a dataset, say ${X_k}, k \in [1, ..., N]$ with dimensionality $D$ to dimensionality $M$ where $M < D$.

Essentially, it answers, *"How do we reduce the dimensionality of a set of samples?"*.

It aims to maximize the preservation of the original variation of $x_k$ with dimensionality $D$, while compressing $x_k$ down to dimensionality $M$, again where $M < D$.

PCA wants to find the direction at which the variance has the maximum direction.

If $V$ is a principal component, the variance of dataset $X$ projected onto $V$ is given as:

$$
Variance = \frac{1}{N} \sum_{k=1}^N(V^TX_k - V^T\bar{X})^2
$$
$$
Variance = V^T(\frac{1}{N} \sum_{k=1}^N(X_k - \bar{X})(X_k - \bar{X})^T)V
$$
$$
Variance = V^TCV
$$

where $C$ turns to be the covariance matrix.

We want to maximize $V^TCV$ subject to $||V|| = 1$

This is done through Lagrange Multipliers, $\mathcal{L}(x, \lambda)= f(x) + \lambda g(x)$, where $g(x)$ is the constraint, $1 - ||V||$,and $f(x)$ is the function we want to maximize, in this case the variability of the projection $V$ onto $X$, $f(x) = V^TCV$.

$\mathcal{L}(x, \lambda)= V^TCV + \lambda (1 - ||V||)$

We take the derivative as:

$\frac{∂\mathcal{L}}{∂v} = 2CV - 2\lambda V$

and then simplify to:

$2CV = 2\lambda V$<br>
$CV = \lambda V$ (eigenvector / value problem)<br>
$CV = \lambda I V$, where $I$ is the identity matrix.<br>
$CV - \lambda IV = 0$<br>
$V (C - \lambda I) = 0$

Where now we want to find the $det()$ of $C - \lambda I$, which will result in a polynomial.

We then find the roots of this polynomial to find the largest eigenvalue ($\lambda$) which corresponds to the Lagrange Multiplier that maximizes the the variability of the projected data whilst having the constraint of $||V|| = 1$.

This Lagrange Multiplier is used to then find the eigenvector which serves as the principal component $(P)$ for reducing the dimensionality of the original data vector, $X_k$

To reduce it, all we do is apply a Vector Projection as:

$K = \frac{XP}{||P||^2}$<br>
$X_{kPCA} = PK$ 

where $K$ is the multiplier of the principal component $P$ where their multiplication represents the projection of the vector $X$, as $X_{kPCA}$.

In practice, when reducing the total number of features / dimensionality of a dataset,

- The number of dimensions $=$ number of features
  - You want to reduce the number of features to `n_components`, whilst keeping the "important" ones.
- The matrix, $V$, should be shape of (`samples`, `n_components`) if the original $X$ was shape (`samples`, `features`). Vice versa.
- The row of the matrix, $V$, as $V_k$ should be orthonormal, meaning each row is orthogonal to another and the $L_2$ norm / magnitude of each row is equal to $1$, signalling a unit vector.

> *PCA was confusing to me at first. If it still is for u, check [this out](https://www.youtube.com/watch?v=dhK8nbtii6I&t=442s).*