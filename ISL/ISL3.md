# 3 | Linear Regression

## Simple Linear Regression

- A means tor estimating $Y$ given a single input coefficient $X$.

<div align = 'center'>

$Y ≈ \beta_0 + \beta_1X$
</div>

*notation ~* may say that we are regressing $Y$ onto $X$, i.e., we can regress `tv` onto `sales` as:

<div align = 'center'>

`sales` $≈ \beta_0 + \beta_1$`tv`
</div>

$\beta_0$ is the intercept while $\beta_1$ are the parameters / coefficients of the linear model.

### 3.1.1 Estimating Coefficients

If we had $X$ be the `tv` advertisements and $Y$ be the total `sales`, the job is to find estimates of coefficients $\beta_0$ and $\beta_1$, which are able to map $X$ to $Y$ as accurately as can be, so that $y_i ≈ \beta_0 + \beta_1$ for $i = 1 \rightarrow n$.

Closeness can be measured and minimzed in a variety of ways, but is more commonly done through the *least squares* criterion.

If $\hat{y_i} = \hat{\beta_0} + \hat{\beta_1}x_i$, is the prediction of $Y$ ast the $ith$ value of $X$, then $e_i = y_i = \hat{y_i}$ is the residual for the prediction, where the residual is the difference between the true value and the predicted value.

The residual sum of squares is then defined as:

$RSS = e_1^2 + e_2^2 + e_3^2 + ... e_n^2$ 

Then the minimizer for the coefficients are:

$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$

$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$

where the $\bar{x}$ and $\bar{y}$ are the means of the samples.

> Note that this is purely for simple lienar regression as of yet.

### 3.1.2 Assessing Accuracy of Coefficient Estimates

Given that the true estimate for $Y$ is given by:

<div align = 'center'>

$Y = f(x) + \epsilon$

</div>

The $\epsilon$ captures the unknown variation, arising due to a lack of explainability from the captured features $x$. It represents the irreducbile error that isn't representable by the function $f$ arising from a lack of datapoints $x$.

The true population regression line is always unknown, as we don't know the true representation between $X$ and $Y$ prior, but finding the optimal regression line and coefficients $\beta_i$ can be done better with the more data we're able to accumulate.

The closer we get to representing all possible samples with our set of data, the better the regression line will fit.

A means to comparing a least squraes line and a theoretical, true population regression line, can be through the standard error:

$Var(\hat{\mu}) = SE(\hat{\mu})^2 = \frac{\sigma^2}{n}$

where $\sigma$ is the standard deviation of each sample $y_i$ within $Y$. It tells us how much $\hat{\mu}$ deviates from the true $\mu$ of the entire population

This standard error can be computed for the parameters $\hat{\beta_0}$ and $\hat{\beta_1}$, as:

$SE(\hat{\beta}_0)^2 = \sigma^2 \left[ \frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2} \right]$

$SE(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$

## 3.2 Multiple Linear Regr ession

In a situation where we have multiple predictors $X$ up until $X_p$, to predict $Y$, one may be able to fit individual $p$ regression lines, for each predictor $X$

But this is unsatisfactory as:

- Each regression line doesn't consdier the other $X_{p-1}$ datapoints, thereby the association between how each impacts each other is not distinguishable.
- There can be improper estimates as the irreducibel error per each regression line is higher.

A better approach is to construct a regression line with multiple $X$ values as:

$Y = \beta + \beta X_1 + \beta_2 X_2 + ... \beta_pX_p$

### 3.2.1 Estimating Regression COefficients

Coefficients of multiple linear regression are estimated in the same manner with the aim to minimize the sum of squared residuals or RSS.

$RSS = \sum(y_i - \hat y_i)^2$

This is done through selectively choosing the $\beta$'s that minimize $RSS$.

Computing them, unlike the simple straightforward OLS equation is more complex and takes the form of matrix operations, whicih can be done simply using a software package ( numpy for isntance ).

### 3.5 Comparision of Linear Reg ression with K-Nearest Neighbors

Parametric methods, like linear regression, advantages as they are easier to fit, but their disadvant age is that they make assumptions about the form of function $f$.

If the assumption is far from the true $f$, it will cease to be reliable for proper prediction, and will fit horribly.

Non paramteric methods can be a better means as they don't assume a funtional form of $f$. 

A simple one is KNN as a regressor.

It identifies the $K$ nearest neighbors to a point $x$, represented by $N_0$ and then estimates $f(x)$ using the average of all training responses in $N_0$.

<div align = 'center'>

$\hat{f}(x_0) = \frac{1}{K} \sum_{x_i \in N_0} y_i$
</div>

The choices of $K$ matters, as the higher lower $K$ is, the more flexible and variance it will have. While the higher $K$ is the lower variance it will have will increasing it's bias.

Finding an optimal balance is essential.

In essence then, choosing a non parametric approach can be beneficial in some cases, but choosing a parametric model, such as a linear regression, can outperform a non-parametric method when the estimated model function is close to the true form of $f$.

If a paramteric method is correctly chosen to be close to $f$ it will outperform non-parametric methods.

LEFT OFF ON PAGE 114