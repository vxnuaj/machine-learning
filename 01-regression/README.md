# Regression

### Polynomial Linear Regresion

A type of linear regression, that adds higher order linear combination to a vanilla linear regression regressor.

It adds a degree $n$ to a given feature $x_i$ which then allows for the estimated regression line to fir as a curve to non-linear data. 

<div align = 'center'>

$y = b_0 + b_1x_1 + b_2x_1^2 + ... + b_nx^n$
</div>

Then, polynomial regression is able to better model a set of non-linear data, but again, is still a parameteric model and making an assumption of the functional basis of $f$ with $\hat{f}$ with the wrong degree of a polynomial can lead to incorrect estimatins of $f$.

Doing so enhances predictability while still keeping an interpretable model.

Albeit, it's increase in flexibility allows for better modelling of data, of course at a risk of overfitting.

It's still called linear as the coefficients $b_i$ are still performing a linear combination.

