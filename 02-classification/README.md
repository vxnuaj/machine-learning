## Classification

## Logistic Regression

Performs binary classification, defined as:

<div align = 'center'>

$p(X) = \frac{e^{\beta_0 + \beta_1X}}{1 + e^{\beta_0 + \beta_1X}} = \frac{1}{1 - e^{\beta_0 + \beta_1X}}$
</div>

and optimizes for the correct value through maximum likelihood.

For binary data, logistic regression is able to better capture the probability of a given $X$ belonging to $1$ or $0$, based on a rounded probability value based on a threshold.

The odds of a given $X$ are computed as:

<div align = 'center'>

$\frac{P(X)}{1 - P(X)}$
</div>

where then that is equivalent to $e^{\beta_0 + \beta_1 X}$

<div align = 'center'>

$\frac{P(X)}{1 - P(X)} = e^{\beta_0 + \beta_1 X}$
</div

The higher a given probability is, the higher the odds wil be the inverse being true for a lower probability.

The $log(odds)$ is the logarithm of the above as:

<div align = 'center'>

$logit = log(\frac{P(X)}{1 - P(X)}) = \beta_0 + \beta_1 X$
</div>

This is also known as the logit.

This logit is then used in the logistic function as:

<div align = 'center'>

$p(logit) = \frac{1}{1 - e^{logit}}$
</div>

Where an increase in an $X_i$ by a given unit $n$ will increase the $logit$ by a factor of $n * \beta_i$ or increases the regular odds by a factor of $e^{\beta_i}$.

While the increase in the $logit$ might appear to be linear, the characteristic of the logistic sigmoid function as $\frac{1}{1 - e^{z}}$ makes the increase in the output probability proprotionally non-linear to the input $logit$. The rate of change to the current $p(X)$ depends on the current value of $X$ as the sigmoid function is in an $S$-shaped curve.

Interpreting this can be useful for understanding how the input features of a model change a specific outcome.

Non-linear models, as linear regression, can then be modeled through a *likelihood function*, where the coefficients $\beta_0$ and $\beta_1$ are chosen to maximize the likelihood function.

### Multiple Logistic Regresion

Posits the same process as logistic regression with the addition of multiple feature $X$.

<div align = 'center'>

$p(X) = \frac{e^{\beta_0 + \beta_1X_1 + ... + \beta_nX_n}}{1 + e^{\beta_0 + \beta_1X_1 ... + \beta_nX_n}} = \frac{1}{1 - e^{\beta_0 + \beta_1X_1 + ... + \beta_nX_n}}$
</div>

### Multinomial Logistic Regression

An extension of the logistic regression function, also can be known as softmax regression,

You can train different models for each $K$ class as:

<div align = 'center'>

$Pr(Y = k \mid X = x) = \frac{e^{\beta_{k0} + \beta_{k1} x_1 + \ldots + \beta_{kp} x_p}}{1 + \sum_{l=1}^{K-1} e^{\beta_{l0} + \beta_{l1} x_1 + \ldots + \beta_{lp} x_p}}$
</div>

where the numerator is the total probabiity over all classes and the denominator is the probabiity for class $k$.

This then computes the probabiity of a $k$ belonging in $K$

Then you can compute the probability for a chosen reference class $K$ as:

<div align = 'center'>

$Pr(Y= K | X = x) = \frac{1}{1 + \sum e^{Z}}$
</div>

Then, the log odds of $k$ belonging in $K$ is defined as:

<div align = 'center'>

$log(\frac{Pr(Y =k|X=x)}{Pr(Y = K | X = x)})=βk_0+βk_1x_1+···+β_{kp}x_p$
</div>

In essence, the log odds is transformed into a probability through the logistic sigmoid function.

Note that the reference class is purely used to compute the log odds of $k$ compared to $K$.

The result, whether we choose another class $k$ to be $K$, of the log odds will the same no matter which $K$. 

The difference is that in when optimizing for model interpretation, the coefficients $\beta$ will differ in value, choosing different $K$ so one must be careful when interpreting the model.

### K-Nearest-Neighbors

A K-Nearest Neighbor Clasifier aims to classify a datapoint $X$ by taking a look at the labels of the nearest $K$ datapoints, to take the probability of $X$ belonging to a class $Y$.

<br>
<div align = 'center'>
<img src = '../images/knn.png' width = 400>
</div>
<br>

Mathematically, can be defined as:

$Pr(Y = j | X = x_0) = \frac{1}{K} \sum_{i∈N_k} I(y_i = j)$

Where we choose the $K$ nearest neighbors, then compute through $I(y_i = j)$ the probability that a given $x_0$ belongs to a class $Y$ and average it over $K$.  

This is computed multiple times for all $j$ classes to properly estimate the probability of $x_0$ belonging to the set of $j$ classes.

This is done by comparing the amount of nearest classes to a given $x_0$. The class that has the highest amount of datapoints near to $x_0$ is then chosen as the prediction class that $x_0$ might belong to.

The KNN then classifies the test observation $x_0$ to the class with the highest probability.

If there happens to be a tie between the number of datapoints nearest to $x_0$ amongst seperate classes, an odd number $K$ can be chosen which breaks a tie in terms of binary classification, but in multi-class classification, the point that's closest to $x_0$, computed per the $L_2$ norm, can be chosen as the class which $x_0$ belongs to. 

Note that the choice of $K$ can determine if the model overfits the training set or not. 

When $K$ is a small value, the KNN is less rigid and more flexible, thereby being unable to generalize to real data, as it has more variance.

When $K$ is a large value, the KNN is more rigid and less flexible, thereby having a higher bias. 

The ***pro*** of choosing KNN as a model is it's simple to implement, adapts well to new training data, and is easy to interpret.

The ***cons*** can include

- Slow prediction, due to many distance calculations
- Generating insights into the data can be difficult as you have no parameters that tell you how much of a certain feature in $x_0$ contribute to the output $Y$ 
- Suffering from the curse of dimensionality. When adding extra dimensions to features, they tend to disproportionately skew in terms of distance. This can decorrelate a given data point form another despite belonging within the same class.

Applications have included 
- Automated web usage data mining and reocmmendation systems -- A KNN takes the web usage of a user and then builds a recommendation system using a KNN.

<details> <summary> Other notes</summary>

To implement the kd-tree,

We have to take a column of the dataset and take the median. Then we split in half based on the median. Anything lower than the median goes in one group anything higher goes into the other. This is done for each available segment until we reach a minimum which is denoted by the leaf size.

</details>

### Support Vector Machines

A linear support vector machine is a linear classifier that aims to draw a boundary between a set of datapoints, based on a linear decision boundary and "Support Vectors".

<img src = "https://www.baeldung.com/wp-content/uploads/sites/4/2021/03/svm-all.png"></img>

Support vectors are the margins or 'padding' that surround the linear classifier of a Support Vector Machine. 

> *Note that support vector machines can be extended to Non-Linear Support Vector Machines via the Kernel Trick.*

The goal of a support vector machine is to find the line or hyper plane of a decision boundary that best separates the classes.

> *In a 2-dimensional space, a line is suitable. In $n$ dimensional spaces, where $n > 2$, a hyperplane is needed.*

We want to find the Support Vectors which correspond to the largest gap between the decision boundary and the datapoints by minimizing the Hinge Loss.

The hinge loss is a loss function used for training classification models. 

It's made for maximum margin classifiers, meaning classifiers that aim to optimize the margin distance between classes. 

The hinge loss can be defined as:

$loss = max(0, 1 - y \cdot f(x))$

where $y$ is the true class label and $f(x)$ corresponds to a prediction from the given classifier, typically a Support Vector Machine.

Thereby, if the classifier is uncertain about the classification where $f(x) = 0$, the $loss$ value will hit exactly $1$ as:

$max(0, 1 - y \cdot 0) \rightarrow max(0, 1 - 0) \rightarrow 1$

If a given label is $1$ and the predicted class is $2$, the $loss$ value will be:

$max(0, 1 - 1 \cdot 2) \rightarrow max(0, 1-2), \rightarrow 0$

Note that 
- The hinge loss does not punish correct classifications.
- The as soon as a datapoint enters the margins defined by the support vectors the loss value begins to rise.
- For the loss value corresponding to a decision boundary, the loss hits exactly 1.

The decision boundary is defined as $w \cdot x + b = 0$.

To classify predictions, if the SVM bases it's binary classifications based on the sign of an output. 

If the prediction of an SVM $f(x)$ is $>0$, the prediction class is $+1$, while if $f(x) < 0$, the prediction class is $-1$.

The weights $w$, represent the distance of the hyperplane or decision boundary that separates the classes in $x$ as best as possible from the corresponding support vectors. They yield the coordinates of a vector orthogonal to the hyper plane.

> *Orthogonal meaning the vectors are perpendicular. 
> $w \cdot x$ if $x$ is the hyperplane, will yield $0$*

Therefore the $L_2$ norm of $w$, as $||w||$, will yield the distance between the decision boundary and it's corresponding Support Vectors.

If we have $w$ as:

$w = \begin{pmatrix} 3 \\ 0 \end{pmatrix}$

and $x$ as:

$x = \begin{pmatrix} 6 \\ 4 \end{pmatrix}$

then within the equation of an SVM as $w \cdot x + b$, the dot product of $w$ and $x$ yields $18$.

Assuming that $b$ is $0$, the prediction of the SVM is $18$. Thereby, given that the prediction is positive, the class predicted is $+1$.

The margin between the optimal line or decision boundary and a given datapoint is called the geometric margin, computed as the Euclidean Distance defined as:

$\gamma = \frac{y_i(w^Tx_i + b)}{||w||}$

where the numerator represents the predicted vector derived from the boundary $w$, inputs $x$, and bias $b$.

The denominator scales the input down based on the euclidean norm of $w$ or the distance from the decision boundary to the Support Vectors. This makes sure that when computing the geometric mean, the input $x$ is independent to scaling by $w$.

A support vector machine can be sensitive to outliers, where if a single datapoint of a different class is grouped near to datapoints of different classes, then the decision boundary and support vectors will skew, indicating a level of overfitting.

Thereby, we can rely on Regularization for SVMs, similar to L2 Regularization where we add an additional loss penalty to the hinge loss.

Regularization in SVMs allows for the decision boundary and the corresponding support vectors to become non-complex, where the decision boundary doesn't overfit to nuanced features, and rather generalizes.

Similar to L2 Regularization or L1 Regularization, we add a penalty term to the hinge loss of an SVM.

$L_{regularized} = SVMCost(\beta_i) + C ||\beta||$

where $C$ is the tunable hyperparameter denoting the level of regularization we want.

If $C$ is a large value, the regularization onto the model will be higher. 
If $C$ is a small value, the regularization onto the model will be smaller.

Note that if the hinge loss is smaller, the penalty term will be higher. Inversely, if the hinge loss is higher, the penalty term would be smaller.

This is as the smaller the hinge loss is, the more complex the coefficients $\beta_i$ are, thereby indicating that they might hold a higher value. 

Therefore, the calculation of the magnitude of $\beta_i$, as $||\beta||$, will be higher the lower the hinge loss is.

#### Other

We can compute the functional margin as $y(wx + b)$. Thereby, we can then compute the support vectors as the vectors with the shortest functional margin. We can then compute the geometric distance of the support vectors as their euclidean distance from the margin boundary.

### Decision Trees

Decision Trees are a type of model that makes use of binary or multiway decisions to converge on a decision that's optimal for the given situation.

Given a datapoint, at each node of a decision tree, there must be a binary or multiway decision path where the tree splits off into other nodes or perhaps leaves that represent a final decision.

The depth of a decision tree defines the length of the longest path from the root node to a leaf node.

- A shallow tree might under-fit data.
- A deep tree might over-fit the data.

The root node is the first node while the leaf nodes represent the final decision or output. The internal decision nodes are the intermediary nodes that represent intermediary decisions between a root node and a leaf node. Branches are what connects the nodes.

The decision tree continues to split until a node is ***pure***, meaning one class remains, a ***maximum depth*** is reached, a ***minimum sample size*** is reached, or a ***performance metric***, say Entropy, is achieved.

There are different types of splitting criteria for a decision tree, such as the Gini Impurity or Entropy

Greedy Search can be used to find the optimal decision path for a tree, based on the maximum information gained from the split.

Decision Trees can tend to have high variance and overfit, given that it's non-parametric and is very flexible. 

The solution to this is to prune the tree perhaps via maximum depth or a performance metric such as classification error or information gain.

Benefits of decision trees include the absence of a need for pre-processing and being able to handle all types of data.

**Gini Impurity**

The splitting criteria for the original CART decision tree algorithm.

It's used to quantify the impurity or disorder of a set of elements.

It is defined as:

$G = \sum_{i = 1}^K p_i(1 - p_i)$

which can be simplified to:

$G = 1 - \sum_{i = 1}^K p_i^2$

where it's interval is in the range $[0, .5]$ and $p_i$ is the probability of an item being classified into class $i$

while for binary classification, it is defined as:

$G = p_1 ( 1 - p_1) + p_2(1-p_2)$

Say for binary classification, when a Decision Tree reaches it's leaf node, outputting a probability of $1$ for a choice, the Gini Index will be $0$ as:

$G = 0( 1 - 0) + 1(1-1) = 0$

Inversely, when the probability is $.5$ for the given choices, the Gini index will be $.5$ as:

$G = .5( 1 - .5) + .5(1-.5) = .25 + . 25$

When the Gini Index reaches $0$, a decision tree is essentially reached the leaf node and is sure of the decision it'd make based on the training data.

**Entropy**

Entropy is the measure of uncertainty or probabilistic disorder in a system, given probabilities $p_i$, where $i$ is a given class.[^1]

$Entropy = - \sum_{i = 1}^K p_i log(p_i)$

Given a 2 class problem where $K = 2$, it can be defined as:

$Entropy = - \sum_{i = 1}^K p_1 log(p_1) + (1 - p_2) log(1 - p_2)$

The bounds of entropy are $[0, 1]$.

**Information Gain**

Information Gain is a metric that denotes the amount of useful information that a given feature yields, to reduce the uncertainty or information entropy in a given scenario.

It can be computed as:

$IG(D, A) = H(D) - H(D, A)$, where $D$ is a dataset and $A$ is a specific feature.

- $H(D)$ is the entropy of a dataset as, $- \sum_{i = 1}^K p_i log(p_i)$, applied prior to using features to split a dataset.

- $H(D, A)$ is the weighted entropy of the subsets of the original $D$ as $\sum_{j=1}^n\frac{D_j}{|D|}H(D_j)$ where $n$ is the number of total subsets and $D_j$ is the number of total instances in the given subset. $H(D_j)$ is then the entropy of the given $D_j$. 

The subtraction, then yields the ***information gain***, which then denotes how much each additional feature or in the context of decision trees, how much each node contributed to reducing the amount of disorder or uncertainty in the given problem.