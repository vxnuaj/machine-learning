# Clustering

Clustering is a means to categorize unlabeled data in specific clusters, where each cluster has as unique identifier, akin to a class label.

Clustering helps differentiate amongst different datapoints when they are unlabeled based on a similarity metric.

Many clustering algorithms have a runtime complexity of $O(n^2)$, as they compare a given datapoint, $n_i$, to all other datapoints, $n_j$, $n-1$ times.

$O(n^2) = \frac{n(n - 1)}{2}$

If I have $4$ datapoints, meaning $n = 4 \rightarrow O(4^2)$ as we compare each point with each and every point.

Dataset: $1, 2, 3, 4$

$(1, 2), (1, 3,), (1, 4), (2, 3), (2, 4), (3, 4)$, we have $6$ total unique comparisons.

While if we have $5$ datapoints, meaning $n = 5 \rightarrow O(5^2)$:

Dataset: $1, 2, 3, 4, 5$

$(1, 2), (1, 3,), (1, 4), (1, 5) (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)$, we have 10 total unique comparisons.

The # of unique comparisions increasing by $4$.

If we add another $n$, such that $n = 6$, the # of unique comparisons would then scale to $15$. The growth of unique comparisions begins to scale at an accelerated rate, increasing faster as $n$ grows. 

In practice, clustering algorithms that have $O(n^2)$ complexity aren't ideal to use as $n$ grows.

### Centroid Based Clustering

Centroid based clustering organizes unlabeled data into non-hierarchical clusters. It's efficient but sensitive to initial conditions of their centroids and outliers that may skew the centroids.

> *Centroids are defined as the arithmetic mean of a given cluster*

### Density Based Clustering

Density based clustering clusters areas of with high sample density. 

This allows for discovering any amount of clusters of any shape, based on the hyperparameters:

- $\epsilon$, denoting the maximum distance between two points, for them to be considered neighbors, to form a cluster.
- MinPoints, denoting the minimum number of points required to form a cluster.

### Distribution Based Clustering

This assumes that data, belonging to a given class, is probabilistic and can be represented by a $PDF$ such as a Gaussian Distribution.

For a given cluster, as a datapoint distances itself away from the center of the cluster, the probability that the datapoint belongs to the cluster decreases. The inverse is true.

### Hierarchical Based Clustering

Hierarchical Based Clustering creates a tree of clusters, where each cluster is a sub component, nested under another cluster.

It clusters based on measuring the dissimilarity of datapoints at a given node.

It's better well suited for hierarchical data such as a taxonomies.

## K-Means Clustering

K-Means is a form of centroid based clustering where the algorithm, $\mathbb{A}$, aims to cluster the algorithm based on iteratively adjusting centroids by recomputing their arithmetic mean for each cluster at each iteration.

K-Means has a complexity of $O(n \cdot k \cdot i \cdot d)$ where:

- $n$ is the number of datapoints
- $k$ is the number of clusters.
- $i$ is the number of iterations.
- $d$ is the number of features.

It aims for the goal, $min(WCSS)$ where $WCSS = \sum_{i=1}^k \sum_{x\in C} ||x - \mu||^2$

1. Choose the number of clusters, $k$ (hyperparamter, randomly or via *k-means++*[^1])
2. Randomly choose $k$ centroids
3. Based on the euclidean distance, assign each point to the nearest $k_i$ centroid
4. For each given cluster, calculate the centroid by taking the arithmetic mean of all points in the cluster.
5. Reassign ech point to the nearest centroid.
6. Repeat until no points change clusters and the $WCSS$ is minimized[^2]

[^1]: Clarify what K-means++ is
[^2]: how does this includde WCSS?
