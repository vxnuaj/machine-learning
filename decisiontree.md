## decision tree

- Need to determine the optimal split feature of X and the optimal split threshold of the feature.
  - To do so, we iterate over each feature and iterate over individual values of a split threshold of a given feature.
  - At each iteration of the split threshold, we determine the information gain.

        So essentially, I need to be able to split the data into halves where one half represents the samples higher than the thresholds for a given feature. Then compute the information gain

        To compute the information gain, we need to compute the parent_entropy minus the weighted entropy of the child nodes.

        Therefore, we must compute the individual entropy of the subsets first to then finally get the weighted entropy (left off here.)

Now we finished computing the information_gain for a given node, we need to determine how we can choose the correct split and return a proper Node.

How do we grow a tree?


Take a set of samples X.

Identity the ideal feature
Identity the ideal split threshold for the feature
Split left samples that are below the split threshold and split right samples that are above the split threshold