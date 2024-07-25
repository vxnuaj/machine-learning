import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load MNIST data

print('loading data')

mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target

print('reducing dimensionality with TSVD')

# Reduce dimensionality with TSVD
tsvd = TruncatedSVD(n_components=50)
X_reduced = tsvd.fit_transform(X)

#print('reducing dimensionality again with t_SNE')

# Further reduce with t-SNE
#tsne = TSNE(n_components=2, random_state=42)
#X_tsne = tsne.fit_transform(X_reduced)

print('splitting data')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

print('training the tree')

# Train a decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print(f'evaluating tree')

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')