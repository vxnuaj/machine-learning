import jax.numpy as jp
from sklearn.cluster import KMeans


model = KMeans(n_clusters = 3, init = 'random', max_iter = 10)

X = jp.array([
[1.0, 2.0],
[1.5, 1.8],
[5.0, 8.0],
[8.0, 8.0],
[1.0, 0.6],
[9.0, 11.0],
[8.0, 2.0],
[10.0, 2.0],
[9.0, 3.0]
])

model.fit(X)
print(model.inertia_)
