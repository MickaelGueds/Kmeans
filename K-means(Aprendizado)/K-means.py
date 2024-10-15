from sklearn.cluster import KMeans
import numpy as np
k = 5
kmeans = KMeans(n_clusters = k)
X = 2 * np.random.rand(100, 1)
y_pred = kmeans.fit_predict(X)


