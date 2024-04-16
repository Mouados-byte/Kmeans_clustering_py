import numpy as np

class KMeans:
  def __init__(self, n_clusters=3, max_iter=300):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    
  def fit(self, X):
    self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
    
    for _ in range(self.max_iter):
      self.labels = np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)
      new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
      
      if np.all(self.centroids == new_centroids):
        break
      self.centroids = new_centroids
  
  def predict(self, X):
    return np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)
  
  def fit_predict(self, X):
    self.fit(X)
    return self.predict(X)
  
  def visualize(self, X):
    import matplotlib.pyplot as plt
    labels = self.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='r', marker='x')
    plt.show()
  
  
# Exemple
X = np.random.rand(100, 2)
kmeans = KMeans(n_clusters=8)
kmeans.visualize(X)
labels = kmeans.predict(X)
print(labels)
