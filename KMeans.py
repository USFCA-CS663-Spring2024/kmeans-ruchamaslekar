# kmeans.py
from cluster import cluster
import numpy as np

class KMeans(Cluster):
    def __init__(self, k=5, max_iterations=100):
        super().__init__()
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        X = np.array(X)

        # Initialize centroids randomly
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iterations):
            # Assign each instance to the closest centroid
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

            # Update centroids based on mean of assigned instances
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return labels.tolist(), centroids.tolist()
