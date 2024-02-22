import numpy as np

class KMeans:
    def __init__(self, k=5, max_iter=100, balanced=False):
        self.k = k
        self.max_iter = max_iter
        self.balanced = balanced  
                    
    def fit(self, X):
        # Randomly initializing centroids
        centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iter):
            # Calculating Euclidian distance
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

            # Assigning labels based on the nearest centroid
            labels = np.argmin(distances, axis=1)

            if self.balanced:
                cluster_sizes = np.bincount(labels, minlength=self.k)
                target_size = int(len(X) / self.k)  # target input length by number of clusters

                for cluster in range(self.k):
                    instances_to_remove = cluster_sizes[cluster] - target_size

                    if instances_to_remove > 0:
                        cluster_instances = np.where(labels == cluster)[0]
                        instances_to_remove_idx = np.random.choice(cluster_instances, instances_to_remove, replace=False)

                        for idx in instances_to_remove_idx:
                            distances_to_centroids = np.linalg.norm(X[idx] - centroids, axis=1)
                            sorted_clusters = np.argsort(distances_to_centroids)
                
                            # Finding nearest cluster having less than target size
                            for closest_cluster in sorted_clusters:
                                if cluster_sizes[closest_cluster] < target_size:
                                    labels[idx] = closest_cluster
                                    cluster_sizes[cluster] -= 1
                                    cluster_sizes[closest_cluster] += 1
                                    break

            # Finding the centroids after assigning labels.
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # When there is no change in new centroids and previous values of centroids.
            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids      
        
        self.labels_ = labels
        self.cluster_centers_ = centroids

        return self.labels_, self.cluster_centers_