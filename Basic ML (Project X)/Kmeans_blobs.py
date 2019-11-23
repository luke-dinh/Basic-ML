import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
X, y = make_blobs (n_samples= 300, centers= 5, cluster_std= 0.6, random_state= 0)
plt.scatter(X[:,0], X[:,1])
wcss = []
for i in range(1,11):
    kmeans = KMeans (n_clusters= i, init = 'k-means++', max_iter=300, n_init=10, random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title ('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans (n_clusters=5, init = 'k-means++', max_iter=300, n_init=10, random_state= 0)
pred_y= kmeans.fit(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(X[:,0], X[:,1], c= kmeans.labels_.astype(float), s=50, alpha=1)
plt.scatter (kmeans.cluster_centers_[0,0], kmeans.cluster_centers_[0,1], s=100, c='r')
plt.scatter (kmeans.cluster_centers_[1,0], kmeans.cluster_centers_[1,1], s=100, c='g')
plt.scatter (kmeans.cluster_centers_[2,0], kmeans.cluster_centers_[2,1], s=100, c='y')
plt.scatter (kmeans.cluster_centers_[3,0], kmeans.cluster_centers_[3,1], s=100, c='w')
plt.scatter (kmeans.cluster_centers_[4,0], kmeans.cluster_centers_[4,1], s=100, c='k')
plt.show()
