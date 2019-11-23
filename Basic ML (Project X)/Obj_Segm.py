import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from sklearn.cluster import KMeans

img = mping.imread('d:/1.jpg')
plt.axis('off')

X = img.reshape((img.shape[0]*img.shape[1],img.shape[2]))
for K in [3]:
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)
    img4 = np.zeros_like(X)
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    # reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation='nearest')
    plt.axis('off')
    plt.show()

