import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import DataAcquisition as da

sns.set()

from sklearn.cluster import KMeans

data = da.getNormalisedData()

def clusterAndDraw():
    kmeans = KMeans(4)
    kmeans.fit(data)

    identified_clusters = kmeans.fit_predict(data)

    data_maped = data.copy()
    data_with_clusters = data_maped.copy()
    data_with_clusters['Cluster'] = identified_clusters

    plt.scatter(data_with_clusters[3], data_with_clusters[7], c = data_with_clusters['Cluster'], cmap = 'rainbow')
    plt.show()

def elbowMethod():
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(i)
        kmeans.fit(data)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)

    print(wcss)
    number_clusters = range(1,10)
    plt.plot(number_clusters, wcss)
    plt.show()

#elbowMethod()
clusterAndDraw()