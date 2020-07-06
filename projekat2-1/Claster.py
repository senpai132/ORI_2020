import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import DataAcquisition as da

sns.set()

from sklearn.cluster import KMeans

data = None

def clusterAndDraw(input_data, n, ind):
    kmeans = KMeans(n)
    kmeans.fit(input_data)

    identified_clusters = kmeans.fit_predict(input_data)

    data_maped = input_data.copy()
    data_with_clusters = data_maped.copy()
    data_with_clusters['Cluster'] = identified_clusters

    if ind == 0:
        plt.scatter(data_with_clusters[3], data_with_clusters[7], c = data_with_clusters['Cluster'], cmap = 'rainbow')
    else:
        plt.scatter(data_with_clusters['BALANCE'], data_with_clusters['TENURE'], c=data_with_clusters['Cluster'], cmap='rainbow')

    plt.show()

def elbowMethod(data):
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


def chooseDataAcquisition():
    print("1 - Scaled data")
    print("2 - Normalised data")
    print("3 - Raw data")
    print("4 - Customly normalised data")
    print("-----------------------------")
    x = input("Choose preprocessing type: ")

    if x == '1':
        data = da.getScaledData()
        #elbowMethod(data)
        clusterAndDraw(data, 4, 0)
    elif x == '2':
        data = da.getNormalisedData()
        #elbowMethod(data)
        clusterAndDraw(data, 4, 0)
    elif x == '3':
        data = da.getRawData()
        #elbowMethod(data)
        clusterAndDraw(data, 3, 1)
    elif x == '4':
        data = da.getCustomlyNormalisedData()
        #elbowMethod(data)
        clusterAndDraw(data, 2, 1)
    else:
        print("Nevalidan ulaz")

#elbowMethod()
#clusterAndDraw()
chooseDataAcquisition()