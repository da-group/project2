# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:12:03 2018

@author: Riven
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.cluster import KMeans,DBSCAN,	AgglomerativeClustering
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import calinski_harabaz_score



filepath = './dataset/crime2017_preprocessed.csv'
myData = pd.read_csv(filepath, sep=',', encoding='latin1')
myData = myData.drop(['REPORT_DAT','BLOCK','BLOCK_GROUP','START_DATE','END_DATE','ANC','PSA'],axis = 1)

def NormalizeData():
    fixData=pd.concat([myData['CCN'], myData['SHIFT'], myData['METHOD'], myData['OFFENSE'], myData['XBLOCK'], myData['YBLOCK'],myData['WARD'],myData['DISTRICT'],myData['NEIGHBORHOOD_CLUSTER'],myData['CENSUS_TRACT'],myData['VOTING_PRECINCT'],myData['LATITUDE'],myData['LONGITUDE'],myData['PSA_bin']], 
                 axis=1, keys=['CCN', 'SHIFT', 'METHOD', 'OFFENSE', 'XBLOCK','YBLOCK','WARD','DISTRICT','NEIGHBORHOOD_CLUSTER','CENSUS_TRACT','VOTING_PRECINCT','LATITUDE','LONGITUDE','PSA_bin'])
    x = fixData.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    return normalizedDataFrame


#K-Means
def KmeansClustering():
    normalizedDataFrame = NormalizeData()
    #Let k = 5
    k = 5
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    pprint("KMeans: K = "+str(k))
    pprint(cluster_labels)

    
     #Use Calinski-Harabaz procedures to measure the cluster quality
    calinski_avg =calinski_harabaz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average calinski_harabaz_score is :", calinski_avg)
    print('\n')
  
"""
    #plot PCA 
    X = normalizedDataFrame.values
    y = cluster_labels
    pca = decomposition.PCA(n_components=3)
    X = pca.fit_transform(X)
    
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()

    for label in [0,1,2,3,4]:
        ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), label,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

plt.show()
    
    
  """  
    
    

#Agglomerative Clustering
def Agglomerative():
    normalizedDataFrame = NormalizeData()
    #Let k = 5
    k = 5
    Agglomerative = AgglomerativeClustering(n_clusters=k)
    cluster_labels = 	Agglomerative.fit_predict(normalizedDataFrame)
    pprint("AgglomerativeClustering: K = "+str(k))
    pprint(cluster_labels)
    
     #Use Calinski-Harabaz procedures to measure the cluster quality
    calinski_avg =calinski_harabaz_score(normalizedDataFrame, cluster_labels)

    print("For n_clusters =", k, "The average calinski_harabaz_score is :", calinski_avg)
    print('\n')
    

#DBScan
def DBScan():
    normalizedDataFrame = NormalizeData()
    dbscan = DBSCAN(algorithm='auto', eps=3, leaf_size=30, metric='euclidean',
    metric_params=None, min_samples=2, n_jobs=None, p=None)
    cluster_labels = dbscan.fit_predict(normalizedDataFrame)
    pprint("DBSCAN:")
    pprint(cluster_labels)
    
     #Use Calinski-Harabaz procedures to measure the cluster quality
    calinski_avg =calinski_harabaz_score(normalizedDataFrame, cluster_labels)
    print("The average calinski_harabaz_score is :", calinski_avg)
    print('\n')



     
    
if __name__=="__main__":
  KmeansClustering()
  Agglomerative()
  DBScan()