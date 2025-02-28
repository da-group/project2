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
import argparse


filepath = './dataset/crime2017_preprocessed.csv'


def getArguments():
    # get and parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', type=str, default=filepath, help='the file path')
    return parser.parse_args()




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
  
    #plot PCA 
    normalizedDataFrame["labels"] = cluster_labels
    X = normalizedDataFrame.values
    pca = decomposition.PCA(n_components=2)
    plot_columns = pca.fit_transform(normalizedDataFrame)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.savefig('./plot/kmeans_pca.png')
    plt.show()

    

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
    
     #plot PCA 
    normalizedDataFrame["labels"] = cluster_labels
    X = normalizedDataFrame.values
    pca = decomposition.PCA(n_components=2)
    plot_columns = pca.fit_transform(normalizedDataFrame)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.savefig('./plot/agglomerative_pca.png')
    plt.show()


#DBScan
def DBScan():
    normalizedDataFrame = NormalizeData()
    #normalizedDataFrame= normalizedDataFrame.sample(frac=0.2, replace=False,axis=0)
    #print(normalizedDataFrame)
    
    dbscan = DBSCAN(algorithm='auto', eps=0.2, leaf_size=30, metric='euclidean',
    min_samples=10, n_jobs=1, p=None)
    cluster_labels = dbscan.fit_predict(normalizedDataFrame)
    pprint("DBSCAN:")
    pprint(cluster_labels)
    
     #Use Calinski-Harabaz procedures to measure the cluster quality
    calinski_avg =calinski_harabaz_score(normalizedDataFrame, cluster_labels)
    print("The average calinski_harabaz_score is :", calinski_avg)
    print('\n')
    
     #plot PCA 
    normalizedDataFrame["labels"] = cluster_labels
    X = normalizedDataFrame.values
    pca = decomposition.PCA(n_components=2)
    plot_columns = pca.fit_transform(normalizedDataFrame)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.savefig('./plot/dbscan_pca.png')
    plt.show()




     
    
if __name__=="__main__":
    args = getArguments()
    myData = pd.read_csv(args.f, sep=',', encoding='latin1')
    myData = myData.drop(['REPORT_DAT','BLOCK','BLOCK_GROUP','START_DATE','END_DATE','ANC','PSA'],axis = 1)
    KmeansClustering()
    Agglomerative()
    DBScan()