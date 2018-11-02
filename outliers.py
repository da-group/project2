# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:18:22 2018

@author: Riven
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing

filepath = './dataset/crime2017_preprocessed.csv'
myData = pd.read_csv(filepath, sep=',', encoding='latin1')
myData = myData.drop(['REPORT_DAT','BLOCK','BLOCK_GROUP','START_DATE','END_DATE','ANC'],axis = 1)


def lof(k):
   clf = LocalOutlierFactor(n_neighbors=k,contamination=0.01)
   labels = clf.fit_predict(myData)
   outlier_counts = 0
   for label in labels:
       if label == -1:
           outlier_counts= outlier_counts+1
   print(labels)
   print(outlier_counts)

    
def main():
    lof(35)
    lof(25)
    lof(45)
    

if __name__ == '__main__':
    main()
