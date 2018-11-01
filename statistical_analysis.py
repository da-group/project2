##########################################
# Author: Zheyi Wang
# E-mail: zeyikwong@gmail.com
##########################################

import numpy as np
import argparse
import pandas as pd


def getArguments():
    # get and parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='./dataset/crime2017_cleaned.csv', help='the file path')
    return parser.parse_args()

'''
determine the mean (mode if categorical), median, 
and standard deviation of attributes in the dataset
'''
def describe(data):
    # describe the nominal attributes, including the mode
    print(data.describe(include=[np.object]))
    # describe the numeric attributes, including the mean, median, std
    print(data.describe())


# def LOF(colum):


def dropUselessAttr(data):
    '''
    drop the attributes that are not useful
    in crime2017_cleaned.csv: "OBJECTID" is not helpful
    '''
    data.drop(['OBJECTID'],axis=1,inplace=True)


def handleMissingValues(data):
    '''
    count the missing value of a column
    '''
    print("# of missing value of attributes:")
    for column in data.columns:
        NaN_list = data[column].isnull()  # bool list indicate missing value
        NaN_num = sum(NaN_list)  # number of missing value
        '''
        print the attribute with missing values
        '''
        if NaN_num > 0:
            if data[column].dtype == object:
                print(column + ": " + str(NaN_num))
                print("mode: " + str(data[column].mode()))
                print("")
            else:
                print(column + ": " + str(NaN_num))
                print("mode: " + str(data[column].mode()))
                print("mean: " + str(data[column].mean()))
                print("")
            '''
            fill the missing value with mode
            because the mean isn't valid
            '''
            data[column].fillna(data[column].mode()[0], inplace=True)


def checkMissing(data):
    '''
    check if there are still missing values after handling the missing values
    '''
    for column in data.columns:
        NaN_list = data[column].isnull()
        NaN_num = sum(NaN_list)
        print(column + ": " + str(NaN_num))

def binning(data):
    '''
    bin the attribute PSA (min:101, max:708)
    '''
    # pre-bin
    psa_bins = [100, 200, 300, 400, 500, 600, 700, 800]
    data['PSA_bin'] = np.digitize(data['PSA'], psa_bins)
    return data


def main():
    args = getArguments()
    myData = pd.read_csv(args.f, sep=',', encoding='latin1')
    # summary of the data
    dropUselessAttr(myData)
    describe(myData)
    # handle missing values
    handleMissingValues(myData)
    checkMissing(myData)
    # bin the data
    binning(myData)
    describe(myData)

if __name__ == '__main__':
    main()