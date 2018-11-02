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
    parser.add_argument(
        '-f', type=str, default='./dataset/crime2017_cleaned.csv', help='the file path')
    return parser.parse_args()


#############################
# determine the mean (mode if categorical), median, 
# and standard deviation of attributes in the dataset
def describe(data):
    # describe the nominal attributes, including the mode
    print(data.describe(include=[np.object]))
    # describe the numeric attributes, including the mean, median, std
    print(data.describe())


# def LOF(colum):


#############################
# drop the attributes that are not useful
def dropUselessAttr(data):
    '''
    in crime2017_cleaned.csv: "OBJECTID" is not helpful
    '''
    data.drop(['OBJECTID'], axis=1, inplace=True)

#############################
# handle missing values
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
            because the mean isn't valid in most cases
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


#############################
# bin some numeric attributes
def binning(data):
    '''
    bin the attribute PSA (min:101, max:708) in crime2017_cleaned.csv
    '''
    for column in data.columns:
        if column == 'PSA':
            # pre-bin
            psa_bins = [100, 200, 300, 400, 500, 600, 700, 800]
            data['PSA_bin'] = np.digitize(data['PSA'], psa_bins)
    return data


#############################
# convert nominal attributes to numeric attributes
def convertNominalAttr(data):
    for column in data.columns:
        '''
        for crime2017_cleaned.csv
        '''
        if column == 'SHIFT':
            shift_mapping = {'DAY': 1, 'EVENING': 2, 'MIDNIGHT': 3}
            data[column] = data[column].map(shift_mapping)
        if column == 'METHOD':
            method_mapping = {'OTHERS': 1, 'KNIFE': 2, 'GUN': 3}
            data[column] = data[column].map(method_mapping)
        if column == 'NEIGHBORHOOD_CLUSTER':
            # e.g. 'Cluster 20'(string) ==> 20(int64)
            neighborhoodcluster_mapping = {}
            for i in range(len(data[column])):
                neighborhoodcluster_mapping[data[column][i]] = int(data[column][i].split(' ')[1])
            data[column] = data[column].map(neighborhoodcluster_mapping)
        if column == 'VOTING_PRECINCT':
            # e.g. 'Precinct 100'(string) ==> 100(int64)
            votingprecinct_mapping = {}
            for i in range(len(data[column])):
                votingprecinct_mapping[data[column][i]] = int(data[column][i].split(' ')[1])
            data[column] = data[column].map(votingprecinct_mapping)
        if column == 'OFFENSE':
            # the bigger the number, the higher level of the crime
            offense_mapping = {'THEFT/OTHER': 1, 'THEFT F/AUTO': 2, 'MOTOR VEHICLE THEFT': 3, 'BURGLARY': 4,
                               'ARSON': 5, 'ASSAULT W/DANGEROUS WEAPON': 6, 'ROBBERY': 7, 'SEX ABUSE': 8, 'HOMICIDE': 9}
            data[column] = data[column].map(offense_mapping)


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
    # convert some nominal attributes to numeric attributes
    convertNominalAttr(myData)
    describe(myData)
    # write into a new .csv file
    myData.to_csv(args.f.replace('_cleaned.csv', '_preprocessed.csv'), sep=',',index=None)


if __name__ == '__main__':
    main()
