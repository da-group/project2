##########################################
# Author: Zheyi Wang
# E-mail: zeyikwong@gmail.com
##########################################

import numpy as np
import argparse
import pandas as pd
from apyori import apriori

def getArguments():
    # get and parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='./dataset/crime2017_cleaned.csv', help='the file path')
    parser.add_argument('-s', type=float, default=0.2, help='Minimum support ratio (must be > 0, default: 0.1).')
    parser.add_argument('-c', type=float, default=0.5, help='Minimum confidence (default: 0.5).')
    return parser.parse_args()

#############################
# association rule mining by running Apriori
def associationRules(data,min_s,min_c):
    # convert the DataFrame into a list
    list_data = data.astype(str).values.tolist()
    results = list(apriori(list_data, min_support=min_s, min_confidence=min_c))
    for i in range(0, len(results)):
        print("item set:")
        print(list(results[i][0]))
        print("details:")
        print(results[i])
        print("")
    '''
    list_rules = [list(results[i][0]) for i in range(0, len(results))]
    print(list_rules)
    '''
    return results

def main():
    args = getArguments()
    myData = pd.read_csv(args.f, sep=',', encoding='latin1')
    # association rule mining to find the frequent patterns
    min_support = args.s
    min_confidence = args.c
    while min_support<1:
        print("*****************************************")
        print("min_support = " + str(min_support))
        print("min_confidence = " + str(min_confidence))
        print("*****************************************")
        associationRules(myData, min_support, min_confidence)
        min_support+=0.1


if __name__ == '__main__':
    main()
