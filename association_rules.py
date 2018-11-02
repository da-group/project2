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
    return parser.parse_args()

#############################
# association rule mining by running Apriori
def associationRules(data):
    # convert the DataFrame into a list
    list_data = data.astype(str).values.tolist()
    results = list(apriori(list_data, min_support=0.4, min_confidence=0.7))
    list_rules = [list(results[i][0]) for i in range(0,len(results))]
    print(list_rules)
    return results

def main():
    args = getArguments()
    myData = pd.read_csv(args.f, sep=',', encoding='latin1')
    # association rule mining to find the frequent patterns
    associationRules(myData)


if __name__ == '__main__':
    main()
