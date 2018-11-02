####################################
# Author: Jiachi Zhang
# E-mail: zhangjiachi1007@gmail.com
####################################

import numpy as np
import pandas as pd
from scipy import stats


def getArguments():
    '''
    get and parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='./dataset/crime2017.csv', help='the file path')
    parser.add_argument('-m', type=str, help="choose ttest or avano to use")
    parser.add_argument('--attr', nargs='+', help="first is attribute grouped by, second is target attribute")
    parser.add_argument('--div', nargs='+', help="two values in groupby attribute and they should be converted to int")
    return parser.parse_args()


def tTest(myData, groupA, divL, targetA):
    # give a statistic analysis of relations of tow attributes
    myData.groupby(groupA)[targetA].describe()
    # two dataset to test paired samples
    SeriA = myData[targetA][groupA==int(div[0])]
    SeriB = myData[targetA][groupA==int(div[1])]
    s, p = stats.ttest_ind(setosa['sepal_width'], versicolor['sepal_width'])
    print("sValue is", s)
    print("pValue is", p)


def ANOVA(myData, groupA, divL, targetA):
    pass



def main():
    args = getArguments()
    myData = pd.read_csv(args.f, sep=',', encoding='latin1')
    if args.m=="ttest":
        tTest(myData, args.attr[0], args.div, args.attr[1])
    elif args.m == "anova":
        pass


if __name__ == '__main__':
    main()
