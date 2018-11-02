####################################
# Author: Jiachi Zhang
# E-mail: zhangjiachi1007@gmail.com
####################################

import argparse
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def getArguments():
    '''
    get and parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='./dataset/crime2017_preprocessed.csv', help='the file path')
    parser.add_argument('-m', type=str, help="choose ttest or avano or logistic regression to use")
    parser.add_argument('--attr', nargs='+', help="first is attribute grouped by, second is target attribute")
    parser.add_argument('--div', nargs='+', help="two values in groupby attribute and they should be converted to int")
    parser.add_argument('--lab', type=str, help="label to predict")
    return parser.parse_args()


def tTest(myData, groupA, divL, targetA):
    # give a statistic analysis of relations of tow attributes
    myData.groupby(groupA)[targetA].describe()
    # two dataset to test paired samples
    SeriA = myData[myData[groupA]==int(divL[0])][targetA]
    SeriB = myData[myData[groupA]==int(divL[1])][targetA]
    plt.subplot(1, 2, 1)
    SeriA.plot(kind='hist', title='distribution of target when group is '+divL[0])
    plt.subplot(1, 2, 2)
    SeriB.plot(kind='hist', title='distribution of target when group is '+divL[1])
    plt.show()
    s, p = stats.ttest_ind(SeriA, SeriB)
    return s, p


def ANOVA(myData, groupA, divL, targetA):
    # give a statistic analysis using ANOVA
    myData.groupby(groupA)[targetA].describe()
    # get series list
    l = [myData[myData[groupA]==int(divL[i])][targetA] for i in range(len(divL))]
    s, p = stats.f_oneway(*l)
    return s, p



def logisticRegression(myData, labelA):
    model = LogisticRegression()
    # convert to numpy array
    X = myData.values
    y = myData[labelA].values.reshape(-1)
    # divide into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    model.fit(X_train, y_train)
    # predict
    pred = model.predict(X_test)
    # evaluate
    acc = model.score(X_test, y_test)
    cm = metrics.confusion_matrix(y_test, pred)
    return pred, acc, cm



def main():
    args = getArguments()
    myData = pd.read_csv(args.f, sep=',', encoding='latin1')
    if args.m == "ttest":
        s, p = tTest(myData, args.attr[0], args.div, args.attr[1])
        print('using t test, statistic value is', s, 'p value is', p)
    elif args.m == "anova":
        s, p = ANOVA(myData, args.attr[0], args.div, args.attr[1])
        print('using anova, statistic value is', s, 'p value is', p)
    elif args.m == "logis":
        pred, acc, cm = logisticiRegression(myData, args.lab)
        print("accuracy is", acc)
        print("confusion matrix is")
        print(cm)


if __name__ == '__main__':
    main()
