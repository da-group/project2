####################################
# Author: Jiachi Zhang
# E-mail: zhangjiachi1007@gmail.com
####################################

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SAVE_PATH = './plot/'

def getArguments():
    '''
    get and parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='./dataset/crime2017_preprocessed.csv', help='the file path')
    parser.add_argument('--hvar', nargs='+', help="attributes' names to draw histgrams")
    parser.add_argument('--bins', nargs='+', help="bin number list with respect to values in hvar")
    parser.add_argument('--qvar', nargs='+', help="attributes' names to draw correlation graphs");
    return parser.parse_args()


def plotHist(myData, attribute_list, bin_num, save_path):
    for a, bn in zip(attribute_list, bin_num):
        array = np.array(myData[a])
        bins = [array.min()+(array.max()-array.min())*1.0*i/int(bn) for i in range(int(bn)+1)]
        plt.title(a)
        plt.xlabel('value')
        plt.ylabel('frequency')
        plt.hist(array, bins)
        plt.savefig(save_path+a+'.png')
        plt.show()


def plotCor(myData, attributeList, save_path):
    myData = myData[attributeList]
    pd.scatter_matrix(myData, diagonal='kde')
    name = save_path
    for a in attributeList:
        name += a+'_'
    name = name[:-1]
    plt.savefig(name+'.png')
    corr = myData.corr()
    plt.show()
    plt.figure()
    plt.imshow(corr)
    plt.colorbar()
    tick_marks = [i for i in range(len(myData.columns))]
    plt.xticks(tick_marks, myData.columns, rotation='vertical')
    plt.yticks(tick_marks, myData.columns)
    plt.show()


def main():
    args = getArguments()
    myData = pd.read_csv(args.f, sep=',', encoding='latin1')
    plotHist(myData, args.hvar, args.bins, SAVE_PATH)
    plotCor(myData, args.qvar, SAVE_PATH)


if __name__ == '__main__':
    main()

