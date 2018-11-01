####################################
# Author: Jiachi Zhang
# E-mail: zhangjiachi1007@gmail.com
####################################

import argparse
import matplotlib.pyplot as plt
import pandas as pd

SAVE_PATH = './plot/'

def getArguments():
    '''
    get and parse command line arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='./dataset/crime2017.csv', help='the file path')
    parser.add_argument('--hvar', nargs='+', help="attributes' names to draw histgrams")
    parser.add_argument('--qvar', type=str, nargs='+', help="attributes' names to draw correlation graphs");
    return parser.parse_args()


def plotHist(myData, attribute_list, save_path):
    pass


def plotCor(myData, attributeList, save_path):
    pass


def main():
    args = getArguments()
    myData = pd.read_csv(args.f, sep=',', encoding='latin1')
    plotHist(myData, args.hvar, SAVE_PATH)
    plotCor(myData, args.qvar, SAVE_PATH)


if __name__ == '__main__':
    main()

