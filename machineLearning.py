import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np

def load(filename):
    attributes = ['CCN', 'REPORT_DAT', 'SHIFT', 'METHOD', 'OFFENSE', 'BLOCK', 'XBLOCK', 'YBLOCK', 'WARD', 'ANC', 'DISTRICT', 'PSA', 'NEIGHBORHOOD_CLUSTER', 'BLOCK_GROUP', 'CENSUS_TRACT', 'VOTING_PRECINCT', 'LATITUDE', 'LONGITUDE', 'START_DATE', 'END_DATE', 'PSA_bin']
    return pd.read_csv(filename, names=attributes)

def binning(data):
    #binning the results, without this the accuracy is too low to accept
    #to make roc work, we need divide the Y in two parts
    bins = [0, 2, 4]
    data = pd.cut(data, bins, labels=[0, 1])
    data = np.array(data).reshape(-1,1).ravel()
    return data

def separateData(mydata):
    #separate data to train set and test set
    columns = ['METHOD', 'OFFENSE', 'DISTRICT', 'PSA', 'VOTING_PRECINCT', 'SHIFT']
    myData = pd.DataFrame(mydata, columns=columns)
    myData = pd.DataFrame(myData[1:], dtype='float64')
    print(myData.dtypes)
    valueArray = myData[1:3000].values
    X = valueArray[:, 0:4]
    Y = valueArray[:, 5]
    Y = binning(Y)
    test_size = 0.20
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    return X_train, X_test, Y_train, Y_test

def training(X_train, Y_train):
    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
    num_instances = len(X_train)
    seed = 7
    scoring = 'accuracy'
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NaiveBayes', MultinomialNB()))
    models.append(('RDForest', RandomForestClassifier()))
    models.append(('SVM', SVC()))

    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std)
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        # cross validation
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

def linear(X_train, Y_train, X_validate, Y_validate):
    #linear regression evaluated by MSE
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    prediction = lr.predict(X_validate)
    mse = mean_squared_error(Y_validate, prediction)
    print('Linear regression: ')
    print('MSE: ', mse)

def knnLearner(X_train, Y_train, X_validate, Y_validate):
    #knn classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    prediction = knn.predict(X_validate)
    fpr, tpr, threshold = roc_curve(Y_validate, prediction)
    plt.plot(fpr, tpr, color='darkorange')
    plt.title('KNN classifier')
    plt.savefig('knn.jpg')

    print('KNN Classifier:')
    print(accuracy_score(Y_validate, prediction))
    print(confusion_matrix(Y_validate, prediction))
    print(classification_report(Y_validate, prediction))

def decisionTree(X_train, Y_train, X_validate, Y_validate):
    #cat decision tree
    cat = DecisionTreeClassifier()
    cat.fit(X_train, Y_train)
    prediction = cat.predict(X_validate)
    fpr, tpr, threshold = roc_curve(Y_validate, prediction)
    plt.plot(fpr, tpr, color='darkorange')
    plt.title('CAT classifier')
    plt.savefig('cat.jpg')

    print('CAT Classifier:')
    print(accuracy_score(Y_validate, prediction))
    print(confusion_matrix(Y_validate, prediction))
    print(classification_report(Y_validate, prediction))

def naiveB(X_train, Y_train, X_validate, Y_validate):
    #naiveBayes
    nb = MultinomialNB()
    nb.fit(X_train, Y_train)
    prediction = nb.predict(X_validate)
    fpr, tpr, threshold = roc_curve(Y_validate, prediction)
    plt.plot(fpr, tpr, color='darkorange')
    plt.title('naive Bayes classifier')
    plt.savefig('nb.jpg')

    print('Naive bayes Classifier:')
    print(accuracy_score(Y_validate, prediction))
    print(confusion_matrix(Y_validate, prediction))
    print(classification_report(Y_validate, prediction))

def randomForest(X_train, Y_train, X_validate, Y_validate):
    #random forest
    ranf = RandomForestClassifier()
    ranf.fit(X_train, Y_train)
    prediction = ranf.predict(X_validate)
    fpr, tpr, threshold = roc_curve(Y_validate, prediction)
    plt.plot(fpr, tpr, color='darkorange')
    plt.title('Random forest classifier')
    plt.savefig('rf.jpg')

    print('Random forest Classifier:')
    print(accuracy_score(Y_validate, prediction))
    print(confusion_matrix(Y_validate, prediction))
    print(classification_report(Y_validate, prediction))

def svmClassifier(X_train, Y_train, X_validate, Y_validate):
    #svm
    svm = RandomForestClassifier()
    svm.fit(X_train, Y_train)
    prediction = svm.predict(X_validate)
    fpr, tpr, threshold = roc_curve(Y_validate, prediction)
    plt.plot(fpr, tpr, color='darkorange')
    plt.title('SVM classifier')
    plt.savefig('svm.jpg')

    print('SVM Classifier:')
    print(accuracy_score(Y_validate, prediction))
    print(confusion_matrix(Y_validate, prediction))
    print(classification_report(Y_validate, prediction))

if __name__ == '__main__':
    mydata = load('./dataset/crime2017_preprocessed.csv')
    X_train, X_test, Y_train, Y_test = separateData(mydata)
    training(X_train, Y_train)
    knnLearner(X_train, Y_train, X_test, Y_test)
    linear(X_train, Y_train, X_test, Y_test)
    svmClassifier(X_train, Y_train, X_test, Y_test)
    naiveB(X_train, Y_train, X_test, Y_test)
    randomForest(X_train, Y_train, X_test, Y_test)