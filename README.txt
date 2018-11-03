statistical_analysis.py: implement for the basic stastical analysis and further data preprocessing
       The basis stastical analyzing includes determining the mean (or mode), median, std and also the missing values of attributes
       The date preprocessing includes following:
              1. dropping useless attributes
              2. handling missing values
              3. binning numeric variables
              4. converting some nominal attributes to numerical attributes (or encoding)
       The preprocessed dataset will be saved with suffix "_preprocessed" in ./dataset
Usage: -f path of data set, default is ./dataset/crime2017_cleaned.csv
Example: python/python3 statistical_analysis.py -f ./dataset/crime2017_cleaned.csv


association_rules.py: implement for association rule mining, using the Apriori algorithm. 
(source of the package: https://pypi.python.org/pypi/apyori/1.1.1)
Usage: -f path of data, default is ./dataset/crime2017_cleaned.csv
       -s minimum support ratio (must be > 0, default: 0.1)
       -c minimum confidence (default: 0.5)
Example: python/python3 association_rules.py -f ./dataset/crime2017_cleaned.csv -s 0.2 -c 0.7


plot.py: implement to draw histograms and scatter matrix for attributes, the generated images will be saved in ./plot/
Usage: -f path of data, default is ./dataset/crime2017_preprocessed.csv
       --hvar attributes' names to draw histograms
       --bins bin numbers with respect to attributes, number of bins should be the same to number of hvar
       --qvar attributes' names to draw scatter plots
Example: python/python3 plot.py --hvar LATITUDE LONGITUDE --bins 10 10 --qvar LATITUDE LONGITUDE


stat.py: implement for t test, anova and logistic regression
Usage: -f path of data, default is ./dataset/crime2017_preprocessed.csv
       -m method to use, options are ttest, anova and logis
       --attr attributes to use when ttest or anova, first args should be grouped by attribute, second should be target attribute
       --div values of grouped by attribute, when ttest, we have to give 2 values after --div, when anova, we can give more than 2 values
       --lab attribute's name to use as label when building logistic regression
Example: python/python3 stat.py -m ttest --attr METHOD LONGITUDE --div 1 2

machineLearning.py
Usage: run directly, it will use linear regression, knn, naive Bayes, SVM, random forest and decision tree to train the model with data
crime2017_preprocessed.csv



