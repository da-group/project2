plot.py: implement to draw histograms and scatter matrix for attributes, the generated images will be saved in ./plot/
Usage: -f path of data, default is ./dataset/crime2017_cleaned_preprocessed.csv
       --hvar attributes' names to draw histograms
       --bins bin numbers with respect to attributes, number of bins should be the same to number of hvar
       --qvar attributes' names to draw scatter plots
Example: python/python3 plot.py --hvar LATITUDE LONGITUDE --bins 10 10 --qvar LATITUDE LONGITUDE


stat.py: implement for t test, anova and logistic regression
Usage: -f path of data, default is ./dataset/crime2017_cleaned_preprocessed.csv
       -m method to use, options are ttest, anova and logis
       --attr attributes to use when ttest or anova, first args should be grouped by attribute, second should be target attribute
       --div values of grouped by attribute, when ttest, we have to give 2 values after --div, when anova, we can give more than 2 values
       --lab attribute's name to use as label when building logistic regression
Example: python/python3 stat.py -m ttest --attr METHOD LONGITUDE --div 1 2



