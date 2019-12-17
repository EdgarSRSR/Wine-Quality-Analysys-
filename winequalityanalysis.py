import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
# we use the LinearRegression function from sklearn
from sklearn.linear_model import LinearRegression
# we use the data splitting method for model from sklearn
from sklearn.model_selection import train_test_split
# import of mean square error
from sklearn.metrics import mean_squared_error
# import metrics for  error calculations
from sklearn import metrics
# imports the seaborn library for creating graphs in python
import seaborn as sns
# callling the GaussianNB
from sklearn.naive_bayes import GaussianNB
# for modifying CSV files
import csv
# confusion matrix
from sklearn.metrics import confusion_matrix

# this code modified the 'winequality-red.csv' and 'winequality-white.csv', the original document has 
# the values separated through ';' characters, but I think a CSV (comma-separated values) file is 
# better suited by being separated by commas. It also makes it easier to handle the data that way.

# with open('winequality-red.csv', newline='') as csvfile:
#	data = csvfile.read().replace(";",",");

# with open('winequality-red.csv', "w") as file:
#	file.write(data);

# with open('winequality-white.csv', newline='') as csvfilewhite:
#	data = csvfilewhite.read().replace(";",",");

# with open('winequality-white.csv', "w") as whitefile:
#	whitefile.write(data);

# allows pandas to show all the columns contained in the files
pd.set_option('display.expand_frame_repr', False)


'''Working with Red Wines
First Im going to work with the Multivariant Linear Regression. 
This is a good idea because we have a set of eleven parameters and 
we want to see how they affect the quality of the wines.'''

# shows the first five rows of the document with all the columns

# shows the first five rows of the Red Wines data 
redwines = pd.read_csv("winequality-red.csv")
shape = redwines.shape
print(shape)
head = redwines.head()
print(head)

print("Red Wine Database Analysis")

# I separated the Quality column from the rest of the data
# redata will have the characteristics of the red wine
redata = redwines.drop('quality', axis=1)
# redqual will have the quality of the red wines
redqual = redwines['quality']
#printing of data
print("show first rows of data for red wine")
print(redata.head())
print("show first rows of quality values for red wine")
print(redqual.head())

# plot a graph showing the distribution of the wines and their quality
plt.hist(redqual,bins=30)
plt.title('distribution of red wines and their quality')
plt.xlabel("quality red wines");
plt.ylabel("number of red wines")
plt.show()


# Shows the coefficient of how much the physico chemical characteristics influence in each other
print('Coefficient of physico-chemical characteristics in Red Wine')
correlation = redwines.corr()['quality'].drop('quality')
print(correlation)
# plots a heat map showing the influence of the coefficient in each other 
plt.figure(figsize=(12, 6))
plt.title('Coefficient of physico-chemical characteristics in Red Wine')
sns.heatmap(redwines.corr(), annot=True)
plt.show()


#Multivariant Linear Regression

print('*** Multivariant Linear Regression ***')

# we create our data sets for training and testing, 70% of the data will be for training
X_train, X_test, Y_train, Y_test = train_test_split(redata,redqual,train_size=.7)


# the multi variable linear regression function
multival_linreg = LinearRegression()
# fitting training data in the function
multival_linreg.fit(X_train,Y_train)
# set of predicted results using test data
testprediction = multival_linreg.predict(X_test)

print('Predictions using Multivariant Linear Regression')
# first five predictions
print('test prediction: ', testprediction[0:5])
# predicted results using training data
trainprediction = multival_linreg.predict(X_train)
# first five predictions
print('train prediction: ', trainprediction[0:5])


print("mean squared error: ")
# mean squared error with the training set
trainrmse = mean_squared_error(trainprediction, Y_train)**0.5
print('train meansquared ', trainrmse)
# mean squared error with the testig set
testrmse = mean_squared_error(testprediction, Y_test)**0.5
print('train meansquared ', testrmse)

# rounding the data from the test prediction
preddata = np.round_(testprediction)
print('predicted data for 5 first samples', preddata[0:5])
print('real values:\n', redqual[0:6])

# regression coefficients
print('coefficient new')
print(multival_linreg.coef_)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, testprediction))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, testprediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, testprediction)))
print('Accuracy Score:', (multival_linreg.score(X_test, Y_test))*100,'%')

# Working with White Wines

#For the White Wines I'm using the Naive Bayes Algorithm included in the SciKit library. 
#The Naive Bayes is used for classification of data. The highest ranking in white wines is 9, 
#so we are going to use the ranking as classes and see in which one can each wine be classified.

# gathering white data samples
whitewines = pd.read_csv("winequality-white.csv")

# shows the first five rows of the White Wines data 
# I separated the Quality column from the rest of the data
# whitedata will have the characteristics of the white wine
whitedata = whitewines.drop('quality', axis=1)
# whitequal will have the quality of the red wines
whitequal = whitewines['quality']

print("show first rows of data for red wine")
print(whitedata[0:5])
print("show first rows of quality values for red wine")
print(whitequal.head())

# plot a graph showing the distribution of the wines and their quality
plt.hist(whitequal,bins=30)
plt.title('distribution of white wines and their quality')
plt.xlabel("quality white wines");
plt.ylabel("number of white wines")
plt.show()

# Shows the coefficient of how much the physico chemical characteristics influence in each other
print('Coefficient of physico-chemical characteristics in White Wine')
correlation = whitewines.corr()['quality'].drop('quality')
print(correlation)
# plots a heat map showing the influence of the coefficient in each other 
plt.figure(figsize=(12, 6))
plt.title('Coefficient of physico-chemical characteristics in White Wine')
sns.heatmap(whitewines.corr(), annot=True)
plt.show()

# Naive Bayes Algorithm

# we create our data sets for training and testing, 70% of the data will be for training
Xg_train, Xg_test, Yg_train, Yg_test = train_test_split(whitedata,whitequal,train_size=.7)

print('*** Naive Bayes Algorithm ***')

#Create a Gaussian Classifier
gaussb = GaussianNB()

#Train the model using the training sets
gaussb.fit(Xg_train, Yg_train)

#Predict the response for test dataset
ypredgauss = gaussb.predict(Xg_test)
print('predictions using test data set for Gaussian Classifier')
print('test prediction', ypredgauss[0:5])

# predicted results using training data
predtrainingauss  = gaussb.predict(Xg_train)
# Predict response for training test
print('Comparing results for first 5 samples')
print('training prediction Gaussian Classifier', predtrainingauss[0:5])
print('actual quality values: \n', whitequal[0:5])

print("mean squared error for Gaussian Classifier: ")
# mean squared error with the training set
gtrainrmse = mean_squared_error(predtrainingauss, Yg_train)**0.5
print('train meansquared ', gtrainrmse)
# mean squared error with the testig set
gtestrmse = mean_squared_error(ypredgauss, Yg_test)**0.5
print('train meansquared ', gtestrmse)

print('Mean Absolute Error:', metrics.mean_absolute_error(Yg_test, ypredgauss))
print('Mean Squared Error:', metrics.mean_squared_error(Yg_test, ypredgauss))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Yg_test, ypredgauss)))

print("Accuracy of the Naive Bayes algorithm:",(metrics.accuracy_score(Yg_test, ypredgauss))*100,'%')

# Restructuring The Sample Data

# The results for the last algorithms have not been effective, 
# to fix this we can create new classes based in the range of Quality values 
# to make the data useful for class focuses algorithms

#New Classes for the Red Wines
print('the last algorithms have not been effective, to fix this we can create new classes based in the range of Quality values')
print('We create cluster the quality grades of The Red Wines into three classes: low, high, medium')
rquality = redwines["quality"].values
rcategory = []
for num in rquality:
    if num < 5:
        rcategory.append("low")
    elif num > 6:
        rcategory.append("high")
    else:
        rcategory.append("medium")
[(i, rcategory.count(i)) for i in set(rcategory)]
print("number of low quality red wines: ",rcategory.count("low"))
print("number of medium quality red wines: ",rcategory.count("medium"))
print("number of high quality red wines: ",rcategory.count("high"))

#  separate the quality from the other characteristics for the training
rcategory = pd.DataFrame(data=rcategory, columns=["category"])
rdata = pd.concat([redwines, rcategory], axis=1)
rdata.drop(columns="quality", axis=1, inplace=True)
x2 = rdata.iloc[:, :-1].values
y2 = rdata.iloc[:, -1].values

X2_train, X2_test, Y2_train, Y2_test = train_test_split(x2, y2, test_size=0.7)

#Train the model using the training sets
gaussb.fit(X2_train, Y2_train)

# Naive Bayes Algorithm Red Wines
print('Naive Bayes Algorithm applied with Red Wines clustered in Three classes')
#Predict the response for test dataset
y2pred = gaussb.predict(X2_test)
print('predictions using test data set')
print('test prediction Gaussian Classifier new classes: ', y2pred[0:5])

# predicted results using training data
predtraining2  = gaussb.predict(X2_train)
# Predict response for training test
print('for the first 5 samples in the data')
print('training prediction Gaussian Classifier new classes: ', predtraining2[0:5])
print('actual data base: ', rcategory[0:5])

print("Accuracy Gaussian Classifier with new classes:",(metrics.accuracy_score(Y2_test, y2pred))*100,'%')

print('Confusion Matrix for Red Wines')

# confusion matrix Red wines
print('Low Medium High')

cmred = confusion_matrix(rcategory[1:100], predtraining2[1:100], labels=["low", "medium", "high"])
print(cmred)


# New Classes for the White Wines

wquality = whitewines["quality"].values
wcategory = []
for num in wquality:
    if num < 5:
        wcategory.append("low")
    elif num > 7:
        wcategory.append("high")
    else:
        wcategory.append("medium")
[(i, wcategory.count(i)) for i in set(wcategory)]
print("number of low quality white wines: ",wcategory.count("low"))
print("number of medium quality white wines: ",wcategory.count("medium"))
print("number of high quality white wines: ",wcategory.count("high"))

#  prepare the data for training
# Naive Bayes Algorithm White Wines
print('Naive Bayes Algorithm applied with White Wines clustered in Three classes')
wcategory = pd.DataFrame(data=wcategory, columns=["category"])
wdata = pd.concat([whitewines, wcategory], axis=1)
wdata.drop(columns="quality", axis=1, inplace=True)
x3 = wdata.iloc[:, :-1].values
y3 = wdata.iloc[:, -1].values


X3_train, X3_test, Y3_train, Y3_test = train_test_split(x3, y3, test_size=0.7)

#Train the model using the training sets
gaussb.fit(X3_train, Y3_train)

#Predict the response for test dataset
y3pred = gaussb.predict(X3_test)
print('predictions using test data set')
print('test prediction Gaussian Classifier new classes white wines:', y3pred[0:5])

# predicted results using training data
predtraining3  = gaussb.predict(X3_train)
# Predict response for training test
print('for the first 5 samples in the database')
print('training prediction Gaussian Classifier new classes white wines: ', predtraining3[0:5])
print('actual data base', wcategory[0:5])

print("Accuracy Gaussian Classifier new classes White Wines: ",(metrics.accuracy_score(Y3_test, y3pred))*100,'%')


print('Confusion Matrix for White Wines')

# confusion matrix white wines

print('Low Medium High')

cmw = confusion_matrix(wcategory[1:100], predtraining3[1:100], labels=["low", "medium", "high"])
print(cmw)

