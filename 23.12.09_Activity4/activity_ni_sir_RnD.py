# MULTIPLE LINEAR REGRESSION TEMPLATE

# To Import Libraries
import numpy as np #numerical
import pandas as pd #visualization
import matplotlib.pyplot as plt #data manipulation

# To Import the dataset
dataset = pd.read_csv("50_startups.csv")

# Preliminary Analysis of the Dataset

# A. To Know How Much of the Data is Missing
missing_data = dataset.isnull().sum().sort_values(ascending = False)

# B. To Check Column Names and Total Records
dataset_count = dataset.count()

# C. To view the information about the dataset
print(dataset.info())

# D. To view the statistical summary of the dataset
statisctic = dataset.describe(include = 'all')

#  CREATE THE MATRIX OF INDEPENDENT VARIABLE, X (YearsExperience)
X = dataset.iloc[:, [0]].values

# CREATE THE MATRIX OF DEPENDENT VARIABLE, Y (Salary)
Y = dataset.iloc[:, 4:5].values

# To view the scatter plot of the dataset
import seaborn as sns

sns.pairplot(dataset) 

# To determine the pearsons coefficient of correlation
from scipy.stats import pearsonr

print("MODEL 3: R&D Spend")

# A. For the R&D Spend Vs. Profit
X_axis = dataset["R&D Spend"]
Y_axis = dataset["Profit"]

r_value, p_value = pearsonr(x = X_axis, y = Y_axis)
print("Pearson Coefficient of Correlation:", r_value)
print("P-value:", p_value)


dataset_correlation = dataset.corr(numeric_only = True)
sns.heatmap(dataset_correlation, annot = True, linewidth = 3)

# TO SPLIT THE WHOLE DATASET INTO TRAINING DATASET AND TESTING DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 0)


# To Fit the Training Dataset into a Multiple Linear Regrression Model
from sklearn.linear_model import LinearRegression

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(X_train, Y_train)

# To Predict the Output of the Testing dataset
Y_predict = multiple_linear_regression.predict(X_test)



# To Apply The K-fold Cross validation for the Multiple Linear Regression
from sklearn.model_selection import KFold

k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)

# Try the following performance metrics
 # A. mean_absolute_error = "neg_mean_absolute_error"
 # B. mean_squared_error = "neg_mean_squared_error"
 # C. r-sqaured_error = "r2"
 
from sklearn.model_selection import cross_val_score



# A. For the mean absolute error (MAE) as scoring for cross validation
MAE = (cross_val_score(multiple_linear_regression, X = X, y = Y, cv = k_fold, scoring = "neg_mean_absolute_error"))*-1
MAE_ave = MAE.mean()
MAE_dev = MAE.std()

print(" The Mean Absolute Error of KFolds:", MAE)
print(" ")

print(" The Average Mean Absolute Error of KFolds:", MAE_ave)
print(" ")
print(" The Standard Deviation Mean Absolute Error of KFolds:", MAE_dev)
print(" ")

# B. For the mean squared error (MSE) as scoring for cross validation

MSE = (cross_val_score(multiple_linear_regression, X = X, y = Y, cv = k_fold, scoring = "neg_mean_squared_error"))*-1
MSE_ave = MSE.mean()
MSE_dev = MSE.std()

print(" The Mean Squared Error of KFolds:", MSE)
print(" ")

print(" The Average Mean Squared Error of KFolds:", MSE_ave)
print(" ")
print(" The Standard Deviation Mean Squared Error of KFolds:", MSE_dev)
print(" ")

# C. For the r-sqaured_error (R2) as scoring for cross validation

r2 = (cross_val_score(multiple_linear_regression, X= X, y = Y, cv = k_fold, scoring = "r2"))
r2_ave = r2.mean()
r2_dev = r2.std()

print(" The Mean R-Squared Error of KFolds:", r2)
print(" ")

print(" The Average Mean R-Squared Error of KFolds:", r2_ave)
print(" ")
print(" The Standard Deviation Mean R-Squared Error of KFolds:", r2_dev)
print(" ")

#To Evaluate the performance of SLR Using Hold-Out Validation

# A. For the Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error
MAE_holdout = mean_absolute_error(Y_test, Y_predict)
print("Mean Absolute Error: %4f"
      %MAE_holdout)
print(" ")

# B. For Mean Squared Error (MSE) 
from sklearn.metrics import mean_squared_error
MSE_holdout = mean_squared_error(Y_test, Y_predict)
print("Mean Squared Error: %4f"
      %MSE_holdout)
print(" ")

# C. For the Root Mean Squared Error/Deviation (RMSE/RMSD)
from math import sqrt
RMSE_holdout = sqrt(MSE_holdout)
print("Root Mean Squared Error: %4f"
      %RMSE_holdout)
print(" ")

# D. For Explained Variance Score (EVS)
from sklearn.metrics import explained_variance_score
EVS_holdout = explained_variance_score(Y_test, Y_predict)
print("Explained Variance Score: %4f"
      %EVS_holdout)
print(" ")

# E. For the Coefficient of Determination Regression Score Function (R^2)
from sklearn.metrics import r2_score
r2_holdout = r2_score(Y_test, Y_predict)
print("R^2 Score: %4f"
      %r2_holdout)
print(" ")



