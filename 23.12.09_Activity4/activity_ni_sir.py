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
X = dataset.iloc[:, 0:4].values

# CREATE THE MATRIX OF DEPENDENT VARIABLE, Y (Salary)
Y = dataset.iloc[:, 4:5].values

# To view the scatter plot of the dataset
import seaborn as sns

sns.pairplot(dataset) 

# To determine the pearsons coefficient of correlation
from scipy.stats import pearsonr

# A. For the R&D Spend Vs. Profit
X_axis = dataset["R&D Spend"]
Y_axis = dataset["Profit"]

r_value, p_value = pearsonr(x = X_axis, y = Y_axis)
print("Pearson Coefficient of Correlation:", r_value)
print("P-value:", p_value)

# B. For the Aministration Spending Vs. Profit
X_axis = dataset["Administration"]
Y_axis = dataset["Profit"]

r_value, p_value = pearsonr(x = X_axis, y = Y_axis)
print("Pearson Coefficient of Correlation:", r_value)
print("P-value:", p_value)

# C. For the Marketing Spend Vs. Profit
X_axis = dataset["Marketing Spend"]
Y_axis = dataset["Profit"]

r_value, p_value = pearsonr(x = X_axis, y = Y_axis)
print("Pearson Coefficient of Correlation:", r_value)
print("P-value:", p_value)

dataset_correlation = dataset.corr(numeric_only = True)
sns.heatmap(dataset_correlation, annot = True, linewidth = 3)

# B. To Encode and Create the Dummy Variable for the Categorical Data in the Independent Variable, X (State)(array of float/int dapat)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer([("State", OneHotEncoder(categories = "auto"), [3])], remainder = "passthrough")
X = column_transformer.fit_transform(X)
X = X.astype(float)

# To avoid the Dummy Variable Trap fo the Ctaegorical Data (State) in the Independent Variable, X (for reg. only)

# Note: We remove the Column Index 0 of the Dimmy Variable
X_dummytrap = X.copy() # To Preserve the Original X Variable
X_dummytrap = X_dummytrap[:, 1:]

# TO SPLIT THE WHOLE DATASET INTO TRAINING DATASET AND TESTING DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_dummytrap, Y, train_size = 0.8, test_size = 0.2, random_state = 0)

# A. For Standardization Feature Scaling

from sklearn.preprocessing import StandardScaler # For the data that is NOT normally distributed

standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_train_standard[:, 2:5] = standard_scaler.fit_transform(X_train_standard[:, 2:5])

X_test_standard = X_test.copy()
X_test_standard[:, 2:5] = standard_scaler.transform(X_test_standard[:, 2:5])


# To Fit the Training Dataset into a Multiple Linear Regrression Model
from sklearn.linear_model import LinearRegression

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(X_train_standard, Y_train)

# To Predict the Output of the Testing dataset
Y_predict = multiple_linear_regression.predict(X_test_standard)


"""
# To Predict A particular Profit

# input = [R&D Administration Marketing State]

input_test = [[91992.39, 135495.07, 252664.93, "California"]]

input_test = [[1, 0, 0, 91992.39, 135495.07, 252664.93]]

input_test = [[0, 0, 91992.39, 135495.07, 252664.93]]

input_test[:, 2:5] = standard_scaler.transform(input_test[:, 2:5])

y_test = multiple_linear_regression.predict(input_test)

"""

# To Apply The K-fold Cross validation for the Multiple Linear Regression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from math import sqrt

k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)

# Try the following performance metrics
 # A. mean_absolute_error = "neg_mean_absolute_error"
 # B. mean_squared_error = "neg_mean_squared_error"
 # C. r-sqaured_error = "r2"
 

#To Feature scale the X_dummytrap_standard
X_dummytrap_standard = X_dummytrap.copy()
X_dummytrap_standard[:, 2:5] = standard_scaler.fit_transform(X_dummytrap_standard[:, 2:5])

def multiple_linear_regression_template():
      # A. For the mean absolute error (MAE) as scoring for cross validation
      MAE = (cross_val_score(multiple_linear_regression, X = X_dummytrap_standard, y = Y, cv = k_fold, scoring = "neg_mean_absolute_error"))*-1
      MAE_ave = MAE.mean()
      MAE_dev = MAE.std()

      print(" The Mean Absolute Error of KFolds:", MAE)
      print(" ")

      print(" The Average Mean Absolute Error of KFolds:", MAE_ave)
      print(" ")
      print(" The Standard Deviation Mean Absolute Error of KFolds:", MAE_dev)
      print(" ")

      # B. For the mean squared error (MSE) as scoring for cross validation

      MSE = (cross_val_score(multiple_linear_regression, X = X_dummytrap_standard, y = Y, cv = k_fold, scoring = "neg_mean_squared_error"))*-1
      MSE_ave = MSE.mean()
      MSE_dev = MSE.std()

      print(" The Mean Squared Error of KFolds:", MSE)
      print(" ")

      print(" The Average Mean Squared Error of KFolds:", MSE_ave)
      print(" ")
      print(" The Standard Deviation Mean Squared Error of KFolds:", MSE_dev)
      print(" ")

      # C. For the r-sqaured_error (R2) as scoring for cross validation

      r2 = (cross_val_score(multiple_linear_regression, X= X_dummytrap_standard, y = Y, cv = k_fold, scoring = "r2"))
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
      MAE_holdout = mean_absolute_error(Y_test, Y_predict)
      print("Mean Absolute Error: %4f"
            %MAE_holdout)
      print(" ")

      # B. For Mean Squared Error (MSE) 
      MSE_holdout = mean_squared_error(Y_test, Y_predict)
      print("Mean Squared Error: %4f"
            %MSE_holdout)
      print(" ")

      # C. For the Root Mean Squared Error/Deviation (RMSE/RMSD)
      RMSE_holdout = sqrt(MSE_holdout)
      print("Root Mean Squared Error: %4f"
            %RMSE_holdout)
      print(" ")

      # D. For Explained Variance Score (EVS)
      EVS_holdout = explained_variance_score(Y_test, Y_predict)
      print("Explained Variance Score: %4f"
            %EVS_holdout)
      print(" ")

      # E. For the Coefficient of Determination Regression Score Function (R^2)
      r2_holdout = r2_score(Y_test, Y_predict)
      print("R^2 Score: %4f"
            %r2_holdout)
      print(" ")


print("MODEL 1: ALL INDEPENDENT VARIABLES")
multiple_linear_regression_template()

'''
MODEL 2
'''

#To Feature scale the X_dummytrap_standard
X_dummytrap_standard = X_dummytrap.copy()
X_dummytrap_standard[:, 2:5] = standard_scaler.fit_transform(X_dummytrap_standard[:, 2:5])

# Get the 0th index and 2nd index of the X_dummytrap_standard
X_dummytrap_standard = X_dummytrap_standard[:, [0, 2]]

print("MODEL 2: R&D Spend and Marketing Spend")
multiple_linear_regression_template()

'''
MODEL 3
'''

#To Feature scale the X_dummytrap_standard
X_dummytrap_standard = X_dummytrap.copy()
X_dummytrap_standard[:, 2:5] = standard_scaler.fit_transform(X_dummytrap_standard[:, 2:5])

# Get the 0th index and 2nd index of the X_dummytrap_standard
X_dummytrap_standard = X_dummytrap_standard[:, [0]]

print("MODEL 3: R&D Spend")
multiple_linear_regression_template()
