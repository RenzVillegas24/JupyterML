# SIMPLE LINEAR REGRESSION - DATA PREPROCESSING


# %% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %% Import the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Analysis of the dataset 
# %% A. To know if there is any missing data
missing_data = dataset.isnull().sum().sort_values(ascending=False)

# %% B. To check collumn names and total records
dataset_count = dataset.count()

# %% C. TO view the information about the dataset
dataset_info = dataset.info()

# %% D. To view the statistical summary of the dataset
dataset_describe = dataset.describe()

# %% To create the matrix of the INDEPENDENT VARIABLE, X (YearsExperience)
X = dataset.iloc[:, :-1].values # All rows, all columns except the last one

# %% To create the matrix of the DEPENDENT VARIABLE, Y (Salary)
Y = dataset.iloc[:, 1].values # All rows, only the last column

# %% To view the scatter plot of the dataset
import seaborn as sns

sns.jointplot(x='YearsExperience', y='Salary', data=dataset, kind='reg')
sns.pairplot(dataset)

# %% To determine the Pearson's coefficient of correlation
from scipy.stats import pearsonr

X_axis = dataset['YearsExperience']
Y_axis = dataset['Salary']

r_value, p_value = pearsonr(x = X_axis, y = Y_axis)
print("Peason's coefficient of correlation:", r_value)
print()
print("p-value:", p_value)

dataset_correlation = dataset.corr()


# %% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %% To fit the training dataset into a Simple Linear Regression Model
from sklearn.linear_model import LinearRegression

simple_linear_regression = LinearRegression()
simple_linear_regression.fit(X_train, Y_train)

# %% To predict the Test set results
Y_pred = simple_linear_regression.predict(X_test)

# %% To vvisualize the Training dataset and the Simple Linear Regression Model
import matplotlib.patches as mpatches

plt.scatter(X_train, Y_train, color='red')

Y_predicted_xtrain = simple_linear_regression.predict(X_train) # Predicted values of Y_train
plt.plot(X_train, Y_predicted_xtrain, color='blue') # Plotting the regression line
plt.title('Salary vs Experience versus Salary using the Training Dataset')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

red_patch = mpatches.Patch(color='red', label='Training Dataset')
blue_patch = mpatches.Patch(color='blue', label='Simple Linear Regression Model')

plt.legend(handles=[red_patch, blue_patch])
plt.show()
# %% Show the actual values of Y_train and the predicted values of Y_train


plt.scatter(X_test, Y_test, color='green')
Y_predicted_xtest = simple_linear_regression.predict(X_test) # Predicted values of Y_train
plt.scatter(X_test, Y_predicted_xtest, color='yellow')

plt.plot(X_test, Y_predicted_xtest, color='blue') # Plotting the regression line
plt.title('Plot of Years of Experience Versus Salary using Testing Dataset')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

green_patch = mpatches.Patch(color='green', label='Actual Salary')
blue_patch = mpatches.Patch(color='blue', label='Simple Linear Regression Model')
yellow_patch = mpatches.Patch(color='yellow', label='Predicted Salary')
plt.legend(handles=[green_patch, blue_patch, yellow_patch])
plt.show()
# %% To determine the constant and the coefficient of the Simple Linear Regression Model
constant = simple_linear_regression.intercept_
coefficient = simple_linear_regression.coef_

print("The constant value (b0) is: %.2f" % constant)
print("The coefficient value (b1) is: %.2f" % coefficient)

# %% To predict a particular salary
Y_predicted_particular = simple_linear_regression.predict([[28.5]])
print("The predicted salary for 28.5 years of experience is: %.2f" % Y_predicted_particular)

# %% To apply the k-Fold Cross Validation for the simple linear regression model
from sklearn.model_selection import KFold

K_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# %% Try the following performance metrics:
# 1. Mean Absolute Error (MAE) = "neg_mean_absolute_error"
# 2. Mean Squared Error (MSE) = "neg_mean_squared_error"
# 3. R2 Score = "r2"
from sklearn.model_selection import cross_val_score

# %% A. Mean Absolute Error (MAE) as the performance metric
MAE = cross_val_score(estimator=simple_linear_regression, X=X, y=Y, cv=K_fold, scoring="neg_mean_absolute_error")
MAE_average = MAE.mean()
MAE_deviation = MAE.std()

print("Absolute Error of k-fols is:", MAE)
print("The Mean Absolute Error (MAE) is: %.4f" % MAE_average)
print("The Standard Deviation of the Mean Absolute Error (MAE) is: %.4f" % MAE_deviation)


# %% B. Mean Squared Error (MSE) as the performance metric
MSE = cross_val_score(estimator=simple_linear_regression, X=X, y=Y, cv=K_fold, scoring="neg_mean_squared_error")
MSE_average = MSE.mean()
MSE_deviation = MSE.std()

print("Absolute Error of k-fols is:", MSE)
print("The Mean Absolute Error (MAE) is: %.4f" % MSE_average)
print("The Standard Deviation of the Mean Absolute Error (MAE) is: %.4f" % MSE_deviation)


# %% C. R-squared Score as the performance metric
R2 = cross_val_score(estimator=simple_linear_regression, X=X, y=Y, cv=K_fold, scoring="r2")
R2_average = R2.mean()
R2_deviation = R2.std()

print("Absolute Error of k-fols is:", R2)
print("The Mean Absolute Error (MAE) is: %.4f" % R2_average)
print("The Standard Deviation of the Mean Absolute Error (MAE) is: %.4f" % R2_deviation)
# %% A. For the Maean Absolute Error (MAE) 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
MAE_holdout = mean_absolute_error(Y_test, Y_pred)
print("The Mean Absolute Error (MAE) is: %.4f" % MAE_holdout)
print()

# %% For the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
MSE_holdout = mean_squared_error(Y_test, Y_pred)
print("The Mean Squared Error (MSE) is: %.4f" % MSE_holdout)

# %% C. For the R-squared Score
from math import sqrt
RMSE_holdout = sqrt(MSE_holdout)
print("The Root Mean Squared Error (RMSE) is: %.4f" % RMSE_holdout)
print()
# %% D. For the Explained Variance Score (EVS)
from sklearn.metrics import explained_variance_score
EVS_holdout = explained_variance_score(Y_test, Y_pred)
print("The Explained Variance Score (EVS) is: %.4f" % EVS_holdout)
print()
# %% E. For the Coefficient of Determination (R2)
from sklearn.metrics import r2_score
R2_holdout = r2_score(Y_test, Y_pred)
print("The Coefficient of Determination (R2) is: %.4f" % R2_holdout)
print()

# %%
