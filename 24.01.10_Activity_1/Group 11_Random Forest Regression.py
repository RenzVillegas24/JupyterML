#!/usr/bin/env python
# coding: utf-8

# <center>
#     <h1><strong>GROUP 11</strong></h1>
#     <h2>The Random Forest Regression Model</h2>
# </center>
# 
# <h3>Group Members</h3>
# Onte, Michael Jethro M.<br>
# Sunga, Jullianne Christille N.<br>
# Villegas, Renz Justine L.<br>

# <h1>Importing Required Libraries</h1>
# This file contains all the libraries that are required for the project.

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
# import random forest regressor
from sklearn.ensemble import RandomForestRegressor


# for showing the proper table

# <h1>Loading and Viewing Data for Cross Validation in 10 folds</h1>
# Import the database for all the features.

# In[2]:


# read the data
df_norm = pd.read_csv('/mnt/c/Users/RenzCute/Codes/Jupyter/24.01.10_Activity_1/Group 11_CompleteDataset.csv')
df_norm


# <h2>Independent and Dependent Variables.</h2>
# Creates a matrix of independent variables and a matrix of dependent variables.

# In[3]:



# create a matrix of dependent variables
X_norm = df_norm.drop(['Price'], axis=1).values


# create a matrix of independent variables
y_norm = df_norm['Price'].values



# <h2>Training and Testing Datasets</h2>
# Split the data into training and testing datasets

# In[4]:


X_norm_train, X_norm_test, Y_norm_train, Y_norm_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=0)

# <h1>Random Forest Regressor</h1>
# Create a random forest regressor model then fit the training datasets

# In[5]:


# create a random forest regressor
norm_random_forest = RandomForestRegressor()
norm_random_forest.fit(X_norm_train, Y_norm_train)

# predict the test set results
Y_norm_pred = norm_random_forest.predict(X_norm_test)

# <h2>K-Fold Cross Validation</h2>
# Use k-fold cross validation to get the accuracy of the model

# In[6]:


norm_kfold = KFold(n_splits=10, random_state=0, shuffle=True)

# <h3>Mean Absolute Error</h3>
# Using mean absolute error of the model as scoring for cross validation

# In[7]:


# calculate the mean absolute error of the model as scoring for cross validation
norm_MAE = -cross_val_score(norm_random_forest, X_norm, y_norm, cv=norm_kfold, scoring='neg_mean_absolute_error')


print( f"Normal Mean Absolute Error of k-Folds: \033[1m\033[94m{(norm_MAE)}\033[0m")
print( f"Average Normal Mean Absolute Error of k-Folds: \033[1m\033[94m{(norm_MAE.mean())}\033[0m")
print( f"Standard Deviation of Normal Mean Absolute Error of k-Folds: \033[1m\033[94m{(norm_MAE.std())}\033[0m")

# <h3>Mean Squared Error</h3>
# Using mean squared error of the model as scoring for cross validation

# In[8]:


norm_MSE = -cross_val_score(norm_random_forest, X_norm, y_norm, cv=norm_kfold, scoring='neg_mean_squared_error')

print( f"Normal Mean Squared Error of k-Folds: \033[1m\033[94m{(norm_MSE)}\033[0m")
print( f"Average Normal Mean Squared Error of k-Folds: \033[1m\033[94m{(norm_MSE.mean())}\033[0m")
print( f"Standard Deviation of Normal Mean Squared Error of k-Folds: \033[1m\033[94m{(norm_MSE.std())}\033[0m")

# <h3>R-Squared</h3>
# Using r-squared of the model as scoring for cross validation

# In[9]:


norm_R2 = cross_val_score(norm_random_forest, X_norm, y_norm, cv=norm_kfold, scoring='r2')

print( f"Normal R2 of k-Folds: \033[1m\033[94m{(norm_R2)}\033[0m")
print( f"Average Normal R2 of k-Folds: \033[1m\033[94m{(norm_R2.mean())}\033[0m")
print( f"Standard Deviation of Normal R2 of k-Folds: \033[1m\033[94m{(norm_R2.std())}\033[0m")

# <h3>Explained Variance Score</h3>
# Using explained variance score of the model as scoring for cross validation

# In[10]:


norm_EV = cross_val_score(norm_random_forest, X_norm, y_norm, cv=norm_kfold, scoring='explained_variance')

print( f"Normal Explained Variance of k-Folds: \033[1m\033[94m{(norm_EV)}\033[0m")
print( f"Average Normal Explained Variance of k-Folds: \033[1m\033[94m{(norm_EV.mean())}\033[0m")
print( f"Standard Deviation of Normal Explained Variance of k-Folds: \033[1m\033[94m{(norm_EV.std())}\033[0m")

# <h3>Root Mean Squared Error</h3>
# Using root mean squared error of the model as scoring for cross validation

# In[11]:


norm_RMSE = np.sqrt(norm_MSE)

print( f"Normal Root Mean Squared Error: \033[1m\033[94m{(norm_RMSE)}\033[0m")
print( f"Average Normal Root Mean Squared Error: \033[1m\033[94m{(norm_RMSE.mean())}\033[0m")
print( f"Standard Deviation of Normal Root Mean Squared Error: \033[1m\033[94m{(norm_RMSE.std())}\033[0m")

# <h2>Hold-Out Validation</h2>
# Use hold-out validation to evaluate the model

# <h3>Mean Absolute Error</h3>
# Using mean absolute error of the model as scoring for hold-out validation <i>(the best value is 0.0)</i>

# In[12]:


norm_MAE_holdout = mean_absolute_error(Y_norm_test, Y_norm_pred)
color = "\033[92m" if norm_MAE_holdout < 0.5 else "\033[91m"

print( f"Normal Mean Absolute Error of Holdout: \033[1m{color}{(norm_MAE_holdout)}\033[0m")

# <h3>Mean Squared Error</h3>
# Using mean squared error of the model as scoring for hold-out validation <i>(the best value is 0.0)</i>

# In[13]:


norm_MSE_holdout = mean_squared_error(Y_norm_test, Y_norm_pred)
color = "\033[92m" if norm_MSE_holdout < 0.5 else "\033[91m"

print( f"Normal Mean Squared Error of Holdout: \033[1m{color}{(norm_MSE_holdout)}\033[0m")

# <h3>R-Squared Error</h3>
# Using r-squared error of the model as scoring for hold-out validation <i>(the best value is 1.0)</i>

# In[14]:


norm_R2_holdout = r2_score(Y_norm_test, Y_norm_pred)
color = "\033[92m" if norm_R2_holdout > 0.5 else "\033[91m"

print( f"Normal R2 of Holdout: \033[1m{color}{(norm_R2_holdout)}\033[0m")

# <h3>Explained Variance Score</h3>
# Using explained variance score of the model as scoring for hold-out validation  <i>(the best value is 1.0)</i>

# In[15]:


norm_EV_holdout = explained_variance_score(Y_norm_test, Y_norm_pred)
color = "\033[92m" if norm_EV_holdout > 0.5 else "\033[91m"

print( f"Normal Explained Variance of Holdout: \033[1m{color}{(norm_EV_holdout)}\033[0m")

# <h3>Root Mean Squared Error</h3>
# Using root mean squared error of the model as scoring for hold-out validation  <i>(the best value is 0.0)</i>

# In[16]:


norm_RMSE_holdout = np.sqrt(norm_MSE_holdout)
color = "\033[92m" if norm_RMSE_holdout < 0.5 else "\033[91m"

print( f"Normal Root Mean Squared Error of Holdout: \033[1m{color}{(norm_RMSE_holdout)}\033[0m")

# <h1>Loading and Viewing Data for Cross Validation in 10 folds</h1>
# Import the database with the features selected.

# In[17]:


# read the data
df_sel = pd.read_csv('/mnt/c/Users/RenzCute/Codes/Jupyter/24.01.10_Activity_1/Group 11_Dataset.csv')
df_sel


# <h2>Independent and Dependent Variables.</h2>
# Creates a matrix of independent variables and a matrix of dependent variables.

# In[18]:



# create a matrix of dependent variables
X_sel = df_sel.drop(['Price'], axis=1).values

# create a matrix of independent variables
Y_sel = df_sel['Price'].values



# <h2>Training and Testing Datasets</h2>
# Split the data into training and testing datasets

# In[19]:


X_sel_train, X_sel_test, Y_sel_train, Y_sel_test = train_test_split(X_sel, Y_sel, test_size=0.2, random_state=0)


# <h1>Random Forest Regressor</h1>
# Create a random forest regressor model then fit the training datasets

# In[20]:


# create a random forest regressor model
sel_random_forest = RandomForestRegressor()
sel_random_forest.fit(X_sel_train, Y_sel_train)

# predict the test set results
Y_sel_pred = sel_random_forest.predict(X_sel_test)

# # CONCLUSION
# Based on the results, the model with the **selected features** performed better than the model with all the features. <br>
# The model with the **selected features** has a lower **mean absolute error**, **mean squared error**, **root mean squared error**, and higher **R2** and **explained variance**.
# 
# The **Support Vector Regression** model was the 3rd best model for the selected features, with the **Random Forest Regression** model being the 2nd and **Decision Tree Regression** model being the best.
# 

# <h2>K-Fold Cross Validation</h2>
# Use k-fold cross validation to get the accuracy of the model

# In[21]:


sel_kfold = KFold(n_splits=10, random_state=0, shuffle=True)

# <h3>Mean Absolute Error</h3>
# Using mean absolute error of the model as scoring for cross validation

# In[22]:


# calculate the mean absolute error of the model as scoring for cross validation
sel_MAE = -cross_val_score(sel_random_forest, X_sel, Y_sel, cv=sel_kfold, scoring='neg_mean_absolute_error')

print(f"Selected Feastures Mean Absolute Error of k-Folds: \033[1m\033[94m{(sel_MAE)}\033[0m")
print(f"Average Selected Feastures Mean Absolute Error of k-Folds: \033[1m\033[94m{(sel_MAE.mean())}\033[0m")
print(f"Standard Deviation of Selected Feastures Mean Absolute Error of k-Folds: \033[1m\033[94m{(sel_MAE.std())}\033[0m")


# <h3>Mean Squared Error</h3>
# Using mean squared error of the model as scoring for cross validation

# In[23]:


sel_MSE = -cross_val_score(sel_random_forest, X_sel, Y_sel, cv=sel_kfold, scoring='neg_mean_squared_error')

print( f"Selected Feastures Mean Squared Error of k-Folds: \033[1m\033[94m{(sel_MSE)}\033[0m")
print( f"Average Selected Feastures Mean Squared Error of k-Folds: \033[1m\033[94m{(sel_MSE.mean())}\033[0m")
print( f"Standard Deviation of Selected Feastures Mean Squared Error of k-Folds: \033[1m\033[94m{(sel_MSE.std())}\033[0m")

# <h3>R-Squared</h3>
# Using r-squared of the model as scoring for cross validation

# In[24]:


sel_R2 = cross_val_score(sel_random_forest, X_sel, Y_sel, cv=sel_kfold, scoring='r2')

print( f"Selected Feastures R2 of k-Folds: \033[1m\033[94m{(sel_R2)}\033[0m")
print( f"Average Selected Feastures R2 of k-Folds: \033[1m\033[94m{(sel_R2.mean())}\033[0m")
print( f"Standard Deviation of Selected Feastures R2 of k-Folds: \033[1m\033[94m{(sel_R2.std())}\033[0m")

# <h3>Explained Variance Score</h3>
# Using explained variance score of the model as scoring for cross validation

# In[25]:


sel_EV = cross_val_score(sel_random_forest, X_sel, Y_sel, cv=sel_kfold, scoring='explained_variance')

print( f"Selected Feastures Explained Variance of k-Folds: \033[1m\033[94m{(sel_EV)}\033[0m")
print( f"Average Selected Feastures Explained Variance of k-Folds: \033[1m\033[94m{(sel_EV.mean())}\033[0m")
print( f"Standard Deviation of Selected Feastures Explained Variance of k-Folds: \033[1m\033[94m{(sel_EV.std())}\033[0m")

# <h3>Root Mean Squared Error</h3>
# Using root mean squared error of the model as scoring for cross validation

# In[26]:


sel_RMSE = np.sqrt(sel_MSE)

print( f"Selected Feastures Root Mean Squared Error: \033[1m\033[94m{(sel_RMSE)}\033[0m")
print( f"Average Selected Feastures Root Mean Squared Error: \033[1m\033[94m{(sel_RMSE.mean())}\033[0m")
print( f"Standard Deviation of Selected Feastures Root Mean Squared Error: \033[1m\033[94m{(sel_RMSE.std())}\033[0m")

# <h2>Hold-Out Validation</h2>
# Use hold-out validation to evaluate the model

# <h3>Mean Absolute Error</h3>
# Using mean absolute error of the model as scoring for hold-out validation <i>(the best value is 0.0)</i>

# In[27]:


sel_MAE_holdout = mean_absolute_error(Y_sel_test, Y_sel_pred)
color = "\033[92m" if sel_MAE_holdout < 0.5 else "\033[91m"

print( f"Selected Feastures Mean Absolute Error of Holdout: \033[1m{color}{(sel_MAE_holdout)}\033[0m")

# <h3>Mean Squared Error</h3>
# Using mean squared error of the model as scoring for hold-out validation <i>(the best value is 0.0)</i>

# In[28]:


sel_MSE_holdout = mean_squared_error(Y_sel_test, Y_sel_pred)
color = "\033[92m" if sel_MSE_holdout < 0.5 else "\033[91m"

print( f"Selected Feastures Mean Squared Error of Holdout: \033[1m{color}{(sel_MSE_holdout)}\033[0m")

# <h3>R-Squared Error</h3>
# Using r-squared error of the model as scoring for hold-out validation <i>(the best value is 1.0)</i>

# In[29]:


sel_R2_holdout = r2_score(Y_sel_test, Y_sel_pred)
color = "\033[92m" if sel_R2_holdout > 0.5 else "\033[91m"

print( f"Selected Feastures R2 of Holdout: \033[1m{color}{(sel_R2_holdout)}\033[0m")

# <h3>Explained Variance Score</h3>
# Using explained variance score of the model as scoring for hold-out validation  <i>(the best value is 1.0)</i>

# In[30]:


sel_EV_holdout = explained_variance_score(Y_sel_test, Y_sel_pred)
color = "\033[92m" if sel_EV_holdout > 0.5 else "\033[91m"

print( f"Selected Feastures Explained Variance of Holdout: \033[1m{color}{(sel_EV_holdout)}\033[0m")

# <h3>Root Mean Squared Error</h3>
# Using root mean squared error of the model as scoring for hold-out validation  <i>(the best value is 0.0)</i>

# In[31]:


sel_RMSE_holdout = np.sqrt(sel_MSE_holdout)
color = "\033[92m" if sel_RMSE_holdout < 0.5 else "\033[91m"

print( f"Selected Feastures Root Mean Squared Error of Holdout: \033[1m{color}{(sel_RMSE_holdout)}\033[0m")

# # CONCLUSION
# Based on the results, the model with the **selected features** performed better than the model with all the features. <br>
# The model with the **selected features** has a lower **mean absolute error**, **mean squared error**, **root mean squared error**, and higher **R2** and **explained variance**.
# 
# The **Random Forest Regression** model was the 2nd best model for the selected features, with the  **Support Vector Regression** model being the 3rd and **Decision Tree Regression** model being the best, while **Linear Regression** model being the worst.
# 

# <center>
#     <h1><b>OPTIMIZATION</b></h1>
# </center>

# In[ ]:


import cuml

# random forest parameters
parameters = {
    'n_estimators': [100, 200, 300, 400],
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3],
    'max_features': [1, 2, 3, 4],
    'max_leaf_nodes': [None, 10, 20, 30, 40],
    'min_impurity_decrease': [0, 0.1, 0.2, 0.3],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'n_jobs': [None, 1, 2, 3, 4],
    'random_state': [None, 0, 1, 2, 3, 4],
    'verbose': [0, 1, 2, 3,],
    'warm_start': [True, False],
    'ccp_alpha': [0, 0.1, 0.2, 0.3],
    'max_samples': [None, 10, 20, 30, 40]
}

# Create GridSearchCV object
sel_grid_search = GridSearchCV(estimator=cuml.ensemble.RandomForestRegressor(),
                           param_grid=parameters,
                           scoring='r2',  # Use R-squared as the scoring metric
                           cv=10,  # You can adjust the number of folds in cross-validation
                           n_jobs=-1)

# Fit the grid search to the data
sel_grid_search = sel_grid_search.fit(X_sel, Y_sel)

print(f"CV RESULTS:")
print(pd.DataFrame(sel_grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']])


color = "\033[92m" if sel_grid_search.best_score_ > 0.5 else "\033[91m"
print(f"BEST ACCURACY SCORE (1.0 is best): \n\033[1m{color}{sel_grid_search.best_score_}\033[0m\n")

print(f"BEST PARAMETERS:")
print(pd.DataFrame(sel_grid_search.best_params_, index=[0]))

# <h2>New Linear Regression Model with Optimized Parameters</h2>
# Create a linear regression model with the optimized parameters

# In[ ]:


#use the values from sel_grid_search.best_score_
sel_lasso_regression_optimized = cuml.ensemble.RandomForestRegressor(**sel_grid_search.best_params_)
sel_lasso_regression_optimized.fit(X_sel_train, Y_sel_train)


# <h2>Prediction Capabilities</h2>
# Check the prediction capabilities of optimized parameters

# In[ ]:


Y_sel_pred_optimized = sel_lasso_regression_optimized.predict(X_sel_test)

# show the comparison between the actual and predicted values
pd.DataFrame({'Actual': Y_sel_test, 'Predicted': Y_sel_pred_optimized}).head(20).applymap('{:f}'.format)

# <h2>K-Fold Cross Validation</h2>
# Use k-fold cross validation to get the accuracy of the model

# In[ ]:


sel_kfold_optimized = KFold(n_splits=10, random_state=0, shuffle=True)

# <h3>Mean Absolute Error</h3>
# Using mean absolute error of the model as scoring for cross validation

# In[ ]:


# calculate the mean absolute error of the model as scoring for cross validation
sel_MAE = -cross_val_score(sel_lasso_regression_optimized, X_sel, Y_sel, cv=sel_kfold_optimized, scoring='neg_mean_absolute_error')

print(f"Selected Feastures Mean Absolute Error of k-Folds: \033[1m\033[94m{(sel_MAE)}\033[0m")
print(f"Average Selected Feastures Mean Absolute Error of k-Folds: \033[1m\033[94m{(sel_MAE.mean())}\033[0m")
print(f"Standard Deviation of Selected Feastures Mean Absolute Error of k-Folds: \033[1m\033[94m{(sel_MAE.std())}\033[0m")


# <h3>Mean Squared Error</h3>
# Using mean squared error of the model as scoring for cross validation

# In[ ]:


sel_MSE = -cross_val_score(sel_lasso_regression_optimized, X_sel, Y_sel, cv=sel_kfold_optimized, scoring='neg_mean_squared_error')

print( f"Selected Feastures Mean Squared Error of k-Folds: \033[1m\033[94m{(sel_MSE)}\033[0m")
print( f"Average Selected Feastures Mean Squared Error of k-Folds: \033[1m\033[94m{(sel_MSE.mean())}\033[0m")
print( f"Standard Deviation of Selected Feastures Mean Squared Error of k-Folds: \033[1m\033[94m{(sel_MSE.std())}\033[0m")

# <h3>R-Squared</h3>
# Using r-squared of the model as scoring for cross validation

# In[ ]:


sel_R2 = cross_val_score(sel_lasso_regression_optimized, X_sel, Y_sel, cv=sel_kfold, scoring='r2')

print( f"Selected Feastures R2 of k-Folds: \033[1m\033[94m{(sel_R2)}\033[0m")
print( f"Average Selected Feastures R2 of k-Folds: \033[1m\033[94m{(sel_R2.mean())}\033[0m")
print( f"Standard Deviation of Selected Feastures R2 of k-Folds: \033[1m\033[94m{(sel_R2.std())}\033[0m")

# <h3>Explained Variance Score</h3>
# Using explained variance score of the model as scoring for cross validation

# In[ ]:


sel_EV = cross_val_score(sel_lasso_regression_optimized, X_sel, Y_sel, cv=sel_kfold, scoring='explained_variance')

print( f"Selected Feastures Explained Variance of k-Folds: \033[1m\033[94m{(sel_EV)}\033[0m")
print( f"Average Selected Feastures Explained Variance of k-Folds: \033[1m\033[94m{(sel_EV.mean())}\033[0m")
print( f"Standard Deviation of Selected Feastures Explained Variance of k-Folds: \033[1m\033[94m{(sel_EV.std())}\033[0m")

# <h3>Root Mean Squared Error</h3>
# Using root mean squared error of the model as scoring for cross validation

# In[ ]:


sel_RMSE = np.sqrt(sel_MSE)

print( f"Selected Feastures Root Mean Squared Error: \033[1m\033[94m{(sel_RMSE)}\033[0m")
print( f"Average Selected Feastures Root Mean Squared Error: \033[1m\033[94m{(sel_RMSE.mean())}\033[0m")
print( f"Standard Deviation of Selected Feastures Root Mean Squared Error: \033[1m\033[94m{(sel_RMSE.std())}\033[0m")

# <h2>Hold-Out Validation</h2>
# Use hold-out validation to evaluate the model

# <h3>Mean Absolute Error</h3>
# Using mean absolute error of the model as scoring for hold-out validation <i>(the best value is 0.0)</i>

# In[ ]:


sel_MAE_holdout = mean_absolute_error(Y_sel_test, Y_sel_pred_optimized)
color = "\033[92m" if sel_MAE_holdout < 0.5 else "\033[91m"

print( f"Selected Feastures Mean Absolute Error of Holdout: \033[1m{color}{(sel_MAE_holdout)}\033[0m")

# <h3>Mean Squared Error</h3>
# Using mean squared error of the model as scoring for hold-out validation <i>(the best value is 0.0)</i>

# In[ ]:


sel_MSE_holdout = mean_squared_error(Y_sel_test, Y_sel_pred_optimized)
color = "\033[92m" if sel_MSE_holdout < 0.5 else "\033[91m"

print( f"Selected Feastures Mean Squared Error of Holdout: \033[1m{color}{(sel_MSE_holdout)}\033[0m")

# <h3>R-Squared Error</h3>
# Using r-squared error of the model as scoring for hold-out validation <i>(the best value is 1.0)</i>

# In[ ]:


sel_R2_holdout = r2_score(Y_sel_test, Y_sel_pred_optimized)
color = "\033[92m" if sel_R2_holdout > 0.5 else "\033[91m"

print( f"Selected Feastures R2 of Holdout: \033[1m{color}{(sel_R2_holdout)}\033[0m")

# <h3>Explained Variance Score</h3>
# Using explained variance score of the model as scoring for hold-out validation  <i>(the best value is 1.0)</i>

# In[ ]:


sel_EV_holdout = explained_variance_score(Y_sel_test, Y_sel_pred_optimized)
color = "\033[92m" if sel_EV_holdout > 0.5 else "\033[91m"

print( f"Selected Feastures Explained Variance of Holdout: \033[1m{color}{(sel_EV_holdout)}\033[0m")

# <h3>Root Mean Squared Error</h3>
# Using root mean squared error of the model as scoring for hold-out validation  <i>(the best value is 0.0)</i>

# In[ ]:


sel_RMSE_holdout = np.sqrt(sel_MSE_holdout)
color = "\033[92m" if sel_RMSE_holdout < 0.5 else "\033[91m"

print( f"Selected Feastures Root Mean Squared Error of Holdout: \033[1m{color}{(sel_RMSE_holdout)}\033[0m")

# # CONCLUSION
# Optimization succeeded, the model is now better than before but still not good enough 
# to be used in production. We will try to use a different model to see if it can improve
# the results.
# 
