#LOGISTIC LINEAR REGRESSION TEMPLATE

# To impport the libraries
import numpy as np #numerical computation
import pandas as pd #data manipulation
import matplotlib.pyplot as plt #visualization

# To import the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

# Preliminary Analysis of the Dataset

# A. To Know How Much of the Data is Missing
missing_data = dataset.isnull().sum().sort_values(ascending = False)
# ascending = False means that the data will be sorted in descending order

# B. To check column names and total records
dataset_count = dataset.count()

# C. To view the information about the dataset
print(dataset.info())

# D. To view the statistical summary of the dataset
dataset_statistic = dataset.iloc[:, 1:3]
statistic = dataset_statistic.describe()

# CREATE THE MATRIX OF INDEPENDENT VARIABLE, X (Age and Estimated Salary)
X = dataset.iloc[:, 1:3].values

# CREATE THE MATRIX OF DEPENDENT VARIABLE, Y (Purchased)
Y = dataset.iloc[:, 3].values

# TO VIEW THE SCATTER PLOT OF THE DATASET
import seaborn as sns

# drop uuid
dataset_scatter = dataset.drop("User ID", axis = 1)
sns.pairplot(dataset_scatter)

# To split the Whole Dataset into Training Dataset and Testing Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = 0)


# For standardzarion of the dataset
from sklearn.preprocessing import StandardScaler


standardscaler = StandardScaler()
X_train_standard = X_train.copy()
X_test_standard = X_test.copy()
X_train_standard = standardscaler.fit_transform(X_train_standard)
X_test_standard = standardscaler.transform(X_test_standard)

# To fit the Logistic Regression Model to the Training Dataset
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(solver = "lbfgs", random_state = 0, penalty="l2")
logistic_regression.fit(X_train_standard, Y_train)

# To predict the Test Dataset
Y_predict_probability = logistic_regression.predict_proba(X_test_standard)
Y_predict = logistic_regression.predict(X_test_standard)


# To view the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict)

sns.heatmap(confusion_matrix, annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()


TP = confusion_matrix[1][1]
TN = confusion_matrix[0][0]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]

# To apply k-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold

k_fold = StratifiedKFold(n_splits = 10, random_state = 0, shuffle = True)

# Try the following Performance Metrics
# A. Accuracy "accuracy"
# B. F1 Score "f1"
# C. Precision "precision"
# D. Recall "recall"
# E. ROC AUC "roc_auc"

from sklearn.model_selection import cross_val_score

# A. To feature scale the X variable using the Standardization feature scaling
X_standard = X.copy()
X_standard = standardscaler.fit_transform(X_standard)

# B. For the accuracy as scoring metric for the cross-validation
accuracies = cross_val_score(estimator = logistic_regression, X = X_standard, y = Y, cv = k_fold, scoring = "accuracy")
accuracies_average = accuracies.mean()
accuracies_standard_deviation = accuracies.std()

print("Accuracies of k-Fold:")
print("Average of the Accuracies of k-Fold: ", accuracies_average)
print("Standard Deviation of the Accuracies of k-Fold: ", accuracies_standard_deviation)
print(" ")

# C. For the f1 as scoring metric for the cross-validation
f1 = cross_val_score(estimator = logistic_regression, X = X_standard, y = Y, cv = k_fold, scoring = "f1")
f1_average = f1.mean()
f1_standard_deviation = f1.std()

print("f1 of k-Fold:")
print("Average of the f1 of k-Fold: ", f1_average)
print("Standard Deviation of the f1 of k-Fold: ", f1_standard_deviation)
print(" ")

# D. For the precision as scoring metric for the cross-validation
precision = cross_val_score(estimator = logistic_regression, X = X_standard, y = Y, cv = k_fold, scoring = "precision")
precision_average = precision.mean()
precision_standard_deviation = precision.std()

print("Precision of k-Fold:")
print("Average of the Precision of k-Fold: ", precision_average)
print("Standard Deviation of the Precision of k-Fold: ", precision_standard_deviation)
print(" ")

# E. For the recall as scoring metric for the cross-validation
recall = cross_val_score(estimator = logistic_regression, X = X_standard, y = Y, cv = k_fold, scoring = "recall")
recall_average = recall.mean()
recall_standard_deviation = recall.std()

print("Recall of k-Fold:")
print("Average of the Recall of k-Fold: ", recall_average)
print("Standard Deviation of the Recall of k-Fold: ", recall_standard_deviation)
print(" ")

# F. For the roc_auc as scoring metric for the cross-validation
roc_auc = cross_val_score(estimator = logistic_regression, X = X_standard, y = Y, cv = k_fold, scoring = "roc_auc")
roc_auc_average = roc_auc.mean()
roc_auc_standard_deviation = roc_auc.std()

print("ROC AUC of k-Fold:")
print("Average of the ROC AUC of k-Fold: ", roc_auc_average)
print("Standard Deviation of the ROC AUC of k-Fold: ", roc_auc_standard_deviation)
print(" ")

# To evaluate the performance of the logistic regression using hold out validation

# A. For the classifiation accruacy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict)
print("Classification Accuracy: %.4f" % classification_accuracy)
print()

# B. Classification Error
classification_error = 1 - classification_accuracy
print("Classification Error: %.4f" % classification_error)
print()

# C. Sensitivity, Recall Score, Propapbility of detection, True Positive Rate
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict, average = "weighted")
print("Sensitivity: %.4f" % sensitivity)


# F. Specificity, True Negative Rate
Specificity = TN / (TN + FP)
print ("Specificity:%.4f" %Specificity)
print("")

# E. For the False Positive Rate 
false_positive_rate = 1 - sensitivity
print("False Positive Rate: %.4f"%false_positive_rate)
print("")

# F. For the False Negative Rate 
false_negative_rate = 1 - Specificity
print("False Negative Rate: %.4f" %false_negative_rate)
print("")

# G. For the Precision or Positive Predictive Value
from sklearn.metrics import precision_score  
precision_score = precision_score(Y_test, Y_predict, average = "weighted")
print("Precision: %.4f" % precision_score)
print("") 

# H. For F1-score 
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict, average = "weighted")
print(" F1-score: %.4f" % f1_score)
print("")

# I. For the Classification Report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict)

# J. For the Precision-Recall Curve 
from sklearn.metrics import precision_recall_curve
precision_value, recall_value, threshold = precision_recall_curve (Y_test, Y_predict)
plt.plot(precision_value, recall_value)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("precision-Recall Curve for the Logistics Regression Model")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.grid(True)
plt.show()

# K. For the ROC Curve with AUC 
# k1. For the ROC (Receiver Operating Characteristics)
from sklearn.metrics import roc_curve
FPR, TPR, thresholds = roc_curve(Y_test, Y_predict)

# k2. For the AUC (Arena Under Curve)
from sklearn.metrics import roc_auc_score
AUC_score = roc_auc_score(Y_test, Y_predict)

# K3. To Plot the ROC with AUC
plt.plot(FPR, TPR, label = "ROC Curve")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

# K4. For the Plot of Baseline for AUC
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), label = "baseline", linestyle = "--")
plt.title("ROC-AUC Curve with AUC = {round(AUC_score, 4)} for the Logistic Regression Model")
plt.xlabel("False-Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity) ")
plt.legend
plt.grid(True)
plt.show()

# To Save the Model

# Using the Joblib to Save the Model

from joblib import dump # to serialize
from joblib import load # to deserialize

## To Save the Model
dump(logistic_regression, 'Logistic_regression.pkl')

## To Load the Model
logreg_from_joblib = load('Logistic_regression.pkl')

## To Make Prediction
prediction_joblib = logreg_from_joblib.predict(X_test_standard)
