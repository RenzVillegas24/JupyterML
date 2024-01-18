# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TO IMPORT DATASET
dataset = pd.read_csv("./Social_Network_Ads.csv")

# Preliminary Analysis of Dataset

# A. To know how much of the data is Missing 
missing_data = dataset.isnull().sum().sort_values(ascending = False)

# B. To Check the Column Names and Total Records
dataset_count = dataset.count()

# C. To View the Information About the Dataset
print(dataset.info())

# D. To View The Statistical Summary Of the Dataset
dataset_statistics = dataset.iloc[:, 1:3]

statistics = dataset_statistics.describe()

dataset_scatter = dataset.drop('User ID', axis = 1)

# To Create the Matrix of the Independent Variable, X (Admin, Marketing, State)
X = dataset_scatter.iloc[:, 0:2].values

# To Create the Matrix of the Dependent Variable, Y (Profit)
Y = dataset_scatter.iloc[:, 2].values

# To View The Scatter Plot of the Dataset
import seaborn as sns
sns.pairplot(dataset_scatter)

# Splitting Test and Train Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 0 )

# A. for Standardization Feature Scaling
from sklearn.preprocessing import StandardScaler # for the data that is not normally distributed 

standard_scaler = StandardScaler()
X_train_standard = X_train.copy()
X_test_standard = X_test.copy ()
X_train_standard= standard_scaler.fit_transform(X_train_standard)
X_test_standard = standard_scaler.transform(X_test_standard)

# To fit the Training Dataset into a Suppiort Vector Machine Model
from sklearn.svm import SVC

support_vector_machine = SVC(kernel = "rbf")
support_vector_machine.fit(X_train_standard, Y_train)

Y_predict = SVC.predict(support_vector_machine, X_test_standard)

# To view the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict)

import seaborn as sns
sns.heatmap(confusion_matrix, annot = True)
plt.title("Confusion Matrix")
plt.xlabel('Predicted Value')
plt.ylabel('Actual value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To Apply the k]-Fold Cross Validation for the Logistic Regression
from sklearn.model_selection import StratifiedKFold

k_fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

# Try the Following Performance Metrics 
# A. accuracy = 'accuracy'
# B. f1-score =  'f1'
# C. precision = 'precision'
# D. recall = 'recall'
# E. roc-auc = 'roc_auc'

from sklearn.model_selection import cross_val_score

X_standard=X.copy()
X_standard= standard_scaler.fit_transform(X_standard)

# B. For the accuracy as scoring metric for the cross-validation
accuracies = cross_val_score(estimator = support_vector_machine, X = X_standard, y = Y, cv = k_fold, scoring = "accuracy")
accuracies_average = accuracies.mean()
accuracies_standard_deviation = accuracies.std()

print("Accuracies of k-Fold:")
print("Average of the Accuracies of k-Fold: ", accuracies_average)
print("Standard Deviation of the Accuracies of k-Fold: ", accuracies_standard_deviation)
print(" ")

# C. For the f1 as scoring metric for the cross-validation
f1 = cross_val_score(estimator = support_vector_machine, X = X_standard, y = Y, cv = k_fold, scoring = "f1")
f1_average = f1.mean()
f1_standard_deviation = f1.std()

print("f1 of k-Fold:")
print("Average of the f1 of k-Fold: ", f1_average)
print("Standard Deviation of the f1 of k-Fold: ", f1_standard_deviation)
print(" ")

# D. For the precision as scoring metric for the cross-validation
precision = cross_val_score(estimator = support_vector_machine, X = X_standard, y = Y, cv = k_fold, scoring = "precision")
precision_average = precision.mean()
precision_standard_deviation = precision.std()

print("Precision of k-Fold:")
print("Average of the Precision of k-Fold: ", precision_average)
print("Standard Deviation of the Precision of k-Fold: ", precision_standard_deviation)
print(" ")

# E. For the recall as scoring metric for the cross-validation
recall = cross_val_score(estimator = support_vector_machine, X = X_standard, y = Y, cv = k_fold, scoring = "recall")
recall_average = recall.mean()
recall_standard_deviation = recall.std()

print("Recall of k-Fold:")
print("Average of the Recall of k-Fold: ", recall_average)
print("Standard Deviation of the Recall of k-Fold: ", recall_standard_deviation)
print(" ")

# F. For the roc_auc as scoring metric for the cross-validation
roc_auc = cross_val_score(estimator = support_vector_machine, X = X_standard, y = Y, cv = k_fold, scoring = "roc_auc")
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


'''
""rbf""
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.9075000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04190763653560052
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.8748654094329845
Standard Deviation of the f1 of k-Fold:  0.05872272653001793
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.8502516708437762
Standard Deviation of the Precision of k-Fold:  0.07849315211629235
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.9085714285714286
Standard Deviation of the Recall of k-Fold:  0.0789902258656913
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.9548234432234433
Standard Deviation of the ROC AUC of k-Fold:  0.02895727761827669
 
Classification Accuracy: 0.9500

Classification Error: 0.0500

Sensitivity: 0.9500
Specificity:0.9483

False Positive Rate: 0.0500

False Negative Rate: 0.0517

Precision: 0.9527

 F1-score: 0.9506

""linear""
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8324999999999999
Standard Deviation of the Accuracies of k-Fold:  0.05921359641163504
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7308753742232004
Standard Deviation of the f1 of k-Fold:  0.10807851314615288
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.8393948887369941
Standard Deviation of the Precision of k-Fold:  0.08674205159868176
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.6661904761904762
Standard Deviation of the Recall of k-Fold:  0.1712624856828476
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.9244000000000001
Standard Deviation of the ROC AUC of k-Fold:  0.03694683754930505
 
Classification Accuracy: 0.9125

Classification Error: 0.0875

Sensitivity: 0.9125
Specificity:0.9828

False Positive Rate: 0.0875

False Negative Rate: 0.0172

Precision: 0.9148

 F1-score: 0.9087

""poly""
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.85
Standard Deviation of the Accuracies of k-Fold:  0.05361902647381803
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7463052599574339
Standard Deviation of the f1 of k-Fold:  0.1063474741904257
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.9084415584415584
Standard Deviation of the Precision of k-Fold:  0.06623686083368499
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.6428571428571429
Standard Deviation of the Recall of k-Fold:  0.13365607198414628
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.9348564102564103
Standard Deviation of the ROC AUC of k-Fold:  0.023651998359386264
 
Classification Accuracy: 0.9125

Classification Error: 0.0875

Sensitivity: 0.9125
Specificity:1.0000

False Positive Rate: 0.0875

False Negative Rate: 0.0000

Precision: 0.9219

 F1-score: 0.9067


""sigmoid""
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.6825
Standard Deviation of the Accuracies of k-Fold:  0.061288253360656325
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.5350931822678207
Standard Deviation of the f1 of k-Fold:  0.08641156370863118
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.5763725490196079
Standard Deviation of the Precision of k-Fold:  0.11851040512015312
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.5171428571428571
Standard Deviation of the Recall of k-Fold:  0.11697626559271815
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.7865985347985348
Standard Deviation of the ROC AUC of k-Fold:  0.050925768224270404
 
Classification Accuracy: 0.8000

Classification Error: 0.2000

Sensitivity: 0.8000
Specificity:0.8621

False Positive Rate: 0.2000

False Negative Rate: 0.1379

Precision: 0.8000

 F1-score: 0.8000

'''