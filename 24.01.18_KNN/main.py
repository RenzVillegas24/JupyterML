# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TO IMPORT DATASET
dataset = pd.read_csv("Social_Network_Ads.csv")

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

for i in range(1, 50, 2):
    print("K:",i)
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


    # To Fit the Training Dataset into a K-Nearest Neighbors Model
    from sklearn.neighbors import KNeighborsClassifier
    k_nearest_neighbor = KNeighborsClassifier(n_neighbors=1)
    k_nearest_neighbor.fit(X_train_standard, Y_train)

    # To predict the Output of the testing Dataset
    Y_predict = k_nearest_neighbor.predict(X_test_standard)

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
    accuracies = cross_val_score(estimator = k_nearest_neighbor, X = X_standard, y = Y, cv = k_fold, scoring = "accuracy")
    accuracies_average = accuracies.mean()
    accuracies_standard_deviation = accuracies.std()

    print("Accuracies of k-Fold:")
    print("Average of the Accuracies of k-Fold: ", accuracies_average)
    print("Standard Deviation of the Accuracies of k-Fold: ", accuracies_standard_deviation)
    print(" ")

    # C. For the f1 as scoring metric for the cross-validation
    f1 = cross_val_score(estimator = k_nearest_neighbor, X = X_standard, y = Y, cv = k_fold, scoring = "f1")
    f1_average = f1.mean()
    f1_standard_deviation = f1.std()

    print("f1 of k-Fold:")
    print("Average of the f1 of k-Fold: ", f1_average)
    print("Standard Deviation of the f1 of k-Fold: ", f1_standard_deviation)
    print(" ")

    # D. For the precision as scoring metric for the cross-validation
    precision = cross_val_score(estimator = k_nearest_neighbor, X = X_standard, y = Y, cv = k_fold, scoring = "precision")
    precision_average = precision.mean()
    precision_standard_deviation = precision.std()

    print("Precision of k-Fold:")
    print("Average of the Precision of k-Fold: ", precision_average)
    print("Standard Deviation of the Precision of k-Fold: ", precision_standard_deviation)
    print(" ")

    # E. For the recall as scoring metric for the cross-validation
    recall = cross_val_score(estimator = k_nearest_neighbor, X = X_standard, y = Y, cv = k_fold, scoring = "recall")
    recall_average = recall.mean()
    recall_standard_deviation = recall.std()

    print("Recall of k-Fold:")
    print("Average of the Recall of k-Fold: ", recall_average)
    print("Standard Deviation of the Recall of k-Fold: ", recall_standard_deviation)
    print(" ")

    # F. For the roc_auc as scoring metric for the cross-validation
    roc_auc = cross_val_score(estimator = k_nearest_neighbor, X = X_standard, y = Y, cv = k_fold, scoring = "roc_auc")
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
dump(k_nearest_neighbor, 'k_nearest_neighbor.pkl')

## To Load the Model
logreg_from_joblib = load('k_nearest_neighbor.pkl')

## To Make Prediction
prediction_joblib = logreg_from_joblib.predict(X_test_standard)
'''

'''
K: 1
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 3
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 5
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 7
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 9
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 11
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 13
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 15
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 17
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 19
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 21
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 23
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 25
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 27
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 29
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 31
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 33
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 35
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 37
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 39
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 41
c:\Users\RenzCute\AppData\Local\Programs\Python\Python310\lib\site-packages\seaborn\axisgrid.py:1274: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.
  fig = plt.figure(figsize=figsize)
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 43
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125

Sensitivity: 0.8875
Specificity:0.9138

False Positive Rate: 0.1125

False Negative Rate: 0.0862

Precision: 0.8893

 F1-score: 0.8883

K: 45
Accuracies of k-Fold:
Average of the Accuracies of k-Fold:  0.8450000000000001
Standard Deviation of the Accuracies of k-Fold:  0.04847679857416328
 
f1 of k-Fold:
Average of the f1 of k-Fold:  0.7870178029855449
Standard Deviation of the f1 of k-Fold:  0.058007274294095905
 
Precision of k-Fold:
Average of the Precision of k-Fold:  0.791404600301659
Standard Deviation of the Precision of k-Fold:  0.10368541325863656
 
Recall of k-Fold:
Average of the Recall of k-Fold:  0.790952380952381
Standard Deviation of the Recall of k-Fold:  0.05081869422308765
 
ROC AUC of k-Fold:
Average of the ROC AUC of k-Fold:  0.8334761904761905
Standard Deviation of the ROC AUC of k-Fold:  0.04266113548079732
 
Classification Accuracy: 0.8875

Classification Error: 0.1125
...
Precision: 0.8893

 F1-score: 0.8883
'''

pass