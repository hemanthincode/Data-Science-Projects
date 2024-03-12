"""
Created on Tue Mar 12 19:29:02 2024

@author: hemanth_bommineni
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the Excel file using pd.read_excel()
df = pd.read_excel('bankruptcy-prevention.xlsx')
df
# Display dataset details
print("Dataset Details:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Check for null values
print("\nNull Values:")
print(df.isnull().sum())

# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
print("\nDuplicate Rows:")
print(duplicate_rows)

# Drop duplicate rows
df = df.drop_duplicates()

# Set style for Seaborn plots
sns.set(style="whitegrid")

# Remove whitespaces from column names
df.columns = df.columns.str.strip()

# Separate features and target variable
X = df.drop('class', axis=1) 
Y = df['class']

#--------------------------------- EDA - Explore Data-----------------------------

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=df, palette='viridis')
plt.title('Distribution of Target Variable')
plt.show()

# Explore numerical features using histogram
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.show()

# Explore relationships between numerical features and the target variable
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='class', y=feature, data=df, palette='muted')
    plt.title(f'{feature} by Target Variable')
    plt.show()


# Correlation matrix for numerical features
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing - Scaling and Standardizing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_features])
scaled_df = pd.DataFrame(scaled_data, columns=numerical_features)

# Display density plots for scaled numerical features
plt.figure(figsize=(12, 8))
for i, feature in enumerate(scaled_df.columns, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(scaled_df[feature], color='skyblue', fill=True)
    plt.title(f'Density Plot of {feature}')
plt.tight_layout()
plt.show()

# Pairplot for numerical features
target_variable = 'class'
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
# Check if the target variable exists in the dataset
if target_variable not in df.columns:
    print(f"Error: Target variable '{target_variable}' not found in the dataset.")
else:
    sns.pairplot(df, hue=target_variable, palette='viridis', vars=numerical_features)
    plt.suptitle('Pairplot of Numerical Features by Target Variable', y=1.02)
    plt.show()

# Identify outliers using box plots
plt.figure(figsize=(15, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[feature], color='skyblue')
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()

#--------------------------------- Model Buildings-----------------------------


# Encode the target variable
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')

DT.fit(X_train,Y_train)
Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Training Accuracy", ac1.round(3))
print("Test Accuracy", ac2.round(3))
#====================================================================
# cross validation
#====================================================================
DT = DecisionTreeClassifier(criterion='gini',max_depth=6)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    DT.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test  = DT.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("Cross validation training results:",k1.mean().round(2))
print("Cross validation test results:",k2.mean().round(2))

#====================================================================
# Bagging Classifier
#====================================================================
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6),
                        n_estimators=100,
                        max_samples=0.6,
                        max_features=0.7)
training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    bag.fit(X_train,Y_train)
    Y_pred_train = bag.predict(X_train)
    Y_pred_test  = bag.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("Cross validation training results:",k1.mean().round(2))
print("Cross validation test results:",k2.mean().round(2))
#====================================================================
# RandomForest Classifier
#====================================================================
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_depth=8,
                        max_samples=0.6,
                        max_features=0.7)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test  = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k1.mean().round(2))
print("Cross validation test results:",k2.mean().round(2))
#====================================================================
          #   knn classifer with its accuracy
#====================================================================
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=9)

KNN.fit(X_train,Y_train)

Y_pred_train = KNN.predict(X_train)
Y_pred_test  = KNN.predict(X_test)

# step6:  metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ac_train = accuracy_score(Y_train,Y_pred_train)
ac_test = accuracy_score(Y_test,Y_pred_test)

print("Training Accuracy:", ac_train.round(2))
print("Test Accuracy:", ac_test.round(2))
#====================================================================
# cross validation
#====================================================================

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test  = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

k1 = pd.DataFrame(training_accuracy)
print("Cross validation training results:",k1.mean().round(2))
k2 = pd.DataFrame(test_accuracy)
print("Cross validation test results:",k2.mean().round(2))
#====================================================================
# GradientBoostingClassifier
#====================================================================
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(learning_rate=0.1,
                                n_estimators=500)
training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    GBC.fit(X_train,Y_train)
    Y_pred_train = GBC.predict(X_train)
    Y_pred_test  = GBC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

k1 = pd.DataFrame(training_accuracy)
print("Cross validation training results:",k1.mean().round(2))
k2 = pd.DataFrame(test_accuracy)
print("Cross validation test results:",k2.mean().round(2))
#====================================================================
# step5: support vector machine
from sklearn.svm import SVC
svclass = SVC(C=1.0,kernel='linear')

svclass.fit(X_train,Y_train)

Y_pred_train = svclass.predict(X_train)
Y_pred_test  = svclass.predict(X_test)

# step6:  metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ac_train = accuracy_score(Y_train,Y_pred_train)
ac_test = accuracy_score(Y_test,Y_pred_test)

print("Training Accuracy:", ac_train.round(2))
print("Test Accuracy:", ac_test.round(2))

#====================================================================
# cross validation
#====================================================================

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    svclass.fit(X_train,Y_train)
    Y_pred_train = svclass.predict(X_train)
    Y_pred_test  = svclass.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

k1 = pd.DataFrame(training_accuracy)
print("Cross validation training results:",k1.mean().round(2))
k2 = pd.DataFrame(test_accuracy)
print("Cross validation test results:",k2.mean().round(2))

                         ## Logistic Regrission                          
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()

logr.fit(X_train,Y_train)

Y_predict_train = logr.predict(X_train)
Y_predict_test = logr.predict(X_test)
                               #  metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ac_train = accuracy_score(Y_train, Y_predict_train)
ac_test = accuracy_score(Y_test, Y_predict_test)

print("TRaining Accuracy:", ac_train.round(2))
print("Test Accuracy:", ac_test.round(2))
                                #  Cross Validation
training_accuracy = []
test_accuracy = []

for i in range(1,100):

    X_train, X_test, Y_train, Y_test =train_test_split(X,Y, test_size= 0.30,random_state=i)
    logr.fit(X_train,Y_train)
    Y_pred_train = logr.predict(X_train)
    Y_pred_test = logr.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
print("Cross validation trainingresults:",k1.mean().round(2))
print("Cross validation test results:",k2.mean().round(2))

# Deploying the Project
import pickle
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(DT, file)
    
    