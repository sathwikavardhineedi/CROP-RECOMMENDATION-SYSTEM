#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
file_path = 'Crop_recommendation.xlsx'
df = pd.read_excel(file_path)
print(df.head())
print(df.shape)


# In[22]:


df.isnull().sum()


# In[23]:


def visualize_outliers_boxplot(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))
    
    for i, column in enumerate(numeric_columns):
        plt.subplot(len(numeric_columns) // 3 + 1, 3, i + 1)
        sns.boxplot(x=df[column], color='lightblue')
        plt.title(f'Boxplot of {column}')
    
    plt.tight_layout()
    plt.show()

visualize_outliers_boxplot(df)


# In[24]:


for column in df.select_dtypes(include=['float64', 'int64']).columns:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

print(f"Dataset size after outlier removal: {df.shape}")

df.head()


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns

def visualize_outliers_boxplot(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))
    
    for i, column in enumerate(numeric_columns):
        plt.subplot(len(numeric_columns) // 3 + 1, 3, i + 1)
        sns.boxplot(x=df[column], color='lightblue')
        plt.title(f'Boxplot of {column}')
    
    plt.tight_layout()
    plt.show()
visualize_outliers_boxplot(df)


# In[26]:


df.shape


# In[29]:


logistic_model = LogisticRegression(max_iter=500, random_state=42)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
scaler = StandardScaler()
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logistic_model = LogisticRegression(max_iter=500, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(solver='liblinear', random_state=42)

logistic_model.fit(X_train_scaled, y_train)

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_report = classification_report(y_test, y_pred_logistic)
logistic_confusion_matrix = confusion_matrix(y_test, y_pred_logistic)

print("Logistic Regression Model Performance")
print(f"Accuracy: {logistic_accuracy:.4f}")
print("Classification Report:\n", logistic_report)

plt.figure(figsize=(10, 8))
sns.heatmap(logistic_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_report = classification_report(y_test, y_pred_knn)
knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)

print("KNN Model Performance")
print(f"Accuracy: {knn_accuracy:.4f}")
print("Classification Report:\n", knn_report)

plt.figure(figsize=(10, 8))
sns.heatmap(knn_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix for KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[11]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm)
svm_confusion_matrix = confusion_matrix(y_test, y_pred_svm)

print("SVM Model Performance")
print(f"Accuracy: {svm_accuracy:.4f}")
print("Classification Report:\n", svm_report)

plt.figure(figsize=(10, 8))
sns.heatmap(svm_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix for SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[12]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    upper_limit = df[col].mean() + 3 * df[col].std()
    lower_limit = df[col].mean() - 3 * df[col].std()
    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = df[numerical_columns]
y = df['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')  

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[16]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier(random_state=42, max_depth=3)  # You can adjust hyperparameters as needed

dt_clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(
    dt_clf,
    feature_names=X.columns,  
    class_names=y.unique().astype(str), 
    filled=True,
    max_depth=3  
)
plt.show()


# In[ ]:




