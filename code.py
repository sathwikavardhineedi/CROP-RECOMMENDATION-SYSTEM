# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import streamlit as st

# Load dataset
file_path = 'Crop_recommendation.xlsx'  # Update with correct path
df = pd.read_excel(file_path)
print(df.head())
print(df.shape)

# Check for missing values
print(df.isnull().sum())

# Function to visualize outliers using boxplots
def visualize_outliers_boxplot(df):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))

    for i, column in enumerate(numeric_columns):
        plt.subplot(len(numeric_columns) // 3 + 1, 3, i + 1)
        sns.boxplot(x=df[column], color='lightblue')
        plt.title(f'Boxplot of {column}')

    plt.tight_layout()
    plt.show()

# Visualize outliers using box plots
visualize_outliers_boxplot(df)

# Remove outliers based on IQR
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

print(f"Dataset size after outlier removal: {df.shape}")

# Re-visualize the outliers after removal
visualize_outliers_boxplot(df)

# Feature Engineering: Normalize numeric columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    upper_limit = df[col].mean() + 3 * df[col].std()
    lower_limit = df[col].mean() - 3 * df[col].std()
    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

# Encoding the target variable
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Split data into features (X) and target (y)
X = df.drop(['label', 'label_encoded'], axis=1)
y = df['label_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the RandomForest model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy * 100:.2f}%')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and label encoder using joblib
joblib.dump(clf, 'crop_rf_model.pkl')
joblib.dump(le, 'crop_label_encoder.pkl')

# Decision Tree Visualization
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt_clf = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_clf.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    dt_clf,
    feature_names=X.columns,
    class_names=y.unique().astype(str),
    filled=True,
    max_depth=3
)
plt.show()


# Streamlit app code
# Title
st.title("🌱 Crop Recommendation System")

# Move the header to be under the main title
st.header("Enter Soil & Weather Conditions")

# Sidebar inputs
nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=100, value=50, key="nitrogen")
phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=100, value=50, key="phosphorus")
potassium = st.number_input("Potassium (K)", min_value=0, max_value=100, value=50, key="potassium")
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, key="temperature")
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, key="humidity")
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5, key="ph")
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, key="rainfall")

# Load the trained model and label encoder
try:
    model = joblib.load('crop_rf_model.pkl')  # Load the trained model
    label_encoder = joblib.load('crop_label_encoder.pkl')  # Load the label encoder

    # Predict button
    if st.button("Recommend Crop"):
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        recommended_crop = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🌾 Recommended Crop: **{recommended_crop}**")

except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
