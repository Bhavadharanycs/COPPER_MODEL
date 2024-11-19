import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle
import streamlit as st

# Load dataset
df = pd.read_csv("your_dataset.csv")

# 1. Data Understanding
categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()

# Handle rubbish values
if 'Material_Reference' in df.columns:
    df['Material_Reference'] = df['Material_Reference'].replace('00000', np.nan)

# Drop unnecessary columns
if 'INDEX' in df.columns:
    df.drop(columns=['INDEX'], inplace=True)

# 2. Data Preprocessing
# Fill missing values
for col in continuous_vars:
    df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_vars:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Treat outliers using Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(df[continuous_vars])
df = df[outliers == 1]

# Treat skewness
for col in continuous_vars:
    if abs(df[col].skew()) > 0.5:
        df[col] = np.log1p(df[col])

# Encode categorical variables
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_vars = pd.DataFrame(encoder.fit_transform(df[categorical_vars]), columns=encoder.get_feature_names_out())
df = pd.concat([df, encoded_vars], axis=1).drop(columns=categorical_vars)

# 3. Feature Engineering
# Drop highly correlated features
corr_matrix = df.corr()
high_corr = [col for col in corr_matrix.columns if any(corr_matrix[col] > 0.9) and col != corr_matrix.columns[0]]
df.drop(columns=high_corr, inplace=True)

# Split data into features and target
if 'Selling_Price' in df.columns:  # Regression
    X = df.drop(columns=['Selling_Price'])
    y = df['Selling_Price']
else:  # Classification
    X = df.drop(columns=['Status'])
    y = df['Status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
if y.dtype == 'float':  # Regression
    model = RandomForestRegressor(random_state=42)
else:  # Classification
    model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Save model and preprocessing steps
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

# 4. Streamlit GUI
st.title("ML Model GUI")

# Task selection
task = st.selectbox("Select Task", ["Regression", "Classification"])

# Input fields for new data
st.subheader("Enter Input Features:")
input_data = {}
for col in X.columns:
    input_data[col] = st.text_input(f"Enter {col}:")

# Predict button
if st.button("Predict"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply feature engineering and scaling
    input_df = encoder.transform(input_df)
    input_df = scaler.transform(input_df)

    # Predict
    model = pickle.load(open("model.pkl", "rb"))
    prediction = model.predict(input_df)
    st.write(f"Prediction: {prediction}")
