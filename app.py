import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import pickle
import streamlit as st

# Load Dataset
file_path = 'Copper_Set.csv'  # Replace with the correct path to your dataset
data = pd.read_csv(file_path, low_memory=False)

# Data Preprocessing
# Identify numeric and non-numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns
non_numeric_cols = data.select_dtypes(exclude=['number']).columns

# Impute numeric columns with mean
valid_numeric_cols = [col for col in numeric_cols if data[col].notna().sum() > 0]
imputer = SimpleImputer(strategy='mean')
data[valid_numeric_cols] = imputer.fit_transform(data[valid_numeric_cols])

# Impute non-numeric columns with mode
for col in non_numeric_cols:
    if data[col].notna().sum() > 0:  # Skip columns with all missing values
        data[col].fillna(data[col].mode()[0], inplace=True)

# Drop columns that are entirely missing or incompatible with ML
data = data.dropna(axis=1, how='all')

# Handle Skewness in 'Selling_Price' using log transformation
if 'Selling_Price' in data.columns:
    data['Selling_Price'] = np.log1p(data['Selling_Price'])

# Treat outliers using Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(data.select_dtypes(include=['number']))
data = data[outliers == 1]

# Initialize placeholders for regression and classification tasks
X_reg, y_reg, X_cls, y_cls = None, None, None, None

# Separate Regression and Classification Tasks
if 'Selling_Price' in data.columns:
    X_reg = data.drop(columns=['Selling_Price', 'Status'], errors='ignore')
    y_reg = data['Selling_Price']

if 'Status' in data.columns:
    X_cls = data.drop(columns=['Status', 'Selling_Price'], errors='ignore')
    y_cls = data['Status']

# Encode categorical variables
encoder = LabelEncoder()
if y_cls is not None:
    y_cls = encoder.fit_transform(y_cls)

# Train-Test Split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = None, None, None, None
X_cls_train, X_cls_test, y_cls_train, y_cls_test = None, None, None, None

if X_reg is not None and y_reg is not None:
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

if X_cls is not None and y_cls is not None:
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42
    )

# Scaling
scaler = StandardScaler()
X_reg_train_scaled, X_reg_test_scaled = None, None
if X_reg_train is not None:
    X_reg_train_scaled = scaler.fit_transform(X_reg_train)
    X_reg_test_scaled = scaler.transform(X_reg_test)

# Train Models
regressor, classifier = None, None
if X_reg_train_scaled is not None and y_reg_train is not None:
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_reg_train_scaled, y_reg_train)

if X_cls_train is not None and y_cls_train is not None:
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_cls_train, y_cls_train)

# Save Models and Preprocessing Objects
if regressor is not None:
    pickle.dump(regressor, open('regressor.pkl', 'wb'))
if classifier is not None:
    pickle.dump(classifier, open('classifier.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Streamlit App
st.title("Copper Industry ML Application")
task = st.selectbox("Select Task", ["Regression (Selling Price)", "Classification (Status)"])

if task == "Regression (Selling Price)":
    if X_reg is None or regressor is None:
        st.error("Regression model is not available due to missing 'Selling_Price' data.")
    else:
        st.header("Predict Selling Price")
        input_data = []
        for col in X_reg.columns:
            value = st.number_input(f"Enter {col}", value=0.0)
            input_data.append(value)
        input_data = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        regressor = pickle.load(open('regressor.pkl', 'rb'))
        pred = regressor.predict(input_data_scaled)
        st.write(f"Predicted Selling Price: {np.expm1(pred[0]):.2f} (Original Scale)")

elif task == "Classification (Status)":
    if X_cls is None or classifier is None:
        st.error("Classification model is not available due to missing 'Status' data.")
    else:
        st.header("Predict Status (Won/Lost)")
        input_data = []
        for col in X_cls.columns:
            value = st.number_input(f"Enter {col}", value=0.0)
            input_data.append(value)
        input_data = np.array(input_data).reshape(1, -1)
        classifier = pickle.load(open('classifier.pkl', 'rb'))
        pred = classifier.predict(input_data)
        status = 'WON' if pred[0] == 1 else 'LOST'
        st.write(f"Predicted Status: {status}")
