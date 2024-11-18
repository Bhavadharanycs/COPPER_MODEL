import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pickle

# Load dataset
@st.cache
def load_data():
    data_path = 'Copper_Set 1.csv'  # Update based on deployment
    return pd.read_csv(data_path)

# Initialize the app
st.title("Copper Industry ML Tool")
st.sidebar.title("Options")
task = st.sidebar.selectbox("Select Task", ["Regression (Selling Price)", "Classification (Lead Status)"])

# Load data
df = load_data()
st.write("Dataset Loaded Successfully!")
st.write(df.head())

# Data Cleaning and Preprocessing
st.subheader("Data Preprocessing")

# Check for Material_Reference column
if 'Material_Reference' in df.columns:
    df['Material_Reference'] = df['Material_Reference'].replace('00000', np.nan)
else:
    st.warning("'Material_Reference' column not found. Skipping related preprocessing.")

# Drop 'INDEX' column if it exists
if 'INDEX' in df.columns:
    df.drop(columns=['INDEX'], inplace=True)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df.iloc[:, :] = imputer.fit_transform(df)

# Detect and treat skewness
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col])

# Outlier treatment
def handle_outliers(data, cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower_bound, upper_bound)

handle_outliers(df, numeric_cols)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Feature-target splitting
if task == "Regression (Selling Price)":
    if "Selling_Price" not in df.columns:
        st.error("'Selling_Price' column is missing in the dataset!")
        st.stop()
    target = "Selling_Price"
elif task == "Classification (Lead Status)":
    if "Status" not in df.columns:
        st.error("'Status' column is missing in the dataset!")
        st.stop()
    target = "Status"
    df = df[df[target].isin(['WON', 'LOST'])]  # Filter relevant statuses

X = df.drop(columns=[target])
y = df[target]

# Model Training
st.subheader("Model Training")

if task == "Regression (Selling Price)":
    y = np.log1p(y)  # Transform target variable
    model = RandomForestRegressor(random_state=42)
    eval_metric = "RMSE"
elif task == "Classification (Lead Status)":
    y = encoder.fit_transform(y)  # Encode target
    model = RandomForestClassifier(random_state=42)
    eval_metric = "Accuracy"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train)

# Save models
pickle.dump(model, open(f"{task.replace(' ', '_')}_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Model Evaluation
if task == "Regression (Selling Price)":
    y_pred = model.predict(X_test)
    y_pred = np.expm1(y_pred)  # Reverse log transform
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))
    st.write(f"Model Evaluation - {eval_metric}: {rmse:.2f}")
elif task == "Classification (Lead Status)":
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Evaluation - {eval_metric}: {accuracy:.2f}")
    st.text(classification_report(y_test, y_pred))

# Streamlit UI for Prediction
st.subheader("Make a Prediction")
user_input = {}
for col in X.columns:
    user_input[col] = st.text_input(col, "")

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input], dtype=float)
        input_df = scaler.transform(input_df)  # Scale inputs
        prediction = model.predict(input_df)

        if task == "Regression (Selling Price)":
            prediction = np.expm1(prediction)  # Reverse log transform
        else:
            prediction = encoder.inverse_transform([int(prediction)])

        st.write(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
