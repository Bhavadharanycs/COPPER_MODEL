import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy.stats import boxcox
import pickle

# Set up Streamlit app
st.set_page_config(page_title="ML Pipeline with Regression & Classification", layout="wide")
st.title("🚀 Machine Learning Pipeline for Classification & Regression")

# Navigation
menu = st.sidebar.radio("Navigation", ["Upload Data", "Data Preprocessing", "EDA", "Train Model", "Make Predictions"])

# Helper Functions
def treat_outliers(df, method="IQR"):
    """Treat outliers in numeric columns using IQR or Isolation Forest."""
    if method == "IQR":
        for col in df.select_dtypes(include=["float", "int"]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    elif method == "IsolationForest":
        clf = IsolationForest(contamination=0.05, random_state=42)
        for col in df.select_dtypes(include=["float", "int"]).columns:
            df[f"{col}_outliers"] = clf.fit_predict(df[[col]])
            df = df[df[f"{col}_outliers"] == 1].drop(columns=[f"{col}_outliers"])
    return df

def preprocess_data(df, target_col):
    """Preprocess dataset for ML."""
    # Treat rubbish values in Material_Reference
    df['Material_Reference'] = df['Material_Reference'].replace(r'^00000.*', np.nan, regex=True)

    # Impute missing values
    imputer = SimpleImputer(strategy="most_frequent")
    df[df.columns] = imputer.fit_transform(df)

    # Encode categorical variables
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Handle skewness in continuous variables
    for col in df.select_dtypes(include=["float", "int"]).columns:
        if df[col].skew() > 1 or df[col].skew() < -1:
            df[col] = np.log1p(df[col])  # Log transformation

    # Scale data
    scaler = StandardScaler()
    features = df.drop(columns=[target_col])
    target = df[target_col]
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    return features_scaled, target, scaler

# Upload Data Section
if menu == "Upload Data":
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["data"] = df
        st.write("### Dataset Preview", df.head())

# Data Preprocessing Section
elif menu == "Data Preprocessing":
    st.header("🔄 Data Preprocessing")
    if "data" not in st.session_state:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["data"]
        target_col = st.selectbox("Select Target Column", df.columns)
        method = st.selectbox("Outlier Treatment Method", ["IQR", "IsolationForest"])

        if st.button("Preprocess Data"):
            df_cleaned = treat_outliers(df.copy(), method)
            X, y, scaler = preprocess_data(df_cleaned, target_col)
            st.session_state["preprocessed_data"] = (X, y, scaler)
            st.success("Data preprocessing completed.")
            st.write("### Preprocessed Features", X.head())
            st.write("### Target Variable", y.head())

# EDA Section
elif menu == "EDA":
    st.header("📊 Exploratory Data Analysis")
    if "data" not in st.session_state:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state["data"]
        st.subheader("Skewness")
        for col in df.select_dtypes(include=["float", "int"]).columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[col], kde=True, ax=axes[0]).set_title(f"Original {col}")
            sns.histplot(np.log1p(df[col]), kde=True, ax=axes[1]).set_title(f"Log-Transformed {col}")
            st.pyplot(fig)

# Train Model Section
elif menu == "Train Model":
    st.header("🧠 Train a Machine Learning Model")
    if "preprocessed_data" not in st.session_state:
        st.warning("Please preprocess your data first.")
    else:
        X, y, scaler = st.session_state["preprocessed_data"]
        task = st.radio("Select Task", ["Regression", "Classification"])

        if task == "Regression":
            model = ExtraTreesRegressor(random_state=42)
        else:
            model = ExtraTreesClassifier(random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Save the model and scaler
        pickle.dump(model, open(f"{task.lower()}_model.pkl", "wb"))
        pickle.dump(scaler, open("scaler.pkl", "wb"))
        st.success(f"{task} model trained and saved successfully.")

# Make Predictions Section
elif menu == "Make Predictions":
    st.header("🔮 Make Predictions")
    model_file = st.file_uploader("Upload Model File (.pkl)", type="pkl")
    scaler_file = st.file_uploader("Upload Scaler File (.pkl)", type="pkl")

    if model_file and scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)

        st.write("### Input Values")
        inputs = {}
        for col in st.session_state["data"].columns:
            if col != st.session_state["data"].columns[-1]:
                inputs[col] = st.number_input(f"Enter value for {col}")

        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Prediction: {prediction[0]}")
