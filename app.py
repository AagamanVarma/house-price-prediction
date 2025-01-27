import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import streamlit as st

# Title
st.title("House Price Prediction App")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload your housing dataset (CSV)", type="csv")

if uploaded_file:
    # Load the dataset
    house_price = pd.read_csv(uploaded_file, sep=',', header=None)
    st.write("Dataset loaded successfully!")
    st.write(house_price.head())  # Show the first few rows for debugging
    st.write(f"Number of columns: {house_price.shape[1]}")  # Verify column count

    house_price.columns = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
        'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
        'LSTAT', 'MEDV'
    ]
    
    # Display the data
    st.subheader("Dataset Overview")
    st.write(house_price.head())
    
    # Basic information
    st.subheader("Dataset Information")
    st.write("Shape of the dataset:", house_price.shape)
    st.write("Missing values:", house_price.isnull().sum())
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    correlation = house_price.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues', ax=ax)
    st.pyplot(fig)
    
    # Splitting the data
    X = house_price.iloc[:, :-1].values
    y = house_price.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    
    # Model training
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    # Predictions
    training_data_prediction = model.predict(X_train)
    test_data_prediction = model.predict(X_test)
    
    # Display accuracy
    score_1 = metrics.r2_score(y_train, training_data_prediction)
    score_2 = metrics.mean_absolute_error(y_train, training_data_prediction)
    st.subheader("Training Data Accuracy")
    st.write(f"R-squared Value: {score_1}")
    st.write(f"Mean Absolute Error: {score_2}")
    
    # Visualization
    st.subheader("Training Data: Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.scatter(y_train, training_data_prediction)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted Prices")
    st.pyplot(fig)
    
    # Test accuracy
    score_3 = metrics.r2_score(y_test, test_data_prediction)
    score_4 = metrics.mean_absolute_error(y_test, test_data_prediction)
    st.subheader("Test Data Accuracy")
    st.write(f"R-squared Value: {score_3}")
    st.write(f"Mean Absolute Error: {score_4}")
    
    # Visualization
    st.subheader("Test Data: Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_data_prediction)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted Prices")
    st.pyplot(fig)
