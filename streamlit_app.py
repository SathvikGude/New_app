import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Function to load and preprocess data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)

    # Convert yes/no columns to numeric
    binary_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for col in binary_columns:
        data[col] = data[col].apply(lambda x: 1 if x == 'yes' else 0)

    # Feature engineering
    data['social_engagement'] = data[['freetime', 'goout', 'romantic']].sum(axis=1)
    data['activity_score'] = data['studytime'] + data['traveltime']
    data['avg_alcohol_consumption'] = (data['Dalc'] + data['Walc']) / 2
    data['travel_study_ratio'] = np.where(data['studytime'] == 0, 0, data['traveltime'] / data['studytime'])
    data['parental_education_score'] = (data['Medu'] + data['Fedu']) / 2

    # Create a dropout label based on the engineered features
    data['dropout'] = data.apply(
        lambda row: 'yes' if (row['absences'] > 20 or row['avg_alcohol_consumption'] > 2 or
                              row['social_engagement'] < 5 or row['activity_score'] < 3 or
                              row['travel_study_ratio'] > 1.5 or row['parental_education_score'] < 2)
        else 'no', axis=1
    )
    return data

# Function to display data summary
def display_data_summary(data):
    st.write("### Data Summary")
    st.write("Basic descriptive statistics:")
    st.write(data.describe(include='all'))

    st.write("### Missing Values")
    st.write(data.isnull().sum())

# Correlation heatmap
def plot_correlation_heatmap(data):
    st.write("### Correlation Heatmap")
    correlation = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="YlGnBu", fmt=".2f")
    st.pyplot()

# Pair plot for selected features
def plot_pairplot(data, selected_features):
    st.write("### Pairplot of Selected Features")
    sns.pairplot(data[selected_features + ['dropout']], hue='dropout', diag_kind='kde')
    st.pyplot()

# Model prediction and evaluation
def run_model(data):
    st.write("### Predictive Model - Logistic Regression")

    # Prepare data for modeling with all engineered features
    features = ['absences', 'social_engagement', 'activity_score', 'avg_alcohol_consumption',
                'travel_study_ratio', 'parental_education_score']
    X = data[features]
    y = data['dropout'].apply(lambda x: 1 if x == 'yes' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred, target_names=["No Dropout", "Dropout"]))

# Main Streamlit app function
def main():
    st.title("Enhanced Student Dropout Prediction")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Display data summary
        display_data_summary(data)

        # Show correlations
        plot_correlation_heatmap(data)

        # Feature selection for pairplot
        all_features = data.columns.tolist()
        selected_features = st.multiselect("Select features for pair plot visualization", all_features, default=['G1', 'G2', 'G3', 'absences'])
        if selected_features:
            plot_pairplot(data, selected_features)

        # Run and display model results
        run_model(data)

if __name__ == "__main__":
    main()
