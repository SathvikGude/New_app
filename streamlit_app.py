import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from io import BytesIO

st.title("Student Dropout Prediction and Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def load_and_process_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    # Feature Engineering
    data['parental_education_score'] = data['Medu'] + data['Fedu']
    data['travel_study_ratio'] = np.where(data['studytime'] == 0, 0, data['traveltime'] / data['studytime'])
    data['avg_alcohol_consumption'] = (data['Dalc'] + data['Walc']) / 2
    data['activity_score'] = data[['activities', 'paid', 'nursery']].applymap(lambda x: 1 if x == 'yes' else 0).sum(axis=1)
    data['support_system_score'] = data[['schoolsup', 'famsup']].applymap(lambda x: 1 if x == 'yes' else 0).sum(axis=1)
    data['grade_trend'] = data['G3'] - data['G1']
    data['high_absenteeism'] = data['absences'].apply(lambda x: 1 if x > 20 else 0)
    return data

def build_model(data):
    features = ['parental_education_score', 'travel_study_ratio', 'avg_alcohol_consumption', 
                'activity_score', 'support_system_score', 'grade_trend', 'high_absenteeism']
    X = data[features]
    y = data['G3'].apply(lambda x: 1 if x < 10 else 0)  # Example criterion for dropout
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Model Accuracy:", model.score(X_test, y_test))
    st.write(classification_report(y_test, y_pred))
    data['dropout'] = model.predict(X)
    return data

def download_csv(data):
    csv_data = data.to_csv(index=False).encode()
    return csv_data

def plot_visualizations(data):
    st.subheader("Visualizations")
    
    # Pie Chart: Attendance percentages
    attendance = data['absences'].apply(lambda x: 'High' if x > 20 else 'Low')
    st.write("Attendance Levels")
    fig, ax = plt.subplots()
    attendance.value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)
    
    # Parent Education Score
    st.write("Parent Education Score")
    fig, ax = plt.subplots()
    sns.histplot(data['parental_education_score'], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Travel Study Ratio
    st.write("Travel Study Ratio")
    fig, ax = plt.subplots()
    sns.histplot(data['travel_study_ratio'], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Social Engagement
    st.write("Social Engagement")
    fig, ax = plt.subplots()
    sns.lineplot(data=data[['freetime', 'goout']])
    st.pyplot(fig)
    
    # Grade Trend
    st.write("Grade Trend")
    fig, ax = plt.subplots()
    sns.lineplot(data=data['grade_trend'])
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(x='parental_education_score', y='dropout', data=data, ax=ax, estimator=lambda x: sum(x == "yes") / len(x) * 100)
    ax.set_title('Dropout Rate by Parental Education Score')
    ax.set_ylabel('Dropout Rate (%)')
    ax.set_xlabel('Parental Education Score')
    st.pyplot(fig)

if uploaded_file:
    data = load_and_process_data(uploaded_file)
    st.write("Uploaded Dataset:", data.head())
    data_with_dropout = build_model(data)
    
    # Download CSV with Dropout column
    csv_data = download_csv(data_with_dropout)
    st.download_button("Download CSV with Dropout Column", data=csv_data, file_name="student_dropout.csv")
    
    # Visualization
    plot_visualizations(data_with_dropout)
