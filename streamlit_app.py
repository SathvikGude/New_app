import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the data
def load_data(file):
    data = pd.read_csv(file)
    
    # Adding derived columns for dropout prediction
    data['parental_education_score'] = data['Medu'] + data['Fedu']
    data['travel_study_ratio'] = np.where(data['studytime'] == 0, 0, data['traveltime'] / data['studytime'])
    data['avg_alcohol_consumption'] = (data['Dalc'] + data['Walc']) / 2
    data['activity_score'] = data[['activities', 'paid', 'nursery']].apply(lambda x: sum(x == "yes"), axis=1)
    data['support_system_score'] = data[['schoolsup', 'famsup']].apply(lambda x: sum(x == "yes"), axis=1)
    data['grade_trend'] = data['G3'] - data['G1']
    data['high_absenteeism'] = data['absences'] > 20  # example threshold
    
    # Dropout prediction (1 for dropout, 0 for no dropout) based on some basic rules
    conditions = [
        (data['parental_education_score'] < 3) | 
        (data['travel_study_ratio'] > 2) |
        (data['avg_alcohol_consumption'] > 3) |
        (data['activity_score'] < 1) |
        (data['support_system_score'] < 1) |
        (data['grade_trend'] < -5) |
        (data['high_absenteeism'] == True)
    ]
    data['dropout'] = np.where(np.any(conditions, axis=0), 'yes', 'no')
    
    return data

# Function to train dropout prediction model
def train_model(data):
    X = data[['parental_education_score', 'travel_study_ratio', 'avg_alcohol_consumption', 
              'activity_score', 'support_system_score', 'grade_trend', 'absences']]
    y = data['dropout'].apply(lambda x: 1 if x == 'yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("### Model Performance")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write(classification_report(y_test, y_pred))
    return model

# Visualization function
# Visualization function
def visualize_data(data):
    st.write("## Data Visualizations")
    
    # Pie chart for attendance (Dropout vs. No Dropout)
    st.write("### Dropout Rate")
    dropout_counts = data['dropout'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(dropout_counts, labels=dropout_counts.index, autopct='%1.1f%%', startangle=140, colors=["#ff9999","#66b3ff"])
    st.pyplot(plt)
    st.write("This pie chart shows the proportion of students who are predicted to drop out vs. those who are not. "
             "A higher dropout rate may indicate underlying issues within the student body that need addressing.")
    
    # Parental education score distribution
    st.write("### Parental Education Score Distribution")
    plt.figure(figsize=(8, 5))
    sns.histplot(data['parental_education_score'], kde=True, color='skyblue')
    st.pyplot(plt)
    st.write("This histogram shows the distribution of parental education scores among students. "
             "Higher scores indicate that both parents have a higher level of education, which is often linked "
             "to better academic support at home.")
    
    # Travel-study ratio distribution
    st.write("### Travel to Study Time Ratio")
    plt.figure(figsize=(8, 5))
    sns.histplot(data['travel_study_ratio'], kde=True, color='salmon')
    st.pyplot(plt)
    st.write("The travel-to-study time ratio highlights the balance between commute time and study time. "
             "A higher ratio could imply that students spend more time commuting than studying, which might "
             "impact their academic focus and performance.")
    
    # Social engagement score distribution
    st.write("### Social Engagement Distribution")
    
    # Convert categorical engagement indicators to numeric (0 or 1)
    data['freetime'] = pd.to_numeric(data['freetime'], errors='coerce').fillna(0)
    data['goout'] = pd.to_numeric(data['goout'], errors='coerce').fillna(0)
    data['romantic'] = pd.to_numeric(data['romantic'], errors='coerce').fillna(0)
    
    # Calculate social engagement
    data['social_engagement'] = data[['freetime', 'goout', 'romantic']].sum(axis=1)
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data['social_engagement'], kde=True, color='lightgreen')
    st.pyplot(plt)
    st.write("This chart shows the distribution of social engagement scores, which considers free time, "
             "outings, and romantic relationships. High engagement could either positively or negatively "
             "impact academic performance, depending on the individual.")
    
    # Grade trend visualization
    st.write("### Grade Trend from G1 to G3")
    plt.figure(figsize=(8, 5))
    plt.plot(data.index, data['G1'], label='G1', color='blue')
    plt.plot(data.index, data['G2'], label='G2', color='orange')
    plt.plot(data.index, data['G3'], label='G3', color='green')
    plt.xlabel("Students")
    plt.ylabel("Grades")
    plt.legend()
    st.pyplot(plt)
    st.write("This line plot shows the progression of grades (G1, G2, and G3) for each student. "
             "A downward trend from G1 to G3 might indicate students who are struggling academically, "
             "which could be linked to dropout risk.")

# Streamlit app layout
st.title("Student Dropout Prediction and Analysis Tool")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())
    
    model = train_model(data)
    
    st.write("### Visualization of Student Data")
    visualize_data(data)
    
    st.write("### Download Processed Data")
    processed_data = data.copy()
    processed_data.to_csv("/mnt/data/processed_student_data.csv", index=False)
    st.download_button(
        label="Download CSV",
        data=processed_data.to_csv(index=False),
        file_name="processed_student_data.csv",
        mime="text/csv"
    )
