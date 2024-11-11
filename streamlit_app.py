import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)

    # Ensure non-numeric values are handled (convert 'yes'/'no' to 1/0 in social engagement-related columns)
    data[['freetime', 'goout', 'romantic']] = data[['freetime', 'goout', 'romantic']].apply(pd.to_numeric, errors='coerce', axis=1)

    # Add calculated columns to the dataframe
    data['social_engagement'] = data[['freetime', 'goout', 'romantic']].sum(axis=1)
    data['activity_score'] = data['studytime'] + data['traveltime']
    data['avg_alcohol_consumption'] = (data['Dalc'] + data['Walc']) / 2
    data['travel_study_ratio'] = data['traveltime'] / data['studytime']
    data['parental_education_score'] = (data['Medu'] + data['Fedu']) / 2

    # Create a dropout prediction column (dummy for now, to be calculated later)
    data['dropout'] = data['absences'].apply(lambda x: 'yes' if x > 20 else 'no')

    return data

# Function to plot graphs
def plot_graphs(data):
    st.subheader('Graphs and Visualizations')

    # Plot: Distribution of 'dropout' column
    st.write('### Dropout Distribution')
    dropout_counts = data['dropout'].value_counts()
    st.bar_chart(dropout_counts)

    # Plot: Distribution of 'social_engagement' column
    st.write('### Social Engagement Distribution')
    sns.histplot(data['social_engagement'], kde=True)
    st.pyplot()

    # Plot: 'activity_score' vs 'dropout'
    st.write('### Activity Score vs Dropout')
    sns.boxplot(x='dropout', y='activity_score', data=data)
    st.pyplot()

    # Plot: 'avg_alcohol_consumption' vs 'dropout'
    st.write('### Average Alcohol Consumption vs Dropout')
    sns.boxplot(x='dropout', y='avg_alcohol_consumption', data=data)
    st.pyplot()

    # Plot: 'travel_study_ratio' vs 'dropout'
    st.write('### Travel-Study Ratio vs Dropout')
    sns.boxplot(x='dropout', y='travel_study_ratio', data=data)
    st.pyplot()

    # Plot: 'parental_education_score' vs 'dropout'
    st.write('### Parental Education Score vs Dropout')
    sns.boxplot(x='dropout', y='parental_education_score', data=data)
    st.pyplot()
    

# Streamlit application logic
def main():
    st.title("Student Dropout Prediction")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Display the dataset with new columns
        st.write('### Dataset with Additional Columns')
        st.write(data)

        # Show graphs and visualizations
        plot_graphs(data)

    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()
