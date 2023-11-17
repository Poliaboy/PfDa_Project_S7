import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv('heart_disease.csv')
    return data

df = load_data()

# Sidebar for navigation
st.sidebar.title('Heart Disease Data Visualizations')
options = st.sidebar.radio('Select a Chart Type:',
    ('Summary Statistics', 'Histogram', 'Correlation Heatmap', 'Scatter Plot'))

# Main content
st.title('Heart Disease Data Exploration')

if options == 'Summary Statistics':
    st.write('Summary Statistics')
    st.write(df.describe())

elif options == 'Histogram':
    st.write('Histogram')
    column = st.selectbox('Select column to create histogram', df.columns)
    bins = st.slider('Select number of bins:', min_value=5, max_value=50, value=10)
    fig, ax = plt.subplots()
    df[column].hist(bins=bins, ax=ax)
    st.pyplot(fig)

elif options == 'Correlation Heatmap':
    st.write('Correlation Heatmap')
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif options == 'Scatter Plot':
    st.write('Scatter Plot')
    x_axis = st.selectbox('Choose a variable for the x-axis', df.columns, index=0)
    y_axis = st.selectbox('Choose a variable for the y-axis', df.columns, index=1)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
    st.pyplot(fig)