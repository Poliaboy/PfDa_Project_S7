import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

# Load dataset
@st.cache_data()
def load_data():
    data = pd.read_csv('heart_disease.csv')
    return data


df = load_data()

# Sidebar for navigation
st.sidebar.title('Heart Disease Data Visualizations')

# Main content
st.title('Heart Disease Data Exploration')
st.write('Summary Statistics')
st.dataframe(df.describe())

