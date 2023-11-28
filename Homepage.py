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
# Main content
# TODO: add presentation text of the dataset and features used
st.title('Heart Disease Data Exploration')
st.write('Summary Statistics')
st.dataframe(df.describe())


