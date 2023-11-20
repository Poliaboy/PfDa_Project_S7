
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Home import load_data

# Load dataset
df = load_data()

st.title('Heart Disease Data Exploration')
st.write('Visualizations of data')

#set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

st.write('Histogram')
column = st.selectbox('Select column to create histogram', df.columns)
bins = st.slider('Select number of bins:', min_value=5, max_value=50, value=10)
fig = sns.displot(df[column], bins=bins, kde=False)
st.pyplot(fig)

# Calculate the correlation matrix
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# You may want to remove the axes labels for a more minimalist design
plt.xticks([])
plt.yticks([])

# Show the plot
st.pyplot(f)

st.write('Scatter Plot')
x_axis = st.selectbox('Choose a variable for the x-axis', df.columns, index=0)
y_axis = st.selectbox('Choose a variable for the y-axis', df.columns, index=1)
hue_option = st.selectbox('Optional categorical variable (hue):', ['None'] + list(df.columns))
hue = None if hue_option == 'None' else hue_option
fig, ax = plt.subplots()
sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[hue], palette='viridis', ax=ax)
st.pyplot(fig)

st.write('Pair Plot')
pairplot_columns = st.multiselect('Select columns for pair plot', df.columns)
if pairplot_columns:
    fig = sns.pairplot(df[pairplot_columns])
    st.pyplot(fig)

st.write('Box Plot')
column = st.selectbox('Select column for box plot', df.columns)
fig, ax = plt.subplots()
sns.boxplot(x=df[column], ax=ax)
st.pyplot(fig)