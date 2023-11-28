
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Homepage import load_data

# Load dataset
df = load_data()
# TODO: Add a selectbox to choose which visualisation to add and delete
# TODO: Add the option to download the visualisation as a png file

st.title('Heart Disease Data Exploration')
st.header('Visualizations of data')
st.subheader('This page contains various tools to visualize the data, such a histograms, heatmaps, scatter plots... ')
st.subheader('You can choose which visualisation you want showed, parameter it and export it.')


#set the aesthetic style of the plots
sns.set_theme(style="whitegrid")

st.write('Histogram')
column = st.selectbox('Select column to create histogram', df.columns)
bins = st.slider('Select the grouping of the data:', min_value=5, max_value=50, value=10)
fig = sns.displot(df[column], bins=bins, kde=False)
st.pyplot(fig)

# Calculate the correlation matrix
st.write('Heatmap')
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
if hue:
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[hue], palette='viridis', ax=ax)
else:
    sns.scatterplot(x=df[x_axis], y=df[y_axis], palette='viridis', ax=ax)
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