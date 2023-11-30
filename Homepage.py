import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")


def get_image_download_link(fig, filename="plot.png", text="Download plot as PNG"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return st.download_button(label=text, data=buf, file_name=filename, mime="image/png")


# Function to load data
@st.cache_data()
def load_data():
    data = pd.read_csv('heart_disease.csv')
    return data


# Load dataset
df = load_data()
numerical_cols = [
    'age',
    'resting_blood_pressure',
    'serum_cholesterol_mg_per_dl',
    'maximum_heart_rate_achieved',
    'st_depression_induced_by_exercise_relative_to_rest',
    'number_of_major_vessels_colored_by_flourosopy'
]

categorical_cols = [
    'gender',
    'chest_pain_type',
    'fasting_blood_sugar_gt_120_mg_per_dl',
    'resting_ecg_results',
    'exercise_induced_angina',
    'slope_of_peak_exercise_st_segment',
    'thalassemia'
]

# Main content
st.title('Heart Disease Data Exploration')
st.write('This dataset contains information about patients and their heart disease status.')
st.write(f"Number of patients: {df.shape[0]}")
st.write(f"Average age: {df['age'].mean():.2f}")  # Replace 'age' with your actual age column name
gender_count = df['gender'].value_counts()  # Replace 'sex' with your actual gender column name
st.write(f"Gender distribution (Male/Female): {gender_count[0]} / {gender_count[1]}")
if st.toggle('Show Dataset .describe()'):
    st.dataframe(df.describe())

# Visualization selection
st.header('Visualizations of Data')
vis_type = st.selectbox("Choose the type of visualization:", ('Histogram', 'Scatter Plot', 'Pair Plot', 'Box Plot',
                                                              'Bar Plot', 'Violin Plot', 'Heatmap'))

if vis_type == 'Histogram':
    st.subheader('Histogram')
    st.write("A histogram displays the distribution of a numerical variable. "
             "It partitions the range of the data into bins and shows the number of observations in each bin.")
    column = st.selectbox('Select column to create histogram', df.columns)
    bins = st.slider('Select the grouping of the data:', min_value=5, max_value=50, value=10)
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=column, hue='illness', element='step', bins=bins, kde=False, ax=ax)
    st.pyplot(fig)


elif vis_type == 'Scatter Plot':
    st.subheader('Scatter Plot')
    st.write("A scatter plot displays values of two numerical variables and can show the relationship between them. "
             "Each dot represents an observation.")
    x_axis = st.selectbox('Choose a variable for the x-axis', numerical_cols, index=0)
    y_axis = st.selectbox('Choose a variable for the y-axis', numerical_cols, index=1)
    hue_option = st.selectbox('Optional categorical variable (hue):', ['None'] + list(df.columns))
    hue = 'illness' if hue_option == 'None' else hue_option
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df[hue], ax=ax)
    st.pyplot(fig)

elif vis_type == 'Bar Plot':
    st.subheader('Bar Plot')
    st.write("A bar plot represents categorical data with rectangular bars. "
             "Each bar’s height is proportional to the count of the category it represents.")
    column = st.selectbox('Select column for bar plot', categorical_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=df[column].value_counts().index, y=df[column].value_counts().values, ax=ax)
    st.pyplot(fig)

elif vis_type == 'Pair Plot':
    st.subheader('Pair Plot')
    st.write("A pair plot is a matrix of scatter plots showing the relationship "
             "between every pair of variables in the dataset.")
    pairplot_columns = st.multiselect('Select columns for pair plot', df.columns)
    if pairplot_columns:
        fig = sns.pairplot(df[pairplot_columns], hue='illness')
        st.pyplot(fig)

elif vis_type == 'Box Plot':
    st.subheader('Box Plot')
    st.write("A box plot shows the distribution of data, highlighting the median, quartiles, and outliers.")
    column = st.selectbox('Select column for box plot', df.columns)
    fig, ax = plt.subplots()
    sns.boxplot(x='illness', y=df[column], ax=ax, data=df)
    st.pyplot(fig)

elif vis_type == 'Violin Plot':
    st.subheader('Violin Plot')
    st.write("A violin plot combines aspects of box plots and density plots, "
             "ideal for comparing distributions across categories.")
    column = st.selectbox('Select numerical column for violin plot', numerical_cols)
    category = st.selectbox('Select category for violin plot', df.columns)
    fig, ax = plt.subplots()
    sns.violinplot(x=df[category], y=df[column], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

elif vis_type == 'Heatmap':
    st.subheader('Heatmap')
    st.write("A heatmap shows relationships between two variables, one plotted on each axis. "
             "Colors indicate correlation coefficients, highlighting how closely related each pair of variables is.")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    st.pyplot(f)

# Allow the user to download the visualization as a PNG file

if st.button('Download Current Visualization'):
    get_image_download_link(plt.gcf())
