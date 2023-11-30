import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Function to load the pre-trained model and preprocessor
@st.cache_data()
def load_model():
    ai_model = joblib.load('SVC.joblib')
    return ai_model


@st.cache_data()
def load_preprocessor():
    ai_preprocessor = joblib.load('heart_disease_preprocessor.joblib')
    return ai_preprocessor


def preprocess_data(data, preprocessor):
    # Apply the preprocessor to the input data
    preprocessed_data = preprocessor.transform(data)

    # Return a DataFrame for easy visualization and manipulation
    return pd.DataFrame(preprocessed_data)


model = load_model()


def main():
    st.title("Heart Disease Prediction App")
    st.write("This application uses a machine learning model to predict the likelihood of heart disease "
             "based on various health metrics."
             "The model employed is a Neural Network classifier, "
             "chosen for its effectiveness in binary classification tasks. "
             "On average, this model achieves an accuracy of around 85%, "
             "making it a reliable tool for preliminary assessment.")

    default_values = {
        'chest_pain_type': 0,
        'resting_blood_pressure': 120,
        'serum_cholesterol_mg_per_dl': 200,
        'fasting_blood_sugar_gt_120_mg_per_dl': 0,
        'resting_ecg_results': 0,
        'maximum_heart_rate_achieved': 150,
        'exercise_induced_angina': 0,
        'st_depression_induced_by_exercise_relative_to_rest': 0.0,
        'slope_of_peak_exercise_st_segment': 2,
        'number_of_major_vessels_colored_by_flourosopy': 0,
        'thalassemia': 3
    }
    categorical_options = {
        'chest_pain_type': [0, 1, 2, 3, 4],
        'fasting_blood_sugar_gt_120_mg_per_dl': [0, 1],
        'resting_ecg_results': [0, 1, 2],
        'exercise_induced_angina': [0, 1],
        'slope_of_peak_exercise_st_segment': [1, 2, 3],
        'thalassemia': [3, 6, 7]
    }

    feature_descriptions = {
        'chest_pain_type': "Chest pain type (1: typical angina, 2: atypical angina, "
                           "3: non-anginal pain, 4: asymptomatic).",
        'resting_blood_pressure': "Resting blood pressure in mm Hg on admission to the hospital.",
        'serum_cholesterol_mg_per_dl': "Serum cholesterol in mg/dl.",
        'fasting_blood_sugar_gt_120_mg_per_dl': "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).",
        'resting_ecg_results': "Resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, "
                               "2: showing probable or definite left ventricular hypertrophy).",
        'maximum_heart_rate_achieved': "Maximum heart rate achieved.",
        'exercise_induced_angina': "Exercise induced angina (1 = yes; 0 = no).",
        'st_depression_induced_by_exercise_relative_to_rest': "ST depression induced by exercise relative to rest.",
        'slope_of_peak_exercise_st_segment': "Slope of the peak exercise ST segment (1: upsloping, "
                                             "2: flat, 3: downsloping).",
        'number_of_major_vessels_colored_by_flourosopy': "Number of major vessels (0-3) colored by flourosopy.",
        'thalassemia': "Thalassemia (3: normal, 6: fixed defect, 7: reversible defect).",
    }
    # User input for age and gender
    age = st.number_input("Enter your age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Select your gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    # Option for users to modify the default values
    if st.checkbox("Modify other values"):
        for key, value in default_values.items():
            if key in ['resting_blood_pressure', 'serum_cholesterol_mg_per_dl',
                       'maximum_heart_rate_achieved',
                       'number_of_major_vessels_colored_by_flourosopy']:
                # Numeric input (int)
                st.divider()
                st.write(f"{key.replace('_', ' ').title()}: {feature_descriptions[key]}")
                default_values[key] = st.number_input(f"{key.replace('_', ' ').title()}", value=value)

            elif key == "st_depression_induced_by_exercise_relative_to_rest":
                # Numeric input  (float)
                st.divider()
                st.write(f"{key.replace('_', ' ').title()}: {feature_descriptions[key]}")
                default_values[key] = st.number_input(f"{key.replace('_', ' ').title()}", value=value, step=0.1)
            else:
                # Categorical input
                st.divider()
                st.write(f"{key.replace('_', ' ').title()}: {feature_descriptions[key]}")
                default_values[key] = st.selectbox(f"{key.replace('_', ' ').title()}",
                                                   options=categorical_options[key],
                                                   index=categorical_options[key].index(value))

    # Preparing the data for prediction
    features = [age, gender] + list(default_values.values())
    columns = ['age', 'gender'] + list(default_values.keys())
    input_df = pd.DataFrame([features], columns=columns)
    features = preprocess_data(input_df, load_preprocessor())

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(features)
        result = "ill" if prediction[0] == 1 else "not ill"
        if result == "ill":
            st.error(f"The model predicts the patient is {result}.")
        else:
            st.success(f"The model predicts the patient is {result}.")


if __name__ == "__main__":
    main()
