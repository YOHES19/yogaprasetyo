import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# Load the trained GradientBoosting model
with open('best_gradient_boosting_model.pkl', 'rb') as model_file:
    gb_model = pickle.load(model_file)

# Example of species map (adjust this according to your model's classes)
species_map = ['Grapefruit','Orange' ]  # Ensure the order matches the model's classes

# Set the page title
st.title('Fruit Prediction')

# Input form for user data
st.sidebar.header('Input Features')

# User inputs for features
diameter = st.sidebar.number_input('Diameter (cm)', min_value=0.0, value=5.0)
weight = st.sidebar.number_input('Weight (g)', min_value=0.0, value=100.0)
red = st.sidebar.number_input('Red', min_value=0, max_value=255, value=150)
green = st.sidebar.number_input('Green', min_value=0, max_value=255, value=100)
blue = st.sidebar.number_input('Blue', min_value=0, max_value=255, value=50)

# Feature vector with column names
input_features = pd.DataFrame([[diameter, weight, red, green, blue]], columns=['diameter', 'weight', 'red', 'green', 'blue'])

# Prediction
if st.sidebar.button('Predict'):
    prediction = gb_model.predict(input_features)
    predicted_species = species_map[prediction[0]]  # Map the prediction to the species name
    st.write(f'The predicted fruit species is: {predicted_species}')

# Display the input features
st.subheader('Input Data')
data = {
    'Diameter (cm)': diameter,
    'Weight (g)': weight,
    'Red': red,
    'Green': green,
    'Blue': blue
}
input_df = pd.DataFrame(data, index=[0])
st.write(input_df)
