import streamlit as st
from function import predict

def app(df, x, y):
  st.title('Diabetes Prediction')
  st.write('Please enter the following details for the prediction')

  age = st.number_input('Age', 0, 80, 25)

  hypertension = st.radio('Hypertension', ['Yes', 'No'])
  if hypertension == 'Yes':
    hypertension = 1
  else:
    hypertension = 0

  heart_disease = st.radio('Heart Disease', ['Yes', 'No'])
  if heart_disease == 'Yes':
    heart_disease = 1
  else:
    heart_disease = 0

  smoking_history = st.selectbox('Smoking History', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
  if smoking_history == 'Unknown':
    smoking_history = 1
  elif smoking_history == 'smokes':
    smoking_history = 2
  elif smoking_history == 'formerly smoked':
    smoking_history = 3
  else:
    smoking_history = 0

  bmi = st.number_input('BMI', 10.0, 95.7, 20.0)

  HbA1c_level = st.number_input('HbA1c Level', 3.5, 9.0, 4.0)

  blood_glucose_level = st.number_input('Blood Glucose Level', 80, 300, 100)

  features = [age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]

  if st.button('Prediction'):
    pred, score = predict(x, y, features)
    if pred == 0:
      st.success('You are not likely to have diabetes')
    else:
      st.error('You are likely to have diabetes')
