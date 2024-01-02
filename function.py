import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

@st.cache_data()
def load_data():
    df = pd.read_csv('diabetes_prediction_dataset.csv')

    le = LabelEncoder()
    df["smoking_history"]=le.fit_transform(df["smoking_history"])

    x = df[['age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
    y = df[['diabetes']]
    return df, x, y

@st.cache_data()
def train_model(x,y):
  knn = KNeighborsClassifier(n_neighbors = 3)
  knn.fit(x, y)
  score = knn.score(x, y)

  return knn, score

def predict(x, y, features):
  knn, score = train_model(x,y)

  pred = knn.predict(np.array(features).reshape(1,-1))

  return pred, score