import streamlit as st
from function import load_data, predict
from Tabs import home, prediction, visualize

Tabs = {
    "Home": home,
    "Prediction": prediction,
    "Visualization": visualize
}

st.sidebar.title("Navigation")

selection = st.sidebar.radio("Go to", list(Tabs.keys()))

df, x, y = load_data()

if selection in ["Prediction", "Visualization"]:
  Tabs[selection].app(df, x, y)
else:
  Tabs[selection].app()