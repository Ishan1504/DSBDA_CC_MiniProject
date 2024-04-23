import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path

pkl_path = Path(__file__).parent / 'PKL_Files/mod.pkl'

st.title(" Diabetes_Prediction ")

def load_util():
    model = joblib.load(pkl_path)
    return model

g = st.selectbox(" Enter your gender " , ("Male" , "Female"))

if(g == "Female"):
    gen = 0
else:
    gen = 1

age = st.number_input("Enter your age")
h = st.selectbox("Weather you are suffering from hypertension" , ("No" , "Yes"))
if(h == "No"):
    hyp = 0
else:
    hyp = 1

hd = st.selectbox("Weather you are suffering from any heart_disease" , ("Yes" , "No"))
if(hd == "Yes"):
    hdis = 1
else:
    hdis = 0
bmi = st.number_input("Enter your BMI")
hb = st.number_input("Enter your HbA1c_level")
glu = st.number_input("Enter your blood_glucose_level")

button = st.button("Predict")

if(button):
    model = load_util()
    ans = model.predict([[gen,age,hyp,hdis,bmi,hb,glu]])
    if(ans == 1):
        st.write(" # You are suffering from Diabetes")
    else:
        st.write(" # You are not suffering from Diabetes")
else:
    st.write("Check the Prediction")