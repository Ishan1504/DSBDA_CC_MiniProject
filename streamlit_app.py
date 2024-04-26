import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
pkl_path = Path(__file__).parent / 'PKL_Files/mod.pkl'

# Function to load the trained model

def load_util():
    model = joblib.load(pkl_path)
    return model

# Load the dataset
dataset_path = Path(__file__).parent / 'diabetes_dataset1.csv'
df = pd.read_csv(dataset_path)

# Title of the Streamlit app
st.title("Diabetes Prediction")

# User input fields
st.subheader("Diabetes Prediction for User Input")
g = st.selectbox("Enter your gender", ("Male", "Female"))

if g == "Female":
    gen = 0
else:
    gen = 1

age = st.number_input("Enter your age")
h = st.selectbox("Are you suffering from hypertension", ("No", "Yes"))
hyp = 1 if h == "Yes" else 0
hd = st.selectbox("Are you suffering from any heart disease", ("Yes", "No"))
hdis = 1 if hd == "Yes" else 0
bmi = st.number_input("Enter your BMI")
hb = st.number_input("Enter your HbA1c level")
glu = st.number_input("Enter your blood glucose level")

# Predict button
button = st.button("Predict")

if button:
    model = load_util()
    ans = model.predict([[gen, age, hyp, hdis, bmi, hb, glu]])
    if ans == 1:
        st.write("# You are suffering from Diabetes")
    else:
        st.write("# You are not suffering from Diabetes")






st.title("Data Visualization For Diabetes Dataset")



# Summary statistics

st.subheader("Summary Statistics")

st.write(df.describe())







# Correlation heatmap
st.subheader("Correlation Heatmap")
corr = df.corr()

heatmap_fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
st.pyplot(heatmap_fig)



# Violin plot
st.subheader("Violin Plot")

violin_fig, ax = plt.subplots()

sns.violinplot(x='diabetes', y='age', hue='gender' ,data=df, ax=ax)

st.pyplot(violin_fig)



# Histograms for Age by Gender

# Filter dataset for diabetes = 1

df_diabetes = df[df['diabetes'] == 1]

st.subheader("Histogram")

# Histogram for Age by Gender where Diabetes = 1

fig, ax = plt.subplots()

sns.histplot(data=df_diabetes, x='age', hue='gender', kde=True, bins=20, palette='muted')

plt.xlabel('Age')

plt.ylabel('Frequency')

st.pyplot(fig)

