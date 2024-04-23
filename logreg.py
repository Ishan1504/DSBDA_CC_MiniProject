import pandas as pd
import joblib as jb
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('diabetes_dataset1.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

model = LogisticRegression()
model.fit(x,y)
jb.dump(model,"mod.pkl")

