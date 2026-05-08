import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Students Social Media Addiction.csv")

# Addiction label function
def addiction_label(score):
    if score <= 3:
        return 'Low'
    elif score <= 7:
        return 'Moderate'
    else:
        return 'High'

df['Addiction_Level'] = df['Addicted_Score'].apply(addiction_label)

predictors = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']
X = df[predictors]
y = df['Addiction_Level']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

st.title("Student Social Media Addiction Predictor")

age = st.number_input("Age", 10, 40)
usage = st.number_input("Average Daily Usage Hours", 0.0, 15.0)
sleep = st.number_input("Sleep Hours Per Night", 0.0, 12.0)
mental = st.slider("Mental Health Score",1,10)
conflicts = st.slider("Conflicts Over Social Media",0,10)

if st.button("Predict Addiction Level"):

    input_data = [[age, usage, sleep, mental, conflicts]]

    prediction = model.predict(input_data)

    st.success(f"Predicted Addiction Level: {prediction[0]}")
