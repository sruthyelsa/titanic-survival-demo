import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Titanic Survival Predictor")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2)

embarked_Q = st.checkbox("Embarked at Q (Queenstown)")
embarked_S = st.checkbox("Embarked at S (Southampton)")

# Encode inputs for model
sex_encoded = 1 if sex == "male" else 0
embarked_Q_encoded = 1 if embarked_Q else 0
embarked_S_encoded = 1 if embarked_S else 0

# Prepare feature array matching training features order:
# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_Q_encoded, embarked_S_encoded]])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]
    if prediction[0] == 1:
        st.success(f"The passenger is predicted to SURVIVE with a probability of {prediction_proba:.2f}")
    else:
        st.error(f"The passenger is predicted to NOT SURVIVE with a probability of {1 - prediction_proba:.2f}")
