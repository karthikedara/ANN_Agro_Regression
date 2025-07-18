import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import streamlit as st

#Load the model
model = tf.keras.models.load_model('agriculture.h5')

#Load the Scalar and Encoders

with open('agr_le.pkl','rb') as file:
    le = pickle.load(file)
with open('agr_oe.pkl','rb') as file:
    oe = pickle.load(file)
with open('agr_sc.pkl','rb') as file:
    sc = pickle.load(file)
with open('ohe.pkl','rb') as file:
    ohe = pickle.load(file)

# Streamlit UI
st.title('Agriculture Crop Yield Prediction')
region = st.selectbox('Region',ohe.categories_[0])
Soil_type = st.selectbox('Soil Type',ohe.categories_[1])
crop = st.selectbox('Crop',ohe.categories_[2])
RainFall_mm = st.number_input("RainFall in mm",value=0.0)
Temperature_Celsius = st.number_input('Temp in Celsius',0,100)
Fertilizer_used = st.selectbox('Fertizers used',[0,1])
irrigation_used = st.selectbox('Irrigation used',[0,1])
Weather_condition = st.selectbox('Weather Condition',ohe.categories_[3])
Days_to_harvest = st.number_input('Days to Harvest')

#Load the Input data into Model suitable format
input_data = pd.DataFrame({
    'Region': [region],
    'Soil_Type':[Soil_type],
    'Crop': [crop],
    'Rainfall_mm': [RainFall_mm],
    'Temperature_Celsius' : [Temperature_Celsius],
    'Fertilizer_Used' : [le.transform([Fertilizer_used])[0]],
    'Irrigation_Used' : [le.transform([irrigation_used])[0]],
    'Weather_Condition' : [Weather_condition],
    'Days_to_Harvest' : [Days_to_harvest]
})
input_transformed = oe.transform(input_data)
feature_names = oe.get_feature_names_out()
columns = ['remainder__Rainfall_mm','remainder__Temperature_Celsius','remainder__Days_to_Harvest']
indices = [np.where(feature_names == col)[0][0] for col in columns]

numeric_data = pd.DataFrame(input_transformed[:, indices], columns=['remainder__Rainfall_mm', 'remainder__Temperature_Celsius', 'remainder__Days_to_Harvest'])

#st.write(input_transformed)
scaled_data  = sc.transform(numeric_data)

input_transformed[:, indices] = scaled_data


prediction = model.predict(input_transformed)
predict = prediction[0][0]

st.write('The prediction value is {:.2f}'.format(predict))