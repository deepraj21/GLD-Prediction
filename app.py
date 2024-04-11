import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# my csv columns are Date,SPX,GLD,USO,SLV,EUR/USD use them to predict the GLD where the input parameters will be SPX,USO,SLV,EUR/USD .use any model from sklearn.linear_model to predict GLD.
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('gld_price_data.csv')

# Extract the input features and target variable
X = data[['SPX', 'USO', 'SLV', 'EUR/USD']]
y = data['GLD']

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Create a Streamlit app
st.title('GLD Prediction')
st.write('Enter the input parameters:')
spx = st.number_input('SPX')
uso = st.number_input('USO')
slv = st.number_input('SLV')
eur_usd = st.number_input('EUR/USD')

# Make the prediction
prediction = model.predict([[spx, uso, slv, eur_usd]])

# Display the prediction
st.write('Predicted GLD:', prediction)

# Plot a chart using Plotly
import plotly.express as px

fig = px.line(data, x='Date', y='GLD', title='GLD Historical Data')
st.plotly_chart(fig)
