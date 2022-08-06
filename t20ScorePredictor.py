import streamlit as st
import xgboost
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBRegressor

pipe = pickle.load(open('pipe.pkl','rb'))

teams = [
    ' ',
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'West Indies',
    'Afghanistan',
    'Pakistan',
    'Sri Lanka',
    'Ireland',
    'Zimbabwe'
]
cities = [' ','Barbados', 'Johannesburg', 'Nottingham', 'Mirpur', 'Delhi',
       'Nagpur', 'Cape Town', 'Durban', 'Dubai', 'Sydney', 'Colombo',
       'Dhaka', 'Auckland', 'Abu Dhabi', 'Southampton', 'Trinidad',
       'St Kitts', 'Cardiff', 'Mumbai', 'Guyana', 'Manchester',
       'Hamilton', 'Bangalore', 'Greater Noida', 'Sharjah', 'Lahore',
       'Harare', 'Centurion', 'Mount Maunganui', 'London', 'Pallekele',
       'St Lucia', 'Chandigarh', 'Lauderhill', 'Wellington', 'Hambantota',
       'Adelaide', 'Kolkata', 'Melbourne', 'Christchurch', 'Chittagong']

st.title("T20 cricket score predictor")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team: ', teams)
with col2:
    bowling_team = st.selectbox('Select bowling team: ', teams)

city = st.selectbox("Select City",cities)

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input("Current Score")
with col4:
    overs_done = st.number_input("Overs Completed")
with col5:
    wickets_out = st.number_input("Wickets Out")

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict score'):
    balls_left = 120 - (overs_done*6)
    wickets_left = 10 - wickets_out
    crr = current_score/overs_done

    input_df = pd.DataFrame({'batting_team':[batting_team],
         'bowling_team': [bowling_team],
         'city': [city],
         'current_score': [current_score],
         'balls_left': [balls_left],
         'wickets_left':[wickets_left],
         'crr':[crr],
         'last_five':[last_five]})


    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result)))
