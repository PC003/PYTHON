
import streamlit as st
import pickle
import pandas as pd

# ------------------ LOAD MODELS & ENCODERS ------------------ #
model = pickle.load(open('model.pkl','rb'))
le = pickle.load(open('le.pkl','rb'))        # teams + winner
le1 = pickle.load(open('le1.pkl','rb'))      # city
le2 = pickle.load(open('le2.pkl','rb'))      # venue
ord_enc = pickle.load(open('ord.pkl','rb'))  # match_type

# ------------------ UI ------------------ #
st.set_page_config(page_title="IPL Predictor", layout="centered")

st.title("🏏 IPL Match Winner Predictor")

# Get values from encoders
teams = list(le.classes_)
cities = list(le1.classes_)
venues = list(le2.classes_)

# Inputs
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

if team1 == team2:
    st.error("⚠️ Team1 and Team2 must be different")

toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ['bat', 'field'])

city = st.selectbox("City", cities)
venue = st.selectbox("Venue", venues)

match_type = st.selectbox("Match Type", [
    'League','Eliminator','Qualifier 1','Qualifier 2',
    'Elimination Final','Semi Final','3rd Place Play-Off','Final'
])

# ------------------ PREDICTION ------------------ #
if st.button("Predict Winner"):

    if team1 == team2:
        st.stop()

    # Create dataframe
    input_data = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'city': [city],
        'venue': [venue],
        'match_type': [match_type]
    })

    try:
        # Apply encoders
        input_data['team1'] = le.transform(input_data['team1'])
        input_data['team2'] = le.transform(input_data['team2'])
        input_data['toss_winner'] = le.transform(input_data['toss_winner'])

        input_data['city'] = le1.transform(input_data['city'])
        input_data['venue'] = le2.transform(input_data['venue'])

        input_data['match_type'] = ord_enc.transform(input_data[['match_type']])

        # (Optional) encode toss_decision if you used encoding during training
        # Example:
        input_data['toss_decision'] = input_data['toss_decision'].map({'bat':0, 'field':1})
        input_data = input_data[
    ['city', 'match_type', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision']
]
        # Prediction
        prediction = model.predict(input_data)
        winner = le.inverse_transform(prediction)

        # Probability (if classifier)
        try:
            prob = model.predict_proba(input_data)
            confidence = round(max(prob[0]) * 100, 2)
            st.success(f"🏆 Predicted Winner: {winner[0]}")
            st.info(f"Confidence: {confidence}%")
        except:
            st.success(f"🏆 Predicted Winner: {winner[0]}")

    except Exception as e:
        st.error(f"Error: {e}")

