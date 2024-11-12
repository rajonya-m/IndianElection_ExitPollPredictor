import joblib
import pandas as pd

# Load the trained model
model = joblib.load("saved_model.pkl")

def predict_party(real_emotion_score, emotion_score, age, sentiment_score, economic_cond_national):
    # Prepare input data as a DataFrame for prediction
   # gender_numeric = 1 if gender.lower() == 'male' else 0
    input_data = pd.DataFrame({
        'real_emotion_score': [real_emotion_score],
        'emotion_score': [emotion_score],
        'age': [age],
        'sentiment_score': [sentiment_score],
        'economic_cond_national': [economic_cond_national],
        #'gender': [gender_numeric]
    })

    # Make the prediction
    prediction = model.predict(input_data)[0]
    return prediction
