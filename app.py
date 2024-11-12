import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from .model import predict_party  # Import the prediction function

# Load the dataset for visualization purposes
df = pd.read_excel("Updated_Indian_Election_Data.xlsx")

# Streamlit app title and description
st.title("Indian Election Exit Poll Prediction")
st.write("Predict the preferred political party based on voter data and visualize election insights.")

# Input fields for prediction
st.header("Prediction Input")
real_emotion_score = st.number_input("Real Emotion Score", min_value=0.0, max_value=1.0, step=0.01)
emotion_score = st.number_input("Emotion Score", min_value=0.0, max_value=1.0, step=0.01)
age = st.number_input("Age", min_value=18, max_value=100, step=1)
#gender = st.selectbox("Gender", options=['Male', 'Female'])
sentiment_score = st.number_input("Sentiment Score", min_value=0.0, max_value=1.0, step=0.01)
economic_cond_national = st.slider("economic_cond_national", 0, 5, 2)

# Predict button
if st.button("Predict Party"):
    # Make the prediction
    prediction = predict_party(real_emotion_score, emotion_score, age, sentiment_score, economic_cond_national)
    st.write(f"The predicted preferred party is: **{prediction}**")

# Visualization
st.header("Visualizations")

# Plot real_emotion_score vs vote
st.subheader("Real Emotion Score vs Preferred Party (Vote)")
fig, ax = plt.subplots()
df.groupby("vote")["real_emotion_score"].mean().plot(kind="bar", color=['blue', 'green'], ax=ax)
ax.set_xlabel("Preferred Party")
ax.set_ylabel("Average Real Emotion Score")
st.pyplot(fig)

# Plot other columns as required
#st.subheader("Gender Distribution")
#fig, ax = plt.subplots()
#df['gender'].value_counts().plot(kind="pie", autopct='%1.1f%%', colors=['pink', 'lightblue'], ax=ax)
#ax.set_ylabel("")
#st.pyplot(fig)

st.subheader("Age Distribution by Preferred Party")
fig, ax = plt.subplots()
df.groupby("vote")["age"].plot(kind="kde", ax=ax, legend=True)
ax.set_xlabel("Age")
st.pyplot(fig)

st.subheader("Sentiment Score Distribution")
fig, ax = plt.subplots()
df["sentiment_score"].plot(kind="hist", bins=10, color="purple", ax=ax)
ax.set_xlabel("Sentiment Score")
st.pyplot(fig)

st.subheader("Economic Condition vs Preferred Party")
fig, ax = plt.subplots()
df.groupby("vote")["economic_cond_national"].mean().plot(kind="bar", color=['orange', 'cyan'], ax=ax)
ax.set_xlabel("Preferred Party")
ax.set_ylabel("Average Economic Condition")
st.pyplot(fig)


# Optionally display the data table
st.subheader("Dataset")
if st.checkbox("Show raw data"):
    st.write(df)
