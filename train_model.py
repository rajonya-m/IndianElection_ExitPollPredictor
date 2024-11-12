from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Load the dataset
df = pd.read_excel("Updated_Indian_Election_Data.xlsx")
#df['gender'] = df['gender'].map({'male': 1, 'female': 0})

# Define features and target, using relevant columns for better prediction accuracy
features = ['real_emotion_score', 'emotion_score', 'age', 'sentiment_score', 'economic_cond_national']
X = df[features]
y = df['vote']  # 'vote' should have values like 'BJP' and 'CONGRESS'

# Convert categorical variables (like 'gender') to numeric, if needed
#X = pd.get_dummies(X, columns=['gender'], drop_first=True)
X.columns = X.columns.str.lower()  # Ensure lowercase column names


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, "saved_model.pkl")
