import cv2
import pandas as pd
import numpy as np
from fer import FER  # Emotion detection library
import mediapipe as mp  # MediaPipe for pose detection
import time

# Load the dataset
df = pd.read_excel("Indian_Election_Data_with_Sentiment.xlsx")

# Initialize the emotion detector (using FER-2013)
emotion_detector = FER()

# Initialize MediaPipe Pose for posture detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Current index for processing the dataset
current_index = 0

# Function to capture webcam frames for one person at a time
def capture_webcam_for_person(person_index):
    global current_index

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Initialize lists to capture emotion and posture scores for this person
    emotion_scores_for_person = []
    posture_scores_for_person = []

    # Capture emotion and posture scores for one person (based on current_index)
    if current_index < len(df):
        print(f"Capturing data for person {person_index + 1}/{len(df)}")

        # Allow time for the camera to initialize
        time.sleep(1)

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image.")
                break

            # Flip the frame horizontally for better user experience
            frame = cv2.flip(frame, 1)

            # Emotion detection: using FER library (it returns a tuple with emotion and score)
            emotion, score = emotion_detector.top_emotion(frame)
            emotion_score = score if score else 0  # If no score, default to 0

            # Posture detection: Using MediaPipe pose model to estimate posture
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                # Here, you can calculate posture score based on keypoints or angles
                posture_score = 5  # Dummy value for posture score (implement detailed scoring)
            else:
                posture_score = 0  # Default score if no pose is detected

            # Append the scores to the lists for this person
            emotion_scores_for_person.append(emotion_score)
            posture_scores_for_person.append(posture_score)

            # Display the webcam feed with annotations
            cv2.putText(frame, f"Emotion: {emotion} ({emotion_score})", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Posture Score: {posture_score}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display the webcam feed
            cv2.imshow("Webcam Feed", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After capturing frames for this person, calculate the average scores
        avg_emotion_score = np.mean(emotion_scores_for_person)
        avg_posture_score = np.mean(posture_scores_for_person)

        # Update the dataset with the average scores
        df.at[current_index, 'emotion_score'] = avg_emotion_score
        df.at[current_index, 'posture_score'] = avg_posture_score

        # Release the webcam and close the window
        cap.release()
        cv2.destroyAllWindows()

        # Move to the next person in the dataset
        current_index += 1

# Function to update the dataset and save it after all people have been processed
def update_dataset():
    # Save the updated dataset back to Excel
    df.to_excel("Updated_Indian_Election_Data.xlsx", index=False)
    print("Dataset updated with emotion and posture scores.")

# Function to process each person sequentially
def process_people_sequentially():
    while current_index < len(df):
        # Capture webcam data for one person
        capture_webcam_for_person(current_index)

        # After capturing data for one person, update the dataset
        print(f"Data for person {current_index} processed.")

        # Sleep for 1 second before processing next person
        time.sleep(1)

    # After all persons are processed, update and save the dataset
    update_dataset()

# Start processing people sequentially
process_people_sequentially()
