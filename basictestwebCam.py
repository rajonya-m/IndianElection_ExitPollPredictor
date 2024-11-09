import cv2
import mediapipe as mp
from fer import FER

# Initialize emotion detector and posture detector
emotion_detector = FER()
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

emotion_scores = []
posture_scores = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    emotion_score = 3  # Default to neutral
    posture_score = 3  # Default to neutral posture

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Logic to detect emotion (using FER or facial landmarks)
            emotions = emotion_detector.detect_emotions(frame)
            if emotions:
                emotion_score = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                
            # Logic to calculate posture based on landmarks
            # Example: Using shoulders or head tilt for posture
            # Update posture_score here as needed
            posture_score = 4  # Example

    emotion_scores.append(emotion_score)
    posture_scores.append(posture_score)
    
    # Optionally show the webcam feed
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
