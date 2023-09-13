import mediapipe as mp
import cv2

# Initialize mediapipe face detection and drawing modules
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh()

# Landmark indices
highlighted_landmarks_indices = [
    # Right eye upper
    246, 161, 160, 159, 158, 157, 173,
    # Right eye lower
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    # Left eye upper
    466, 388, 387, 386, 385, 384, 398,
    # Left eye lower
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    # Midway between eyes
    168,
    # Nose tip
    1, 9, 8, 168, 6, 197, 195, 5, 4,
    # Nose bottom
    2,
    # Nose right corner
    98,
    # Nose left corner
    327,
    # Lips upper outer
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # Lips lower outer
    146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # Lips upper inner
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    # Lips lower inner
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # Silhouette
    454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,  132, 93,  234, 127

]

# Capture video
cap = cv2.VideoCapture('/Users/jerryhf/Downloads/00000 (1).mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get the landmarks
    results = face_mesh.process(rgb_frame)

    # Highlight specific landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for index in highlighted_landmarks_indices:
                landmark = face_landmarks.landmark[index]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green dots for highlighted landmarks
    
    # Display the frame
    cv2.imshow('MediaPipe FaceMesh', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
