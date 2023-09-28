import mediapipe as mp
import cv2
from time import sleep

# Initialize mediapipe face detection and drawing modules
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh()

# Landmark indices
highlighted_landmarks_indices = [
    # Right eye upper
    246, 
    # 161, 
    160, 
    159, 
    # 158, 
    157, 
    # 173,
    # # Right eye lower
    # 33, 
    7, 
    163, 
    144, 
    145, 
    153, 
    154, 
    # 155, 
    # 133,
    # # Left eye upper
    466, 
    # 388, 
    387, 
    386, 
    # 385, 
    384, 
    # 398,
    # # Left eye lower
    # 263, 
    249, 
    390, 
    373, 
    374, 
    380, 
    # 381, 
    # 382, 
    # 362,
    # # Midway between eyes
    # 168,
    # # Nose tip
    # 1, 9, 8, 
    6, 195, 5, 4,
    # # Nose bottom
    2,
    # # Nose right corner
    98, 97,
    # # Nose left corner
    327, 326,
    # # Lips upper outer
    # 37,0,267
    # 72,11,302, 40,270,
    37, 0, 267, 40, 270,
    # Middle lips
    57, 291, 178, 81, 14, 13, 402, 314,
    # 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # # Lips lower outer
    91, 84, 17, 314,321, 
    # # Lips upper inner
    # 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    # # Lips lower inner
    # 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # # Silhouette
    # 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,  132, 93,  234, 127
    127, 227, 234, 93, 132, 58, 
    172, 136,
    150, 149, 176, 148, 152, 377,
    400, 378, 379, 365, 397, 288,
    435, 433, 361, 401, 323, 366,
    372, 368, 345, 454, 447, 376,
    433
]

# Capture video
cap = cv2.VideoCapture(r".\aqgy3_0001/00000.mp4")
# cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\bbaf3s.mpg")
paused = False

landmarks_coordinates = []

while cap.isOpened():
    # If not paused, then read next frame
    if not paused:
        ret, frame = cap.read()
        if not ret:
            sleep(1)
            break
        # Scale down the video frame.
        # scale_percent = 50  # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get the landmarks
        results = face_mesh.process(rgb_frame)

        # Highlight specific landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                temp_landmark_coords = []

                for index in highlighted_landmarks_indices:
                    landmark = face_landmarks.landmark[index]
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    temp_landmark_coords.append(f"{x} {y}")
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green dots for highlighted landmarks
                landmarks_coordinates.append(','.join(temp_landmark_coords))
        
    # Display the frame
    cv2.imshow('MediaPipe FaceMesh', frame)
    key = cv2.waitKey(1)
    if key == ord(" "):
        paused = not paused
    elif key == ord("q"):
        break

output_data = '|'.join(landmarks_coordinates)
with open("landmarks_output.txt","w") as file:
    file.write(output_data)

cap.release()
cv2.destroyAllWindows()
