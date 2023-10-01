import os
import cv2
from time import sleep
import mediapipe as mp

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
    6, 195, 5, 4, 1,
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
    454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    127, 227, 234, 93, 132, 58, 
    172, 136,
    150, 149, 176, 148, 152, 377,
    400, 378, 379, 365, 397, 288,
    435, 433, 361, 401, 323, 366,
    372, 368, 345, 454, 447, 376,
    433
]

def process_video(input_video_path, output_file_path):
    """
    Process a specific video and save landmarks to a specific output file.

    Parameters:
    input_video_path (str): Path to the input video file.
    output_file_path (str): Path to the output landmarks file.
    """
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file at {input_video_path}")
        return

    landmarks_coordinates = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        temp_landmark_coords = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for index in highlighted_landmarks_indices:
                    landmark = face_landmarks.landmark[index]
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    temp_landmark_coords.append(f"{x} {y}")
                    cv2.circle(frame, (x,y), 2, (0, 255, 0), -1)
        else:
            # Handle frames with no detected landmarks by appending empty string
            print(f"No landmarks detected in frame {len(landmarks_coordinates) + 1} of video")

        # Append the landmarks of this frame (even if empty) to landmarks_coordinates
        landmarks_coordinates.append(','.join(temp_landmark_coords))

    # Save landmarks to specified output file
    output_data = '|'.join(landmarks_coordinates)
    with open(output_file_path, 'w') as file:
        file.write(output_data)

    cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Example usage
input_video = "ch-sims-videos/aqgy3_0002/00022.mp4"
output_file = "test_00022_landmarks.txt"
process_video(input_video, output_file)