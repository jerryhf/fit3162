import cv2
import os

def visualize_landmarks(input_video_path, input_landmark_path, output_video_path):
    """
    Visualizes landmarks on a video and saves the annotated video.

    Parameters:
    input_video_path (str): Path to the input video file.
    input_landmark_path (str): Path to the input landmark file.
    output_video_path (str): Path to save the output video file.
    """

    # Read landmark coordinates from file
    with open(input_landmark_path, 'r') as file:
        landmarks_data = file.read()
    landmarks_per_frame = landmarks_data.split('|')

    # Open input video file
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Set up codec and output video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    frame_count = 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if there are landmarks for the current frame
        if frame_count < len(landmarks_per_frame):
            # Get landmark coordinates for current frame
            landmarks = landmarks_per_frame[frame_count].split(',')
            for landmark in landmarks:
                if landmark:  # Check if landmark is not an empty string
                    x, y = map(int, landmark.split())
                    length = 5  # Length of each arm of the cross
                    color = (0, 255, 0)  # Green color
                    thickness = 2  # Line thickness

                    cv2.line(frame, (x - length, y), (x + length, y), color, thickness)
                    cv2.line(frame, (x, y - length), (x, y + length), color, thickness)
        
        # Write the frame to the output video, with or without landmarks
        out.write(frame)
        
        frame_count += 1


    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


video_input_id = r"00000.mpg"

video_input = r"D:\Lenovo\Documents\TrueDocuments\MonashLessons\july_2023\final-year-project-fit3162\aqgy3_0001\00000.mp4"
landmark_input = r"D:\Lenovo\Documents\TrueDocuments\MonashLessons\july_2023\final-year-project-fit3162\ch-sims-landmark\aqgy3_0001_landmarks\00000_landmarks.txt"
output_video = f"./00000_visualize.mp4"

# Example of how to use the function
visualize_landmarks(video_input, landmark_input, output_video)
