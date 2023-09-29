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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get landmark coordinates for current frame
        landmarks = landmarks_per_frame[frame_count].split(',')
        for landmark in landmarks:
            x, y = map(int, landmark.split())
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Draw green dots for landmarks
        
        # Write the frame to the output video
        out.write(frame)
        
        frame_count += 1

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()


video_input_id = r"00000"

video_input = f"./aqgy3_0001/{video_input_id}.mp4"
landmark_input = f"./ch-sims-landmark/{video_input_id}_landmarks_output.txt"
output_video = f"./{video_input_id}_visualize.mp4"

# Example of how to use the function
visualize_landmarks(video_input, landmark_input, output_video)
