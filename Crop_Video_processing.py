import os
import cv2
import mediapipe as mp
from moviepy.editor import *
import time

def crop_speaker_face(input_folder, output_folder):
    # Initialize MediaPipe face detection module
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Iterate through all video files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_video_path = os.path.join(input_folder, filename)


            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                # Video file is corrupted or cannot be opened
                print(f"Skipping corrupted file: {filename}")
                continue

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            # Get the audio from the input video
            video = VideoFileClip(input_video_path)
            audio = video.audio

            # Initialize video writer
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            output_video_path = os.path.join(output_folder, filename)
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (512, 512))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces in the current frame using MediaPipe
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                        # Crop the frame to 512x512 centered around the detected face
                        size = 512
                        x1 = max(0, x + w // 2 - size // 2)
                        x2 = min(iw, x + w // 2 + size // 2)
                        y1 = max(0, y + h // 2 - size // 2)
                        y2 = min(ih, y + h // 2 + size // 2)
                        cropped_frame = frame[y1:y2, x1:x2]

                        # Write the cropped frame to the output video
                        out.write(cropped_frame)

            # Release video capture and writer for the current video
            cap.release()
            out.release()


            # Check if input and output videos have the same duration
            if not is_video_corrupted(output_video_path) or not have_same_duration(input_video_path, output_video_path) :
                print(f"Duration mismatch between input and output for {output_video_path} or corrupted output, deleting output.")
                os.remove(output_video_path)
            else:
                print(f"Processed file: {output_video_path}")




def has_multiple_faces(input_video_path, min_detection_confidence=0.5):
    # Initialize MediaPipe face detection module
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)

    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame using MediaPipe
        results = face_detection.process(rgb_frame)

        # Check if there are more than one detected faces
        if results.detections and len(results.detections) > 1:
            return True

    # No frame with more than one face detected
    return False


def have_same_duration(input_video_path, output_video_path):
    input_video = VideoFileClip(input_video_path)
    output_video = VideoFileClip(output_video_path)

    return input_video.duration == output_video.duration

def is_video_corrupted(video_path):
    """
    Check if a video file is corrupted or not.

    Parameters:
    - video_path (str): The path to the video file.

    Returns:
    - bool: True if the video is not corrupted, False otherwise.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Video file is corrupted or cannot be opened
            return False
        # Attempt to read a frame
        ret, _ = cap.read()
        # Check if the read operation was successful
        if not ret:
            return False
        # Video is not corrupted
        return True
    except Exception as e:
        # Any exception while checking indicates a potential issue
        return False


# Example usage:
parent_directory = r'E:\Raw\ch-simsv2s\Raw'
for folder_name in os.listdir(parent_directory):
    if os.path.isdir(os.path.join(parent_directory, folder_name)):
        # Create the output folder if it doesn't exist
        output_folder = os.path.join(r'E:\test1', folder_name)
        os.makedirs(output_folder, exist_ok=True)

        crop_speaker_face(os.path.join(parent_directory, folder_name), output_folder)
