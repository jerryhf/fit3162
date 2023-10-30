import os
from moviepy.editor import VideoFileClip

# Directory containing the videos
root_dir = 'C:\\random\\filtered_muted_face_centered_video2'

# Log file to store paths of faulty videos
text_file = 'videos_not_25fps.txt'

# Iterate through the root_directory
for foldername, subfolders, filenames in os.walk(root_dir):
    for filename in filenames:
        # Only check for MP4 files
        if filename.lower().endswith('.mp4'):
            filepath = os.path.join(foldername, filename)
            try:
                with VideoFileClip(filepath) as vid:
                    fps = vid.fps
                # If fps is not 25, log the faulty video to the text file
                if fps != 25:
                    # The path of the faulty video
                    rel_path = os.path.relpath(filepath, root_dir)
                    # Write the relative path of the faulty video to the log file
                    with open(text_file, 'a') as f:
                        f.write(rel_path + '\n')
            except Exception as e:
                print(f"Error processing file {filepath}: {str(e)}")

print("Finished processing the videos.")





