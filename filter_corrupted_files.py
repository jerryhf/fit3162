import os
import fnmatch

"""
Script to remove all files that start with ._
"""
root_dir = "./ch-sims-videos"  # Replace with the path to your root directory

# Walk through root_dir
for foldername, subfolders, filenames in os.walk(root_dir):
    # Find and remove all ._*.txt files
    for filename in fnmatch.filter(filenames, '._*.mp4'):
        file_path = os.path.join(foldername, filename)
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except OSError as e:
            print(f"Error removing {file_path}: {e}")
