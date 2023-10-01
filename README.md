# FIT3162 - Final Year Project 2

# Info
- Video and landmark in the Google Drive ch-sims-landmark and ch-sims-videos
- Files starting with ._ are removed
- 4 videos I remember are missing from ch-sims-videos, not sure where just continue first
- Dandi have complete LMs (dandi_face_landmark), I have significantly less (ch-sims-landmark) -> 98 landmark is what is expected
- Frame with no detected landmarks are appended as empty strings in the landmark.txt file. e.g. 987 202, ... || 627 405,...

# Current Issues
- Current preprocessing produces a very blurred resulting video, and it is also rotated very weirdly. This is not seen when trying with the original GRID dataset. Possible causes:
    - Ref_face is not modified for our landmark
    - Transformation in preprocess does not work well for our landmarks
- During preprocessing, some frames do not have detected landmark, therefore it is empty string in the landmarks.txt file. Currently it is added as empty string in the landmark.txt file, and will be skipped during preprocessing.
- Some videos do not have detected landmarks at all, this will result in error during cropped video in preprocessing:  
Error: unexpected shape of cropped_video for file 00044_landmarks.txt. Shape: torch.Size([0]) -> Check error_log.txt  
This can be fixed by filtering videos where no landmark is detected at all, I am currently working with the data that has not been filtered out.