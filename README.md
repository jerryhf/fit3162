# FIT3162 - Final Year Project 2

# Info
- Video and landmark in the Google Drive ch-sims-landmark and ch-sims-videos
- Files starting with ._ are removed
- 4 videos I remember are missing from ch-sims-videos, not sure where

# Next steps
- Preprocessing for GRID and LRS is different
    - GRID needs to use extract_frames -> preprocess. preprocess is using ref_face, which is the reference landmark for the face. But it seems to be different format than the LM given for GRID (different length/amount of LMs when I split by '|' and then split by ',')
    - LRS only need to extract audio, and then continue training.
- Maybe the LM given for LRS and GRID is different, such that GRID needs to be processed first while LRS can immediately be used.
- Need to check both approach, which can be used with our extracted LMs.

