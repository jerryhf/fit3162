import os
import glob
import subprocess
import argparse
import fnmatch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Grid_dir', type=str, default="Data dir to GRID_corpus")
    parser.add_argument("--Output_dir", type=str, default='Output dir Ex) ./GRID_imgs_aud')
    args = parser.parse_args()
    return args

args = parse_args()

all_dirs = [dirpath for dirpath, dirnames, filenames in os.walk(args.Grid_dir)]
filtered_dirs = [dir for dir in all_dirs if fnmatch.fnmatch(os.path.basename(dir), 'aqgy*')]
print(filtered_dirs)

for dir in filtered_dirs:
    # Recursively find all .mp4 files in Grid_dir
    vid_files = sorted(glob.glob(os.path.join(args.Grid_dir, '**', '*.mp4'), recursive=True))
    for k, v in enumerate(vid_files):
        t, f_name = os.path.split(v)  # Get the filename
        t, video_dir = os.path.split(t)  # Get the video directory name (which is a subfolder in Grid_dir)
        _, sub_name = os.path.split(t)  # Get the subfolder name in Grid_dir

        # Set up output directories
        out_aud = os.path.join(args.Output_dir, video_dir, 'audio')
        out_im = os.path.join(args.Output_dir, video_dir, 'video', f_name[:-4])
        
        # Create directories if they don't exist
        if not os.path.exists(out_aud):
            os.makedirs(out_aud)
        if not os.path.exists(out_im):
            os.makedirs(out_im)

        # Check if frames already extracted, if not, extract frames and audio
        if len(glob.glob(os.path.join(out_im, '*.png'))) < 75:
            subprocess.call(f'ffmpeg -y -i {v} -qscale:v 2 -r 25 {os.path.join(out_im, "%02d.png")}', shell=True)
            subprocess.call(f'ffmpeg -y -i {v} -ac 1 -acodec pcm_s16le -ar 16000 {os.path.join(out_aud, f_name[:-4] + ".wav")}', shell=True)
