import os
import glob
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Grid_dir', type=str, default="Data dir to GRID_corpus")
    parser.add_argument("--Output_dir", type=str, default='Output dir Ex) ./GRID_imgs_aud')
    args = parser.parse_args()
    return args


args = parse_args()

vid_files = sorted(glob.glob(os.path.join(args.Grid_dir, '**', '*.mp4'), recursive=True))
for k, v in enumerate(vid_files):
    t, f_name = os.path.split(v)
    t, video_dir = os.path.split(t)
    _, sub_name = os.path.split(t)

    # Create the same directory structure in Output_dir
    out_im = os.path.join(args.Output_dir, sub_name, video_dir, f_name[:-4])
    out_png_folder = os.path.join(out_im, 'PNG')
    out_audio_folder = os.path.join(out_im, 'Audio')

    if len(glob.glob(os.path.join(out_png_folder, '*.png'))) < 75:  # Can resume after being interrupted
        if not os.path.exists(out_png_folder):
            os.makedirs(out_png_folder)
        if not os.path.exists(out_audio_folder):
            os.makedirs(out_audio_folder)

        # Adjust the output file paths for PNG and audio
        out_png_file = os.path.join(out_png_folder, f_name[:-4])
        out_audio_file = os.path.join(out_audio_folder, f_name[:-4])

        subprocess.call(f'ffmpeg -y -i {v} -qscale:v 2 -r 25 {out_png_file}_%02d.png', shell=True)
        subprocess.call(f'ffmpeg -y -i {v} -ac 1 -acodec pcm_s16le -ar 16000 {out_audio_file}.wav', shell=True)
    # print(f'{k}/{len(vid_files)}')