import os
import shutil
import glob
import cv2
import torchvision
import torch
from skimage import transform
import numpy as np
from scipy import signal
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
import librosa
import soundfile as sf
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_dir', type=str, default="Data dir of images and audio of GRID")
    parser.add_argument('--Landmark', type=str, default="Data dir of GRID Landmark")
    parser.add_argument('--FPS', type=int, default=25, help="25 for GRID")
    parser.add_argument('--reference', type=str, default='./Ref_face.txt')
    parser.add_argument("--Output_dir", type=str, default='Output dir Ex) ./GRID_processed')
    args = parser.parse_args()
    return args

# The rest of your classes and functions

class Crop(object):
    def __init__(self, crop):
        self.crop = crop

    def __call__(self, img):
        return img.crop(self.crop)

class Preprocessing(Dataset):
    def __init__(self, args, refer_lm, tform):
        self.args = args
        self.refer_lm = refer_lm
        self.tform = tform
        self.file_paths = self.build_file_list()
        

    def build_file_list(self):
        file_list = []
        # print(glob.glob(os.path.join(self.args.Landmark, '**', '*.txt'), recursive=True))
        landmarks = sorted(glob.glob(os.path.join(self.args.Landmark, '**', '*.txt'), recursive=True))
        for lm in landmarks:
            if not os.path.exists(lm.replace(self.args.Landmark, self.args.Output_dir)[:-4] + '.mp4'):
                file_list.append(lm)
        return file_list

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Extracting subfolder_id and file_name without extension
        t, file_name_with_ext = os.path.split(file_path)
        file_name = os.path.splitext(file_name_with_ext)[0].replace('_landmarks', '')
        subfolder_id = os.path.basename(t)
        
        # Construct the directory path to the PNGs
        directory = os.path.join(self.args.Data_dir, subfolder_id.replace('_landmarks', ''), 'video', file_name)
        # print(directory)
        ims = sorted(glob.glob(os.path.join(directory, '*.png')))

        # ims = sorted(glob.glob(os.path.join(file_path.replace(self.args.Landmark, self.args.Data_dir)[:-4], '*.png')))
        frames = []

        for im in ims:
            frames += [cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)]
        v = np.stack(frames, 0)

        t, f_name = os.path.split(file_path)
        t, m_name = os.path.split(t)
        _, s_name = os.path.split(t)
        save_path = os.path.join(self.args.Output_dir, subfolder_id.replace('_landmarks', ''))
        try:
            with open(file_path, 'r', encoding='utf-8') as lf:
                lms = lf.readlines()[0]
        except:
            with open(file_path, 'r', encoding='cp949') as lf:
                lms = lf.readlines()[0]
        lms = lms.split('|')
        print(f"Processing: {file_path}, lms length = {len(lms)}, video frame length = {v.shape[0]}")
        # assert v.shape[0] == len(lms), f'the video frame length {v.shape[0]} differs to the landmark frames {len(lms)} for file {file_path}'
        
        aligned_video = []
        log_file = open('error_log.txt', 'a')
        for i, frame in enumerate(v):
            if i >= len(lms):  # Prevent IndexError and handle frame-landmark mismatch
                error_msg = f"Warning: No landmark available for frame {i} in file {file_path}. Skipping frame.\n"
                print(error_msg)
                with open("error_log.txt","a") as log_file:
                    log_file.write(error_msg)
                continue  # or continue, depending on how you want to handle this

            lm = lms[i].split(',')
            temp_lm = []
            for l in lm:
                if not l:  # this checks if l is an empty string
                    print("Warning: No landmark detected in this frame. Skipping.")
                    continue
                x, y = l.split()
                temp_lm.append([x, y])
            if not temp_lm:
                continue
            temp_lm = np.array(temp_lm, dtype=float)  # 98,2

            source_lm = temp_lm

            self.tform.estimate(source_lm, self.refer_lm)
            mat = self.tform.params[0:2, :]
            aligned_im = cv2.warpAffine(frame, mat, (np.shape(frame)[0], np.shape(frame)[1]))
            aligned_video += [aligned_im[:256, :256, :]]

        aligned_video = np.array(aligned_video)

        #### audio preprocessing ####
        audio_file_path = os.path.join(self.args.Data_dir, subfolder_id.replace('_landmarks', ''), 'audio', f"{file_name}.wav")
        aud, _ = librosa.load(audio_file_path, sr=16000)
        fc = 55.  # Cut-off frequency of the filter
        w = fc / (16000 / 2)  # Normalize the frequency
        b, a = signal.butter(7, w, 'high')
        aud = signal.filtfilt(b, a, aud)
        log_file.close()

        return torch.tensor(aligned_video), save_path, f_name, torch.tensor(aud.copy())

def main():
    args = parse_args()
    eps = 1e-8

    f = open(args.reference, 'r')
    lm = f.readlines()[0]
    f.close()
    lm = lm.split(':')[-1].split('|')[6]
    lms = lm.split(',')
    temp_lm = []

    for lm in lms:
        x, y = lm.split()
        temp_lm.append([x, y])
    temp_lm = np.array(temp_lm, dtype=float)
    refer_lm = temp_lm

    tform = transform.SimilarityTransform()
    Data = Preprocessing(args,refer_lm, tform)
    Data_loader = DataLoader(Data, shuffle=False, batch_size=1, num_workers=3)
    

    for kk, data in enumerate(Data_loader):
        cropped_video, save_path, f_name, aud = data
        aud = aud[0]
        cropped_video = cropped_video[0]
        save_path = save_path[0]
        f_name = f_name[0]
        video_path = os.path.join(save_path, 'video')
        audio_path = os.path.join(save_path, 'audio')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        if not os.path.exists(audio_path):
            os.makedirs(audio_path)
        if len(cropped_video.shape) != 4 or cropped_video.shape[3] != 3:
            with open("error_log.txt", 'a') as log_file:
                error_msg = f"Error: unexpected shape of cropped_video for file {f_name}. Shape: {cropped_video.shape}\n"
                print(error_msg)
                log_file.write(error_msg)
            continue
        torchvision.io.write_video(os.path.join(video_path, f_name[:-4] + '.mp4'), video_array=cropped_video, fps=args.FPS)
        sf.write(os.path.join(audio_path, f_name[:-4] + ".flac"), aud.numpy(), samplerate=16000)
        print('##########', kk + 1, ' / ', len(Data_loader), '##########')


if __name__ == '__main__':
    main()
