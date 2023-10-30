import os
import glob
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import argparse
from torch.utils.data import DataLoader, Dataset
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Data_dir', type=str, default="Data dir of images and audio of GRID")
    parser.add_argument('--Output_dir', type=str, default='Output dir Ex) ./GRID_processed')
    args = parser.parse_args()
    return args

class Preprocessing(Dataset):
    def __init__(self, args):
        self.args = args
        self.file_paths = self.build_file_list()
        
    def build_file_list(self):
        audio_files = sorted(glob.glob(os.path.join(self.args.Data_dir, '**', '*.wav'), recursive=True))
        return audio_files

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Extract parent folder and grandparent folder
        parent_folder, file_name_with_ext = os.path.split(file_path)
        grandparent_folder, folder_id = os.path.split(parent_folder)
        
        file_name = os.path.splitext(file_name_with_ext)[0]
        
        # Append folder_id and 'audio' to save path
        save_path = os.path.join(self.args.Output_dir, grandparent_folder,'audio')
        
        #### audio preprocessing ####
        aud, _ = librosa.load(file_path, sr=16000)
        fc = 55.  # Cut-off frequency of the filter
        w = fc / (16000 / 2)  # Normalize the frequency
        b, a = signal.butter(7, w, 'high')
        aud = signal.filtfilt(b, a, aud)
        
        return torch.tensor(aud.copy()), save_path, file_name

def main():
    args = parse_args()
    
    Data = Preprocessing(args)
    Data_loader = DataLoader(Data, shuffle=False, batch_size=1, num_workers=3)

    for kk, data in enumerate(Data_loader):
        aud, save_path, f_name = data
        aud = aud[0]
        save_path = save_path[0]
        f_name = f_name[0]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        sf.write(os.path.join(save_path, f_name + ".flac"), aud.numpy(), samplerate=16000)
        print('##########', kk + 1, ' / ', len(Data_loader), '##########')


if __name__ == '__main__':
    main()
