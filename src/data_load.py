import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from parse_config import config

class SpeechDataset(Dataset):
    def __init__(self):

        if config.task == "tdsv":
            self.data_config = config.data.TD_SV_data
            self.train_config = config.train.TD_SV_train
        else:
            self.data_config = config.data.TI_SV_data
            self.train_config = config.train.TI_SV_train
            self.frame_length = self.data_config.frame
        
        if config.training:
            self.path = self.data_config.train_path_processed
        else:
            self.path = self.data_config.test_path_processed

        self.speaker_paths = list([os.path.join(self.path, speaker) for speaker in os.listdir(self.path)])

        self.length = len(self.speaker_paths)

    def shuffle(self):
        self.speaker_paths = random.sample(self.speaker_paths, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if config.task == "tisv":
            if idx % self.train_config.N == 0:
                self.frame_length = np.random.randint(self.data_config.frame_low, self.data_config.frame, 1)[0]
        else:
            self.frame_length = self.data_config.frame

        selected_speaker_path = self.speaker_paths[idx]

        utter_per_speaker = np.load(selected_speaker_path)
        shuffle_index = np.random.randint(0, utter_per_speaker.shape[0], self.train_config.M)
        utter_per_speaker = utter_per_speaker[shuffle_index]
        utter_per_speaker = utter_per_speaker[:, :self.frame_length]
        utter_per_speaker = torch.tensor(utter_per_speaker)

        return utter_per_speaker
