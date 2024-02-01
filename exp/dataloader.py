import json
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class TDataset(Dataset):
    def __init__(self, path, specaug=False):
        with open(path) as f:
            self.data = json.load(f)
        self.data_idx = list(self.data.keys())
        self.specaug = specaug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[self.data_idx[index]]
        data_path = data["fbank"] 
        fbank = torch.load(data_path)
        fbank_mean = torch.mean(fbank, dim=0, keepdims=True)
        fbank_std = torch.std(fbank, dim=0, keepdims=True)
        fbank = (fbank - fbank_mean) / fbank_std
        phn = data["phn"]
        duration = data["duration"]
        return fbank, phn, duration

def collate_wrapper(batch):
    fbank = pad_sequence([i[0] for i in batch])
    lens = torch.tensor([len(i[0]) for i in batch], dtype=torch.long)
    phn = [i[1] for i in batch]
    duration = [i[2] for i in batch]
    return fbank, lens, phn, duration

def get_dataloader(path, bs, shuffle, specaug=False):
    dataset = TDataset(path, specaug)
    return DataLoader(
        dataset, 
        batch_size=bs, 
        shuffle=shuffle,
        collate_fn=collate_wrapper, 
        pin_memory=True
    )

