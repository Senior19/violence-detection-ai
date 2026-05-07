import os
import torch
from torch.utils.data import Dataset
from utils.video_utils import extract_uniform_frames

class ViolenceDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = []
        self.augment = augment

        fight_dir = os.path.join(root_dir, "Fight")
        nonfight_dir = os.path.join(root_dir, "NonFight")

        for v in os.listdir(fight_dir):
            if v.endswith(('.mp4', '.avi', '.mov')):
                self.samples.append((os.path.join(fight_dir, v), 1))

        for v in os.listdir(nonfight_dir):
            if v.endswith(('.mp4', '.avi', '.mov')):
                self.samples.append((os.path.join(nonfight_dir, v), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        try:
            frames = extract_uniform_frames(video_path, augment=self.augment)
        except Exception:
            frames = extract_uniform_frames(
                self.samples[(idx + 1) % len(self.samples)][0],
                augment=self.augment
            )
        return frames, torch.tensor(label, dtype=torch.long)