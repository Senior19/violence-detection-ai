import cv2
import numpy as np
import torch
from torchvision import transforms
from config import IMG_SIZE, SEQUENCE_LENGTH

# Training augmentation pipeline
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMG_SIZE, padding=8),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def extract_uniform_frames(video_path, num_frames=SEQUENCE_LENGTH, augment=False):
    """Uniformly sample frames — more stable than random sliding windows."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < num_frames:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
    else:
        indices = [int(i * total / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t = train_transform if augment else val_transform
        frames.append(t(frame))

    cap.release()
    return torch.stack(frames)   # (T, C, H, W)