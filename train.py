import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataset_loader import ViolenceDataset
from models.model import ViolenceModel
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datasets — augment only training data
train_dataset = ViolenceDataset("dataset/train", augment=True)
val_dataset   = ViolenceDataset("dataset/val",   augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

model = ViolenceModel(num_classes=NUM_CLASSES).to(device)

# Label smoothing reduces overconfidence → lower val loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)

# OneCycleLR: fast warmup then cosine decay — faster convergence
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.3
)

# AMP for 2–3× speed on GPU
scaler = GradScaler()

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for videos, labels in train_loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with autocast():
            outputs = model(videos)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item() * videos.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(videos)

            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    lr_now = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {train_loss:.4f} | "
          f"Val Acc: {acc:.4f} | "
          f"LR: {lr_now:.6f}")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': best_acc
        }, "best_model.pth")
        print("  ✅ Best model saved!")

print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")