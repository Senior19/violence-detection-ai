import cv2
import torch
import numpy as np
from collections import deque
from models.model import ViolenceModel
from utils.video_utils import val_transform
from config import IMG_SIZE, SEQUENCE_LENGTH
import torch.nn.functional as F
from torch.cuda.amp import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("best_model.pth", map_location=device)
model = ViolenceModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    frame_buffer.append(val_transform(rgb))

    if len(frame_buffer) == SEQUENCE_LENGTH:
        seq = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)  # (1,T,C,H,W)

        with torch.no_grad(), autocast():
            outputs = model(seq)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        label = "Fight" if pred == 1 else "No Fight"
        conf  = probs[0][pred].item()
        color = (0, 0, 255) if pred == 1 else (0, 200, 0)

        cv2.putText(frame, f"{label} ({conf:.2f})", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        if pred == 1 and conf > 0.75:
            print(f"⚠️  Violence detected! ({conf:.2f})")

    cv2.imshow("Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()