import sys
sys.path.append('..')
from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
import cv2
from collections import deque
from models.model import ViolenceModel
from config import IMG_SIZE, SEQUENCE_LENGTH
import torch.nn.functional as F
from utils.video_utils import extract_frames, create_sliding_windows
import os

app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViolenceModel().to(device)
model_path = os.path.join(os.path.dirname(__file__), '..', 'violence_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

def preprocess(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return frame


@app.get("/")
def home():
    return {"message": "Violence Detection API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # Save uploaded video temporarily
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    
    # Extract frames
    frames = extract_frames(video_path)
    
    if len(frames) < SEQUENCE_LENGTH:
        return {"error": "Video too short, need at least {} frames".format(SEQUENCE_LENGTH)}
    
    # Create sliding windows
    sequences = create_sliding_windows(frames)
    
    predictions = []
    
    for seq in sequences:
        sequence = np.array(seq)
        sequence = torch.tensor(sequence).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            outputs = model(sequence)
            probs = F.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
        
        label = "Fight" if pred.item() == 1 else "NonFight"
        confidence = probs[0][pred].item()
        
        predictions.append({
            "prediction": label,
            "confidence": round(confidence, 3)
        })
    
    # Clean up
    import os
    os.remove(video_path)
    
    # Aggregate results
    fight_count = sum(1 for p in predictions if p["prediction"] == "Fight")
    total = len(predictions)
    risk = "HIGH" if fight_count > total / 2 else "SAFE"
    
    return {
        "total_sequences": total,
        "fight_sequences": fight_count,
        "risk": risk,
        "predictions": predictions[:10]  # Return first 10 for brevity
    }