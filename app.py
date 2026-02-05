import base64
import io
import numpy as np
import librosa
import torch

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

app = FastAPI()

# ======== WORKING PUBLIC MODEL (Render-friendly) ========
MODEL_NAME = "anton-l/wav2vec2-base-superb-antispoofing"

print("Loading AI voice detection model...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

VALID_API_KEY = "sk_test_123456789"

class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

def load_audio_from_base64(b64_string):
    audio_bytes = base64.b64decode(b64_string)
    audio_buffer = io.BytesIO(audio_bytes)

    # Model requires 16kHz audio
    waveform, _ = librosa.load(audio_buffer, sr=16000)
    return waveform

def predict_deepfake(waveform):
    inputs = feature_extractor(
        waveform, 
        sampling_rate=16000, 
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Label mapping for this model:
    # 0 = HUMAN (bona fide)
    # 1 = AI / spoof
    ai_prob = float(probs[0][1])
    human_prob = float(probs[0][0])

    return ai_prob, human_prob

@app.post("/api/voice-detection")
def detect_voice(request: AudioRequest, x_api_key: str = Header(None)):

    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if request.language not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    try:
        waveform = load_audio_from_base64(request.audioBase64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    ai_prob, human_prob = predict_deepfake(waveform)

    if ai_prob > human_prob:
        classification = "AI_GENERATED"
        confidence = round(ai_prob, 2)
        explanation = "Model detected characteristics consistent with synthetic or spoofed speech."
    else:
        classification = "HUMAN"
        confidence = round(human_prob, 2)
        explanation = "Model detected natural speech patterns consistent with human voice."

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
