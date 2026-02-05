import base64
import io
import numpy as np
import librosa
import soundfile as sf
import torch

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

app = FastAPI()

# ======== Load REAL Deepfake Detection Model ========
MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english-deepfake"

print("Loading AI voice detection model... this may take 1–2 minutes on first run.")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

VALID_API_KEY = "sk_test_123456789"

# -------- Request Model --------
class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

def load_audio_from_base64(b64_string):
    """Decode Base64 and convert to proper waveform"""
    audio_bytes = base64.b64decode(b64_string)
    audio_buffer = io.BytesIO(audio_bytes)

    # Load audio at 16kHz (model requirement)
    waveform, sample_rate = librosa.load(audio_buffer, sr=16000)
    return waveform

def predict_deepfake(waveform):
    """Run real AI model on the audio"""
    inputs = feature_extractor(
        waveform, 
        sampling_rate=16000, 
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probability
    probs = torch.nn.functional.softmax(logits, dim=-1)
    ai_prob = float(probs[0][1])   # Probability of AI-generated
    human_prob = float(probs[0][0])

    return ai_prob, human_prob

@app.post("/api/voice-detection")
def detect_voice(request: AudioRequest, x_api_key: str = Header(None)):

    # ---- API Key Validation ----
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ---- Validate Language ----
    supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if request.language not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # ---- Decode Audio ----
    try:
        waveform = load_audio_from_base64(request.audioBase64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # ---- Run REAL AI Model ----
    ai_prob, human_prob = predict_deepfake(waveform)

    if ai_prob > human_prob:
        classification = "AI_GENERATED"
        confidence = round(ai_prob, 2)
        explanation = "Model detected synthetic artifacts consistent with AI-generated speech."
    else:
        classification = "HUMAN"
        confidence = round(human_prob, 2)
        explanation = "Model detected natural speech variations consistent with human voice."

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
