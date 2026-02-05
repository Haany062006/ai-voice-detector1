import base64
import io
import numpy as np
import librosa
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Your API Key (keep same as you used before)
VALID_API_KEY = "sk_test_123456789"

# -------- Request Model --------
class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# -------- Feature Extraction Function --------
def extract_features(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    # Extract key audio features used in deepfake detection
    features = {
        "mean_pitch": float(np.mean(librosa.yin(y, fmin=50, fmax=300))),
        "pitch_variance": float(np.var(librosa.yin(y, fmin=50, fmax=300))),
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
        "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y)))
    }
    return features

# -------- Simple AI vs Human Decision Logic (Dynamic) --------
def classify_voice(features):
    # Heuristic model (simulates AI detection based on real audio properties)

    score = 0

    # Rule 1: AI voices tend to have VERY stable pitch
    if features["pitch_variance"] < 20:
        score += 0.4

    # Rule 2: AI voices often have unnatural spectral patterns
    if features["spectral_centroid"] < 1500:
        score += 0.3

    # Rule 3: AI voices often have low natural variability
    if features["zero_crossing_rate"] < 0.05:
        score += 0.3

    # Final decision
    if score >= 0.5:
        return "AI_GENERATED", round(score, 2), "Low pitch variation and synthetic spectral patterns detected"
    else:
        return "HUMAN", round(1 - score, 2), "Natural pitch variations and organic speech patterns detected"

# -------- API Endpoint --------
@app.post("/api/voice-detection")
def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(None)
):
    # ---- API Key Validation ----
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ---- Validate Language ----
    supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if request.language not in supported_languages:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # ---- Decode Base64 Audio ----
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # ---- Extract Features ----
    features = extract_features(audio_bytes)

    # ---- Classify ----
    classification, confidence, explanation = classify_voice(features)

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
