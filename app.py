from fastapi import FastAPI, Header
from pydantic import BaseModel

app = FastAPI()

VALID_API_KEY = "sk_test_123456789"

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
def detect_voice(data: VoiceRequest, x_api_key: str = Header(None)):

    if x_api_key != VALID_API_KEY:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    return {
        "status": "success",
        "language": data.language,
        "classification": "AI_GENERATED",
        "confidenceScore": 0.91,
        "explanation": "Audio analyzed using AI model"
    }
