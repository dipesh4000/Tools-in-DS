from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime
import hashlib
import json
import redis

app = FastAPI(
    title="TDS Mini Data Science API",
    version="1.0.0",
    description="Prediction API with validation, cache, job status, and background logging"
)

r = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = "en"

class PredictResponse(BaseModel):
    id: str
    label: str
    score: float
    source: str

class FeedbackRequest(BaseModel):
    correct_label: str

def fake_model(text: str) -> dict:
    # Replace this with a real ML model later
    text_lower = text.lower()

    if "good" in text_lower or "great" in text_lower or "excellent" in text_lower:
        return {"label": "positive", "score": 0.90}

    if "bad" in text_lower or "poor" in text_lower or "wrong" in text_lower:
        return {"label": "negative", "score": 0.85}

    return {"label": "neutral", "score": 0.60}

def log_event(event: dict):
    # Small background task only
    with open("api-events.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def make_cache_key(payload: PredictRequest) -> str:
    # Same input should produce same cache key
    raw = payload.model_dump_json()
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"predict:cache:{digest}"

@app.get("/")
def root():
    return {
        "message": "Mini data-science API is running",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    try:
        r.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "down"

    return {
        "api": "ok",
        "redis": redis_status
    }

@app.post("/predict", response_model=PredictResponse, status_code=201)
def predict(payload: PredictRequest, background_tasks: BackgroundTasks):
    cache_key = make_cache_key(payload)

    cached = r.get(cache_key)
    if cached:
        data = json.loads(cached)
        return PredictResponse(**data, source="redis-cache")

    prediction = fake_model(payload.text)

    prediction_id = str(uuid4())

    response = {
        "id": prediction_id,
        "label": prediction["label"],
        "score": prediction["score"],
        "source": "computed"
    }

    # Store full prediction for lookup
    r.setex(f"predict:item:{prediction_id}", 3600, json.dumps(response))

    # Cache same input for 5 minutes
    r.setex(cache_key, 300, json.dumps(response))

    # Store temporary job/status style information
    r.setex(f"predict:status:{prediction_id}", 3600, "completed")

    background_tasks.add_task(log_event, {
        "time": datetime.now().isoformat(),
        "event": "prediction_created",
        "id": prediction_id,
        "label": prediction["label"]
    })

    return response

@app.get("/predict/{prediction_id}")
def get_prediction(prediction_id: str):
    data = r.get(f"predict:item:{prediction_id}")

    if not data:
        raise HTTPException(404, "Prediction not found or expired")

    return json.loads(data)

@app.patch("/predict/{prediction_id}/feedback")
def add_feedback(prediction_id: str, feedback: FeedbackRequest):
    data = r.get(f"predict:item:{prediction_id}")

    if not data:
        raise HTTPException(404, "Prediction not found or expired")

    item = json.loads(data)
    item["feedback"] = feedback.correct_label

    r.setex(f"predict:item:{prediction_id}", 3600, json.dumps(item))

    return {
        "message": "feedback saved",
        "prediction": item
    }

@app.delete("/predict/{prediction_id}", status_code=204)
def delete_prediction(prediction_id: str):
    deleted = r.delete(
        f"predict:item:{prediction_id}",
        f"predict:status:{prediction_id}"
    )

    if deleted == 0:
        raise HTTPException(404, "Prediction not found")