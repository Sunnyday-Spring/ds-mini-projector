from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time
import os
from pythainlp.tokenize import word_tokenize
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ปลดล็อกให้หน้าเว็บดึงข้อมูลได้
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- โหลดโมเดลรอไว้ใน RAM (ทำครั้งเดียวตอนเปิดเครื่อง) ---
print("🚀 Loading model and vectorizer...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'model_v1.joblib'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.joblib'))
print("✅ Model is ready in RAM!")

class PredictRequest(BaseModel):
    text: str

@app.get("/model/info")
def get_info():
    return {"version": "v1.0-netflix-thai", "type": "Logistic Regression"}

@app.post("/predict")
def predict(request: PredictRequest):
    start_time = time.time()
    
    # 1. Preprocess
    tokens = word_tokenize(request.text, engine="newmm")
    clean_text = " ".join(tokens)
    
    # 2. Predict (เร็วมากเพราะทุกอย่างอยู่ใน RAM)
    features = vectorizer.transform([clean_text])
    label = model.predict(features)[0]
    confidence = float(max(model.predict_proba(features)[0]))
    
    latency = (time.time() - start_time) * 1000
    
    return {
        "label": label,
        "confidence": confidence,
        "latency": f"{latency:.2f} ms",
        "model_version": "v1.0-netflix-thai"
    }