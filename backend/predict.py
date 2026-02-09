import sys
import json
import joblib
import os
from pythainlp.tokenize import word_tokenize

# ฟังก์ชัน Preprocessing ที่ตรงกับตอนเทรน [cite: 22, 26]
def preprocess_thai(text):
    text = " ".join(text.split())
    tokens = word_tokenize(text, engine="newmm")
    return " ".join(tokens)

def main():
    if len(sys.argv) < 2:
        return

    input_text = sys.argv[1]
    
    try:
        # ใช้ Absolute path หรือ path ที่สัมพันธ์กับไฟล์นี้
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'models', 'model_v1.joblib')
        vectorizer_path = os.path.join(base_path, 'models', 'tfidf_vectorizer.joblib')

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # ทำความสะอาดข้อมูล [cite: 21]
        clean_text = preprocess_thai(input_text)
        features = vectorizer.transform([clean_text])
        
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        confidence = float(max(probs))

        # ส่งกลับเป็น JSON [cite: 63]
        result = {
            "label": str(prediction),
            "confidence": confidence
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()