import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from pythainlp.tokenize import word_tokenize

# --- ขั้นตอนที่ 1: Dataset Understanding [cite: 142-147] ---
# ชี้ไปที่ไฟล์ข้อมูลที่คุณเพิ่งอัปโหลด
FILE_PATH = 'data/7.synthetic_netflix_like_thai_reviews_5000.csv'
df = pd.read_csv(FILE_PATH)

print("--- ข้อมูลเบื้องต้น ---")
print(f"จำนวนข้อมูล: {len(df)} แถว")
print(f"คลาสข้อมูล: {df['label'].unique()}") # จะเห็นเป็น Positive/Negative

# --- ขั้นตอนที่ 2: Preprocessing [cite: 149-155] ---
def preprocess_thai(text):
    if not isinstance(text, str): return ""
    text = " ".join(text.split()) # Whitespace normalization
    tokens = word_tokenize(text, engine="newmm") # ตัดคำภาษาไทย
    return " ".join(tokens)

print("กำลังทำ Preprocessing (อาจใช้เวลาสักครู่)...")
df['clean_text'] = df['text'].apply(preprocess_thai)

# --- ขั้นตอนที่ 3: Baseline Model Training [cite: 156-165] ---
X = df['clean_text']
y = df['label']

# แบ่งข้อมูล 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้าง TF-IDF (word-level)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ใช้ Logistic Regression และ class_weight='balanced' ตามโจทย์
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# --- ขั้นตอนที่ 4: Evaluation [cite: 166-171] ---
y_pred = model.predict(X_test_tfidf)
print("\n--- ผลการประเมินโมเดล ---")
print(f"Macro-F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
print(classification_report(y_test, y_pred))

# --- บันทึกไฟล์เพื่อนำไป Deploy [cite: 165] ---
os.makedirs('backend/models', exist_ok=True)
joblib.dump(model, 'backend/models/model_v1.joblib')
joblib.dump(vectorizer, 'backend/models/tfidf_vectorizer.joblib')

print("\n[สำเร็จ] โมเดลถูกบันทึกไว้ที่ backend/models/ พร้อมใช้งานแล้ว!")