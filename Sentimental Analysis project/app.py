from flask import Flask, render_template, request
import pickle
import re
from langdetect import detect
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# Load vectorizer and model
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('review', '')
    if not input_text.strip():
        return render_template("index.html", prediction="⚠️ Please enter some text.", review='')

    try:
        lang = detect(input_text)
    except:
        lang = "unknown"

    if lang != "en":
        return render_template("index.html", prediction="⚠️ Please enter review in English.", review=input_text)

    cleaned = clean_text(input_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0].max() * 100

    result = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = f"{probability:.2f}%"

    suggestion = "🎉 Recommended to friends!" if prediction == 1 else "🛠 Suggest improvements."
    word_count = len(input_text.split())
    comment = "✅ Good length." if word_count >= 5 else "⚠️ Try to write a longer review."

    # Extract keywords
    keywords = [word for word in cleaned.split() if word not in ENGLISH_STOP_WORDS and word in vectorizer.vocabulary_]
    keywords = ', '.join(keywords[:5]) or "No key words found"

    return render_template("index.html", prediction=result, confidence=confidence, review=input_text,
                           comment=comment, suggestion=suggestion, keywords=keywords)

if __name__ == '__main__':
    app.run(debug=True)

