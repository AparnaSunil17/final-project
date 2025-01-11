#index.html calling
from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import re
import os

# Load Models
bert_model = BertForSequenceClassification.from_pretrained("bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")
abet_model = joblib.load("abet_model.pkl")
abet_vectorizer = joblib.load("abet_vectorizer.pkl")

# Flask App
app = Flask(__name__, static_folder=".")

# Smishing Detection Logic
def detect_smishing(sms):
    def extract_text_and_url(sms):
        url_pattern = r'(https?://[^\s]+)'
        urls = re.findall(url_pattern, sms)
        text = re.sub(url_pattern, '', sms).strip()
        return text, urls[0] if urls else None

    text, url = extract_text_and_url(sms)
    
    # BERT Text Analysis
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    text_probs = torch.softmax(outputs.logits, dim=1)
    text_result = torch.argmax(text_probs).item()

    # ABET URL Analysis
    if url:
        url_features = abet_vectorizer.transform([url])
        url_result = abet_model.predict(url_features)[0]
    else:
        url_result = 0

    # Determine if it's phishing
    is_phishing = text_result == 1 or url_result == 1
    return {
        "is_phishing": is_phishing,
        "text_result": text_result,
        "url_result": url_result
    }

# Serve Front-End HTML
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# Backend API for Smishing Detection
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    sms = data.get("sms", "")
    result = detect_smishing(sms)
    return jsonify(result)

if __name__ == "__main__":
    # Ensure the app runs on a publicly accessible IP during development
    app.run(host="0.0.0.0", port=5000)
