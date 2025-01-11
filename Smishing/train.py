import re
import torch
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
import joblib  # For saving models

# Train BERT Model
def train_bert_model(train_texts, train_labels, val_texts, val_labels):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    return model, tokenizer

# Train ABET Model
def extract_url_features(url):
    parsed_url = urlparse(url)
    features = [
        len(url),
        len(parsed_url.hostname) if parsed_url.hostname else 0,
        len(parsed_url.netloc.split(".")) - 1,
        1 if any(tld in parsed_url.path or tld in parsed_url.netloc for tld in [".com", ".org", ".net"]) else 0,
        1 if re.match(r'\b\d{1,3}(\.\d{1,3}){3}\b', parsed_url.netloc) else 0,
        sum(1 for char in url if char in "!@#$%^&*()-_+="),
        len(re.findall(r'(https?://)', url)) - 1,
        1 if '-' in parsed_url.netloc else 0,
        1 if len(url) < 20 else 0,
    ]
    return np.array(features)

def combine_features(urls, vectorizer=None, fit=False):
    structural_features = np.array([extract_url_features(url) for url in urls])
    tfidf_features = vectorizer.fit_transform(urls) if fit else vectorizer.transform(urls)
    return np.hstack([tfidf_features.toarray(), structural_features])

def train_abet_model(urls, labels):
    vectorizer = TfidfVectorizer(max_features=100)
    combined_features = combine_features(urls, vectorizer, fit=True)
    base_estimator = ExtraTreesClassifier(n_estimators=50, random_state=42)
    model = AdaBoostClassifier( n_estimators=50, random_state=42)
    model.fit(combined_features, labels)
    return model, vectorizer

# Train BERT and ABET
dataset = pd.read_csv("/content/balanced_dataset.csv")  # Replace with your dataset
urls = dataset["url"].tolist()
labels = dataset["Label"].tolist()

train_texts = ["This is safe", "Urgent! Click this phishing link"]
train_labels = [0, 1]
val_texts = ["Account update needed, click here"]
val_labels = [1]

bert_model, bert_tokenizer = train_bert_model(train_texts, train_labels, val_texts, val_labels)
abet_model, abet_vectorizer = train_abet_model(urls, labels)

# Save Models
bert_model.save_pretrained("bert_model")
bert_tokenizer.save_pretrained("bert_tokenizer")
joblib.dump(abet_model, "abet_model.pkl")
joblib.dump(abet_vectorizer, "abet_vectorizer.pkl")
