import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch

# URL feature extraction
def extract_url_features(urls):
    """Extract features from URLs."""
    features = []
    for url in urls:
        features.append([
            len(url),
            url.count("."),
            url.count("/"),
            url.startswith("https"),
            "?" in url,
        ])
    return np.array(features)

# Preprocess CSV
def preprocess_dataset(csv_file):
    """Load and preprocess dataset."""
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['text', 'label'])
    df['url'] = df['text'].apply(lambda x: re.findall(r'https?://[^\s]+', x)[0] if re.findall(r'https?://[^\s]+', x) else None)
    return df

# Train ABET model
def train_abet_model(features, labels):
    # Vectorize the features (URLs)
    vectorizer = TfidfVectorizer()
    features_vectorized = vectorizer.fit_transform(features)

    # Train a Logistic Regression model
    abet = LogisticRegression()
    abet.fit(features_vectorized, labels)
    return abet, vectorizer

# Train BERT model
def train_bert_model(train_texts, train_labels, val_texts, val_labels):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
    val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        save_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return model, tokenizer

# Interactive prediction using both ABET and BERT models
def predict_sms(message, abet_model, bert_model, bert_tokenizer, vectorizer):
    # Check if URL is present in the message
    url = re.findall(r'https?://[^\s]+', message)
    if url:
        # Extract URL features for ABET
        url_features = extract_url_features(url)
        url_prediction = abet_model.predict(url_features)

        # Use BERT for text classification
        encodings = bert_tokenizer(message, truncation=True, padding=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            output = bert_model(**encodings)
            bert_prediction = torch.argmax(output.logits, dim=-1).item()

        # Combine both model predictions
        final_prediction = (url_prediction[0] + bert_prediction) / 2  # Average the predictions for final decision
        prediction_label = "phishing" if final_prediction >= 0.5 else "ham"

        return prediction_label, url_prediction[0], bert_prediction
    else:
        # Use ABET model if no URL is found
        sms_features = [message]  # No URL, so only use message features
        sms_features_vectorized = vectorizer.transform(sms_features)
        sms_prediction = abet_model.predict(sms_features_vectorized)
        prediction_label = "phishing" if sms_prediction[0] == 1 else "ham"
        return prediction_label, sms_prediction[0], None

# Training pipeline
def train_combined_model(csv_file):
    # Load and preprocess the dataset
    df = pd.read_csv(csv_file)

    # Ensure proper handling of missing or invalid data
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)

    # Split the data for training and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # Train ABET model
    abet_model, vectorizer = train_abet_model(df['url'], df['label'])

    # Train BERT model
    bert_model, bert_tokenizer = train_bert_model(train_texts.tolist(), train_labels.tolist(), val_texts.tolist(), val_labels.tolist())

    return abet_model, bert_model, bert_tokenizer, vectorizer


# Main block
if __name__ == "__main__":
    # Train the model
    csv_file = "url_data.csv"  # Replace with your dataset path
    print("Training models...")
    abet_model, bert_model, bert_tokenizer, vectorizer = train_combined_model(csv_file)
    print("Training completed.")

    # Interactive prediction
    while True:
        sms = input("Enter a message to check for phishing (or type 'exit' to quit): ")
        if sms.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        result, abet_pred, bert_pred = predict_sms(sms, abet_model, bert_model, bert_tokenizer, vectorizer)
        print(f"Prediction: {result}, ABET Prediction: {abet_pred}, BERT Prediction: {bert_pred}")
