# ============================================================
# PRE-TRAIN STOCK SENTIMENT MODELS - TERMINAL USER INTERFACE
# ============================================================
# Models:
# 1. SVM + TF-IDF        -> scikit-learn
# 2. BiLSTM              -> PyTorch
# 3. BERT                -> PyTorch + Hugging Face
#
# TensorFlow/Keras is NOT used in this version.
#
# Workflow:
# 1. Load balanced dataset
# 2. Create and SAVE one fixed 80% train / 20% test split
# 3. Train one selected model in Terminal
# 4. Test on the SAME saved 20% test set
# 5. Save model + predictions + metrics for Streamlit GUI
# ============================================================

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

DATASET_PATH = Path("stock_sentiment_balanced_40_30_30_unique.csv")
SPLIT_DIR = Path("data_splits")
MODEL_DIR = Path("saved_models")
RESULT_DIR = Path("results")

TRAIN_PATH = SPLIT_DIR / "train.csv"
TEST_PATH = SPLIT_DIR / "test.csv"
SPLIT_META_PATH = SPLIT_DIR / "split_metadata.json"

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10
RANDOM_STATE = 42
CLASS_LABELS = ["negative", "neutral", "positive"]

# BiLSTM settings
MAX_VOCAB = 20000
MAX_LENGTH = 100
EMBED_DIM = 128
HIDDEN_DIM = 64
BILSTM_BATCH_SIZE = 64
BILSTM_MAX_EPOCHS = 10
BILSTM_PATIENCE = 2
BILSTM_MIN_DELTA = 0.001

# BERT settings
BERT_MAX_EPOCHS = 5
BERT_TRAIN_BATCH_SIZE = 8 
BERT_EVAL_BATCH_SIZE = 8 

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def ensure_directories():
    SPLIT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

def dataset_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            sha.update(block)
    return sha.hexdigest()

def load_clean_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Put the CSV in the same folder as pretrain_models.py."
        )

    df = pd.read_csv(DATASET_PATH)

    if not {"Sentence", "Sentiment"}.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'Sentence' and 'Sentiment' columns."
        )

    df = df.dropna(subset=["Sentence", "Sentiment"]).copy()
    df["Sentence"] = df["Sentence"].astype(str).str.strip()
    df["Sentiment"] = df["Sentiment"].astype(str).str.lower().str.strip()
    df = df[df["Sentiment"].isin(CLASS_LABELS)]
    df = df.drop_duplicates(subset=["Sentence"]).reset_index(drop=True)

    return df

def print_distribution(title, df):
    counts = df["Sentiment"].value_counts().reindex(CLASS_LABELS, fill_value=0)
    pct = counts / len(df) * 100

    print(f"\n{title}")
    print("-" * 45)
    for label in CLASS_LABELS:
        print(f"{label.capitalize():8s}: {counts[label]:6,d} ({pct[label]:6.2f}%)")
    print(f"{'Total':8s}: {len(df):6,d}")
    print("-" * 45)

def prepare_fixed_split(force=False):
    ensure_directories()
    current_hash = dataset_sha256(DATASET_PATH)

    if (
        not force
        and TRAIN_PATH.exists()
        and TEST_PATH.exists()
        and SPLIT_META_PATH.exists()
    ):
        with open(SPLIT_META_PATH, "r", encoding="utf-8") as file:
            meta = json.load(file)

        if meta.get("dataset_sha256") == current_hash:
            print("\nReusing existing fixed 80/20 split.")
            return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)

    df = load_clean_dataset()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Sentiment"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # SAVE the split BEFORE any model training.
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    with open(SPLIT_META_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_sha256": current_hash,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "train_rows": len(train_df),
                "test_rows": len(test_df)
            },
            file,
            indent=2
        )

    print_distribution("SAVED TRAINING DATASET (80%)", train_df)
    print_distribution("SAVED TESTING DATASET (20%)", test_df)

    return train_df, test_df

def load_fixed_split():
    return prepare_fixed_split(force=False)

def calculate_metrics(y_true, y_pred, training_time):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "F1-Score": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Training Time (s)": float(training_time)
    }

def save_common_results(model_key, test_df, predictions, metrics, extra=None):
    result_df = test_df.copy()
    result_df["Predicted"] = predictions
    result_df["Correct"] = (
        result_df["Sentiment"].to_numpy() == np.asarray(predictions)
    )

    result_df.to_csv(
        RESULT_DIR / f"{model_key}_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    report = classification_report(
        test_df["Sentiment"],
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULT_DIR / f"{model_key}_classification_report.csv",
        encoding="utf-8-sig"
    )

    payload = {
        "model": model_key,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        **metrics
    }

    if extra:
        payload.update(extra)

    with open(
        RESULT_DIR / f"{model_key}_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(payload, file, indent=2)

# ============================================================
# MODEL 1: SVM + TF-IDF
# ============================================================

def train_svm():
    train_df, test_df = load_fixed_split()

    print("\n" + "=" * 64)
    print("PRE-TRAIN MODEL 1: SVM + TF-IDF")
    print("=" * 64)

    start = time.time()

    print("Step 1/4: Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.90,
        min_df=3,
        stop_words="english",
        max_features=30000,
        lowercase=True
    )

    X_train = vectorizer.fit_transform(train_df["Sentence"])
    X_test = vectorizer.transform(test_df["Sentence"])

    print("Step 2/4: Training LinearSVC...")
    model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, train_df["Sentiment"])

    print("Step 3/4: Testing on saved fixed 20% test dataset...")
    predictions = model.predict(X_test)

    elapsed = time.time() - start
    metrics = calculate_metrics(
        test_df["Sentiment"],
        predictions,
        elapsed
    )

    print("Step 4/4: Saving model and results...")
    joblib.dump(model, MODEL_DIR / "svm_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "svm_vectorizer.pkl")

    save_common_results(
        "svm",
        test_df,
        predictions,
        metrics,
        {"Early Stopping": "Not applicable"}
    )

    print(f"\nAccuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-Score : {metrics['F1-Score']:.4f}")
    print(f"Time     : {elapsed:.2f} seconds")
    print("SVM saved successfully.")

# ============================================================
# MODEL 2: PYTORCH BiLSTM
# ============================================================

def tokenize_text(text):
    """Simple lowercase word tokenizer for the PyTorch BiLSTM."""
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())

def build_vocab(sentences):
    """Build vocabulary from TRAINING text only."""
    counter = Counter()

    for text in sentences:
        counter.update(tokenize_text(text))

    most_common = counter.most_common(MAX_VOCAB - 2)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, _ in most_common:
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab

def text_to_ids(text, vocab):
    """Convert one sentence to padded/truncated integer IDs."""
    tokens = tokenize_text(text)
    ids = [vocab.get(token, 1) for token in tokens[:MAX_LENGTH]]

    if len(ids) < MAX_LENGTH:
        ids.extend([0] * (MAX_LENGTH - len(ids)))

    return ids

class TextDataset(Dataset):
    def __init__(self, sentences, labels, vocab):
        self.features = torch.tensor(
            [text_to_ids(text, vocab) for text in sentences],
            dtype=torch.long
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)

        _, (hidden, _) = self.lstm(embedded)

        # hidden[-2] = final forward state
        # hidden[-1] = final backward state
        combined = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        x = self.dropout(combined)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def evaluate_bilstm(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return total_loss / total, correct / total

def train_bilstm():
    train_df, test_df = load_fixed_split()

    print("\n" + "=" * 64)
    print("PRE-TRAIN MODEL 2: PyTorch BiLSTM")
    print("=" * 64)

    train_inner, val_df = train_test_split(
        train_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["Sentiment"]
    )

    train_inner = train_inner.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    encoder = LabelEncoder()
    encoder.fit(train_df["Sentiment"])

    y_train = encoder.transform(train_inner["Sentiment"])
    y_val = encoder.transform(val_df["Sentiment"])
    y_test = encoder.transform(test_df["Sentiment"])

    print("Step 1/6: Building vocabulary from training text only...")
    vocab = build_vocab(train_inner["Sentence"])

    train_dataset = TextDataset(
        train_inner["Sentence"].tolist(),
        y_train,
        vocab
    )

    val_dataset = TextDataset(
        val_df["Sentence"].tolist(),
        y_val,
        vocab
    )

    test_dataset = TextDataset(
        test_df["Sentence"].tolist(),
        y_test,
        vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BILSTM_BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BILSTM_BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BILSTM_BATCH_SIZE,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Vocabulary size: {len(vocab):,}")

    print("Step 2/6: Building PyTorch BiLSTM...")
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        num_classes=len(encoder.classes_)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    history = []

    print("Step 3/6: Training with early stopping...")
    start = time.time()

    for epoch in range(1, BILSTM_MAX_EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total = 0
        correct = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_bilstm(
            model,
            val_loader,
            criterion,
            device
        )

        history.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }
        )

        print(
            f"Epoch {epoch:02d}/{BILSTM_MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Early stopping based on validation loss.
        if best_val_loss - val_loss > BILSTM_MIN_DELTA:
            best_val_loss = val_loss
            patience_count = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_count += 1

            print(
                f"No sufficient validation-loss improvement "
                f"({patience_count}/{BILSTM_PATIENCE})"
            )

            if patience_count >= BILSTM_PATIENCE:
                print("Early stopping activated.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Step 4/6: Testing on saved fixed 20% test dataset...")

    model.eval()
    prediction_ids = []

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)

            logits = model(features)
            batch_predictions = torch.argmax(
                logits,
                dim=1
            )

            prediction_ids.extend(
                batch_predictions.cpu().numpy().tolist()
            )

    predictions = encoder.inverse_transform(
        np.array(prediction_ids)
    )

    elapsed = time.time() - start

    metrics = calculate_metrics(
        test_df["Sentiment"],
        predictions,
        elapsed
    )

    print("Step 5/6: Saving PyTorch BiLSTM...")

    torch.save(
        {
            "model_state_dict": {
                key: value.cpu()
                for key, value in model.state_dict().items()
            },
            "vocab_size": len(vocab),
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": len(encoder.classes_),
            "max_length": MAX_LENGTH
        },
        MODEL_DIR / "bilstm_model.pt"
    )

    with open(
        MODEL_DIR / "bilstm_vocab.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(vocab, file, ensure_ascii=False)

    with open(
        MODEL_DIR / "bilstm_classes.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(encoder.classes_.tolist(), file)

    pd.DataFrame(history).to_csv(
        RESULT_DIR / "bilstm_history.csv",
        index=False,
        encoding="utf-8-sig"
    )

    epochs_completed = len(history)

    print("Step 6/6: Saving metrics and predictions...")

    save_common_results(
        "bilstm",
        test_df,
        predictions,
        metrics,
        {
            "Framework": "PyTorch",
            "Early Stopping": (
                "Activated"
                if epochs_completed < BILSTM_MAX_EPOCHS
                else "Maximum epochs reached"
            ),
            "Completed Epochs": epochs_completed,
            "Maximum Epochs": BILSTM_MAX_EPOCHS
        }
    )

    print(f"\nAccuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-Score : {metrics['F1-Score']:.4f}")
    print(f"Epochs   : {epochs_completed}/{BILSTM_MAX_EPOCHS}")
    print(f"Time     : {elapsed:.2f} seconds")
    print("PyTorch BiLSTM saved successfully.")

# ============================================================
# MODEL 3: BERT
# ============================================================

def train_bert():
    from transformers import (
        BertForSequenceClassification,
        BertTokenizerFast,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments
    )

    class BertDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, index):
            item = {
                key: torch.tensor(value[index])
                for key, value in self.encodings.items()
            }
            item["labels"] = torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
            return item

        def __len__(self):
            return len(self.labels)

    def compute_bert_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            )
        }

    def make_training_arguments():
        common = dict(
            output_dir="./bert_training_output",
            num_train_epochs=BERT_MAX_EPOCHS,
            per_device_train_batch_size=BERT_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=BERT_EVAL_BATCH_SIZE,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_strategy="epoch",
            report_to="none",
            fp16=torch.cuda.is_available(),
            seed=RANDOM_STATE
        )

        try:
            return TrainingArguments(
                eval_strategy="epoch",
                **common
            )
        except TypeError:
            return TrainingArguments(
                evaluation_strategy="epoch",
                **common
            )

    train_df, test_df = load_fixed_split()

    print("\n" + "=" * 64)
    print("PRE-TRAIN MODEL 3: BERT")
    print("=" * 64)

    train_inner, val_df = train_test_split(
        train_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["Sentiment"]
    )

    encoder = LabelEncoder()
    encoder.fit(train_df["Sentiment"])

    y_train = encoder.transform(train_inner["Sentiment"])
    y_val = encoder.transform(val_df["Sentiment"])
    y_test = encoder.transform(test_df["Sentiment"])

    print(
        "Device:",
        "CUDA GPU" if torch.cuda.is_available() else "CPU"
    )

    print("Step 1/5: Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased"
    )

    print("Step 2/5: Tokenizing data...")
    train_encodings = tokenizer(
        train_inner["Sentence"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=128
    )

    val_encodings = tokenizer(
        val_df["Sentence"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=128
    )

    test_encodings = tokenizer(
        test_df["Sentence"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=128
    )

    train_dataset = BertDataset(
        train_encodings,
        y_train.tolist()
    )

    val_dataset = BertDataset(
        val_encodings,
        y_val.tolist()
    )

    test_dataset = BertDataset(
        test_encodings,
        y_test.tolist()
    )

    print("Step 3/5: Loading pretrained BERT...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    )

    trainer = Trainer(
        model=model,
        args=make_training_arguments(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_bert_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001
            )
        ]
    )

    print("Step 4/5: Fine-tuning BERT...")
    start = time.time()

    trainer.train()

    output = trainer.predict(test_dataset)
    prediction_ids = np.argmax(
        output.predictions,
        axis=1
    )

    predictions = encoder.inverse_transform(
        prediction_ids
    )

    elapsed = time.time() - start

    metrics = calculate_metrics(
        test_df["Sentiment"],
        predictions,
        elapsed
    )

    print("Step 5/5: Saving BERT and results...")

    bert_dir = MODEL_DIR / "bert_model"
    bert_dir.mkdir(exist_ok=True)

    model.save_pretrained(
        bert_dir,
        max_shard_size="90MB"
    )

    tokenizer.save_pretrained(
        bert_dir
    )

    with open(
        bert_dir / "label_classes.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            encoder.classes_.tolist(),
            file
        )

    with open(
        RESULT_DIR / "bert_history.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            trainer.state.log_history,
            file,
            indent=2
        )

    save_common_results(
        "bert",
        test_df,
        predictions,
        metrics,
        {
            "Framework": "PyTorch + Transformers",
            "Early Stopping": "Enabled",
            "Completed Epoch": trainer.state.epoch,
            "Maximum Epochs": BERT_MAX_EPOCHS,
            "Best Validation Macro F1": (
                float(trainer.state.best_metric)
                if trainer.state.best_metric is not None
                else None
            )
        }
    )

    print(f"\nAccuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-Score : {metrics['F1-Score']:.4f}")
    print(f"Time     : {elapsed:.2f} seconds")
    print("BERT saved successfully.")

# ============================================================
# TUI MENU
# ============================================================

def show_menu():
    print("\n" + "=" * 64)
    print("STOCK SENTIMENT PRE-TRAINING - TUI")
    print("=" * 64)
    print("1. Prepare / verify fixed 80-20 split")
    print("2. Pre-train SVM + TF-IDF")
    print("3. Pre-train PyTorch BiLSTM")
    print("4. Pre-train BERT")
    print("5. Pre-train ALL models")
    print("0. Exit")
    print("=" * 64)

    while True:
        choice = input("Choose an option: ").strip()

        if choice == "1":
            prepare_fixed_split()
        elif choice == "2":
            train_svm()
        elif choice == "3":
            train_bilstm()
        elif choice == "4":
            train_bert()
        elif choice == "5":
            train_svm()
            train_bilstm()
            train_bert()
        elif choice == "0":
            print("Exit.")
            break
        else:
            print("Invalid option. Choose 0-5.")

def main():
    ensure_directories()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        nargs="?",
        choices=["prepare", "svm", "bilstm", "bert", "all"]
    )

    args = parser.parse_args()

    if args.action == "prepare":
        prepare_fixed_split()
    elif args.action == "svm":
        train_svm()
    elif args.action == "bilstm":
        train_bilstm()
    elif args.action == "bert":
        train_bert()
    elif args.action == "all":
        train_svm()
        train_bilstm()
        train_bert()
    else:
        show_menu()

if __name__ == "__main__":
    main()
