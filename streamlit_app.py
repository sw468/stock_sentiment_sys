"""
Stock Sentiment Analysis Dashboard

Run this application from PowerShell with:

    python -m streamlit run streamlit_app.py
"""
# ============================================================
# STOCK SENTIMENT ANALYSIS DASHBOARD
# ============================================================
# Train first in Terminal:
#   python pretrain_models.py svm
#   python pretrain_models.py bilstm
#   python pretrain_models.py bert
#   python pretrain_models.py all

# Frameworks:
# SVM     -> scikit-learn
# BiLSTM  -> PyTorch
# BERT    -> PyTorch + Hugging Face
#
# TensorFlow/Keras is NOT required.
# ============================================================

import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix
)

MODEL_DIR = Path("saved_models")
RESULT_DIR = Path("results")

CLASS_LABELS = [
    "negative",
    "neutral",
    "positive"
]

MAX_LENGTH = 100

st.set_page_config(
    page_title="Stock Sentiment Results",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Sentiment Analysis")
st.caption(
    "Fast GUI using models pre-trained in the Terminal."
)

with st.sidebar:
    st.header("Choose Model")
    selected_model = st.radio(
        "Select a pre-trained model:",
        [
            "SVM + TF-IDF",
            "BiLSTM",
            "BERT"
        ]
    )

MODEL_KEYS = {
    "SVM + TF-IDF": "svm",
    "BiLSTM": "bilstm",
    "BERT": "bert"
}

model_key = MODEL_KEYS[selected_model]

def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def model_is_ready(key):
    if key == "svm":
        return (
            (MODEL_DIR / "svm_model.pkl").exists()
            and (MODEL_DIR / "svm_vectorizer.pkl").exists()
            and (RESULT_DIR / "svm_metrics.json").exists()
            and (RESULT_DIR / "svm_predictions.csv").exists()
        )

    if key == "bilstm":
        return (
            (MODEL_DIR / "bilstm_model.pt").exists()
            and (MODEL_DIR / "bilstm_vocab.json").exists()
            and (MODEL_DIR / "bilstm_classes.json").exists()
            and (RESULT_DIR / "bilstm_metrics.json").exists()
            and (RESULT_DIR / "bilstm_predictions.csv").exists()
        )

    return (
        (MODEL_DIR / "bert_model").exists()
        and (MODEL_DIR / "bert_model" / "label_classes.json").exists()
        and (RESULT_DIR / "bert_metrics.json").exists()
        and (RESULT_DIR / "bert_predictions.csv").exists()
    )

# ============================================================
# SVM LOADER
# ============================================================

@st.cache_resource
def load_svm():
    model = joblib.load(
        MODEL_DIR / "svm_model.pkl"
    )

    vectorizer = joblib.load(
        MODEL_DIR / "svm_vectorizer.pkl"
    )

    return model, vectorizer

# ============================================================
# PYTORCH BiLSTM
# ============================================================

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(
            hidden_dim * 2,
            64
        )

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):
        embedded = self.embedding(x)

        _, (hidden, _) = self.lstm(embedded)

        combined = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        x = self.dropout(combined)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)

def tokenize_text(text):
    return re.findall(
        r"[A-Za-z0-9']+",
        str(text).lower()
    )

def text_to_ids(text, vocab, max_length=MAX_LENGTH):
    tokens = tokenize_text(text)

    ids = [
        vocab.get(token, 1)
        for token in tokens[:max_length]
    ]

    if len(ids) < max_length:
        ids.extend(
            [0] * (max_length - len(ids))
        )

    return ids

@st.cache_resource
def load_bilstm():
    checkpoint = torch.load(
        MODEL_DIR / "bilstm_model.pt",
        map_location="cpu"
    )

    vocab = read_json(
        MODEL_DIR / "bilstm_vocab.json"
    )

    classes = np.array(
        read_json(
            MODEL_DIR / "bilstm_classes.json"
        )
    )

    model = BiLSTMClassifier(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return (
        model,
        vocab,
        classes,
        checkpoint["max_length"],
        device
    )

# ============================================================
# BERT LOADER
# ============================================================

@st.cache_resource
def load_bert():
    from transformers import (
        BertForSequenceClassification,
        BertTokenizerFast
    )

    bert_dir = MODEL_DIR / "bert_model"

    tokenizer = BertTokenizerFast.from_pretrained(
        bert_dir
    )

    model = BertForSequenceClassification.from_pretrained(
        bert_dir
    )

    classes = np.array(
        read_json(
            bert_dir / "label_classes.json"
        )
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return model, tokenizer, classes, device

# ============================================================
# CHECK SAVED MODEL
# ============================================================

if not model_is_ready(model_key):
    st.error(
        f"{selected_model} has not been pre-trained yet."
    )

    st.code(
        f"python pretrain_models.py {model_key}",
        language="powershell"
    )

    st.stop()

# ============================================================
# DISPLAY SAVED RESULTS
# ============================================================

metrics = read_json(
    RESULT_DIR / f"{model_key}_metrics.json"
)

predictions_df = pd.read_csv(
    RESULT_DIR / f"{model_key}_predictions.csv"
)

st.header(
    f"{selected_model} Results"
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Accuracy",
    f"{metrics['Accuracy']:.4f}"
)

m2.metric(
    "Precision",
    f"{metrics['Precision']:.4f}"
)

m3.metric(
    "Recall",
    f"{metrics['Recall']:.4f}"
)

m4.metric(
    "F1-Score",
    f"{metrics['F1-Score']:.4f}"
)

m5.metric(
    "Training Time",
    f"{metrics['Training Time (s)']:.2f}s"
)

st.caption(
    f"Pre-trained: {metrics.get('trained_at', 'Unknown')}"
)

# ============================================================
# DIRECT TRAINING HISTORY + VALIDATION LOSS DISPLAY
# ============================================================
# BiLSTM and BERT are trained in the Terminal first.
# Their saved history files are loaded here and displayed
# immediately in Streamlit without retraining the model.

st.divider()
st.header("Training History and Validation Loss")

if model_key == "bilstm":
    history_path = RESULT_DIR / "bilstm_history.csv"

    if history_path.exists():
        history = pd.read_csv(history_path)

        if "epoch" in history.columns:
            history = history.set_index("epoch")

        graph1, graph2 = st.columns(2)

        with graph1:
            st.subheader("Training vs Validation Accuracy")

            accuracy_columns = [
                col
                for col in ["accuracy", "val_accuracy"]
                if col in history.columns
            ]

            if accuracy_columns:
                accuracy_df = history[accuracy_columns].copy()
                accuracy_df = accuracy_df.rename(
                    columns={
                        "accuracy": "Training Accuracy",
                        "val_accuracy": "Validation Accuracy"
                    }
                )
                st.line_chart(accuracy_df)
            else:
                st.warning("Accuracy history is not available.")

        with graph2:
            st.subheader("Training vs Validation Loss")

            loss_columns = [
                col
                for col in ["loss", "val_loss"]
                if col in history.columns
            ]

            if loss_columns:
                loss_df = history[loss_columns].copy()
                loss_df = loss_df.rename(
                    columns={
                        "loss": "Training Loss",
                        "val_loss": "Validation Loss"
                    }
                )
                st.line_chart(loss_df)

                if "val_loss" in history.columns:
                    best_epoch = history["val_loss"].idxmin()
                    best_val_loss = history["val_loss"].min()

                    st.metric(
                        "Best Validation Loss",
                        f"{best_val_loss:.4f}"
                    )
                    st.caption(
                        f"Lowest validation loss occurred at epoch {best_epoch}."
                    )
            else:
                st.warning("Loss history is not available.")

        with st.expander("Show complete BiLSTM training history"):
            st.dataframe(
                history.round(4),
                use_container_width=True
            )

    else:
        st.warning(
            "bilstm_history.csv was not found. "
            "Pre-train BiLSTM again to generate the training history."
        )

elif model_key == "bert":
    history_path = RESULT_DIR / "bert_history.json"

    if history_path.exists():
        bert_history = pd.DataFrame(
            read_json(history_path)
        )

        graph1, graph2 = st.columns(2)

        with graph1:
            st.subheader("BERT Training Loss")

            train_rows = bert_history[
                bert_history.get(
                    "loss",
                    pd.Series(
                        index=bert_history.index,
                        dtype=float
                    )
                ).notna()
            ].copy()

            if not train_rows.empty:
                if "epoch" in train_rows.columns:
                    train_chart = train_rows[
                        ["epoch", "loss"]
                    ].dropna().set_index("epoch")
                else:
                    train_chart = train_rows[
                        ["loss"]
                    ].copy()

                train_chart = train_chart.rename(
                    columns={"loss": "Training Loss"}
                )

                st.line_chart(train_chart)
            else:
                st.warning("BERT training loss is not available.")

        with graph2:
            st.subheader("BERT Validation Loss")

            eval_rows = bert_history[
                bert_history.get(
                    "eval_loss",
                    pd.Series(
                        index=bert_history.index,
                        dtype=float
                    )
                ).notna()
            ].copy()

            if not eval_rows.empty:
                if "epoch" in eval_rows.columns:
                    val_chart = eval_rows[
                        ["epoch", "eval_loss"]
                    ].dropna().set_index("epoch")
                else:
                    val_chart = eval_rows[
                        ["eval_loss"]
                    ].copy()

                val_chart = val_chart.rename(
                    columns={"eval_loss": "Validation Loss"}
                )

                st.line_chart(val_chart)

                best_row = eval_rows.loc[
                    eval_rows["eval_loss"].idxmin()
                ]

                st.metric(
                    "Best Validation Loss",
                    f"{best_row['eval_loss']:.4f}"
                )

                if "epoch" in best_row:
                    st.caption(
                        f"Lowest validation loss occurred at epoch "
                        f"{best_row['epoch']:.2f}."
                    )
            else:
                st.warning("BERT validation loss is not available.")

        metric_columns = [
            col
            for col in ["epoch", "eval_accuracy", "eval_f1"]
            if col in bert_history.columns
        ]

        if len(metric_columns) > 1:
            metric_rows = bert_history[
                bert_history.get(
                    "eval_f1",
                    pd.Series(
                        index=bert_history.index,
                        dtype=float
                    )
                ).notna()
            ].copy()

            if not metric_rows.empty:
                st.subheader("BERT Validation Performance")

                chart_columns = [
                    col
                    for col in ["eval_accuracy", "eval_f1"]
                    if col in metric_rows.columns
                ]

                if "epoch" in metric_rows.columns:
                    metric_chart = metric_rows[
                        ["epoch"] + chart_columns
                    ].dropna(
                        subset=chart_columns,
                        how="all"
                    ).set_index("epoch")
                else:
                    metric_chart = metric_rows[
                        chart_columns
                    ].copy()

                metric_chart = metric_chart.rename(
                    columns={
                        "eval_accuracy": "Validation Accuracy",
                        "eval_f1": "Validation Macro F1"
                    }
                )

                st.line_chart(metric_chart)

        # ----------------------------------------------------
        # CLEAN BERT HISTORY TABLES
        # Hugging Face stores training and evaluation logs in
        # separate rows. We separate them so Streamlit does not
        # display many None values.
        # ----------------------------------------------------

        with st.expander("Show clean BERT training history"):
            st.subheader("BERT Training History")

            if "loss" in bert_history.columns:
                train_history = bert_history[
                    bert_history["loss"].notna()
                ].copy()

                train_columns = [
                    col
                    for col in [
                        "epoch",
                        "loss",
                        "grad_norm",
                        "learning_rate"
                    ]
                    if col in train_history.columns
                ]

                train_history = train_history[
                    train_columns
                ].rename(
                    columns={
                        "epoch": "Epoch",
                        "loss": "Training Loss",
                        "grad_norm": "Gradient Norm",
                        "learning_rate": "Learning Rate"
                    }
                )

                if not train_history.empty:
                    st.dataframe(
                        train_history.round(4),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No training-loss rows were found.")
            else:
                st.info("Training-loss history is not available.")

            st.subheader("BERT Validation History")

            if "eval_loss" in bert_history.columns:
                validation_history = bert_history[
                    bert_history["eval_loss"].notna()
                ].copy()

                validation_columns = [
                    col
                    for col in [
                        "epoch",
                        "eval_loss",
                        "eval_accuracy",
                        "eval_f1"
                    ]
                    if col in validation_history.columns
                ]

                validation_history = validation_history[
                    validation_columns
                ].rename(
                    columns={
                        "epoch": "Epoch",
                        "eval_loss": "Validation Loss",
                        "eval_accuracy": "Validation Accuracy",
                        "eval_f1": "Validation F1"
                    }
                )

                if not validation_history.empty:
                    st.dataframe(
                        validation_history.round(4),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No validation rows were found.")
            else:
                st.info("Validation history is not available.")

    else:
        st.warning(
            "bert_history.json was not found. "
            "Pre-train BERT again to generate the training history."
        )

else:
    st.info(
        "SVM does not train by epochs, so validation-loss and "
        "epoch-history graphs do not apply to LinearSVC."
    )

st.divider()

report_tab, matrix_tab, training_tab = st.tabs(
    [
        "Classification Report",
        "Confusion Matrix",
        "Training Information"
    ]
)

with report_tab:
    report = pd.read_csv(
        RESULT_DIR
        / f"{model_key}_classification_report.csv",
        index_col=0
    )

    # --------------------------------------------------------
    # REMOVE MISLEADING ACCURACY ROW
    # sklearn stores accuracy as one scalar value. When it is
    # converted to a DataFrame, pandas repeats the same value
    # under precision, recall, f1-score and support.
    # Accuracy is already shown correctly in the metric card.
    # --------------------------------------------------------

    report = report.drop(
        index="accuracy",
        errors="ignore"
    )

    st.dataframe(
        report.round(4),
        use_container_width=True
    )

with matrix_tab:
    cm = confusion_matrix(
        predictions_df["Sentiment"],
        predictions_df["Predicted"],
        labels=CLASS_LABELS
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_LABELS
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"{selected_model} Confusion Matrix"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with training_tab:
    st.write(
        f"Framework: **{metrics.get('Framework', 'scikit-learn')}**"
    )

    st.write(
        f"Early stopping: **{metrics.get('Early Stopping', 'N/A')}**"
    )

    if model_key == "bilstm":
        st.write(
            f"Completed epochs: "
            f"**{metrics.get('Completed Epochs')} / "
            f"{metrics.get('Maximum Epochs')}**"
        )

        history_path = RESULT_DIR / "bilstm_history.csv"

        if history_path.exists():
            history = pd.read_csv(history_path)

            accuracy_history = history[
                ["accuracy", "val_accuracy"]
            ].copy()

            accuracy_history.columns = [
                "Training Accuracy",
                "Validation Accuracy"
            ]

            st.write(
                "Training and Validation Accuracy"
            )

            st.line_chart(
                accuracy_history
            )

            loss_history = history[
                ["loss", "val_loss"]
            ].copy()

            loss_history.columns = [
                "Training Loss",
                "Validation Loss"
            ]

            st.write(
                "Training and Validation Loss"
            )

            st.line_chart(
                loss_history
            )

    elif model_key == "bert":
        st.write(
            f"Completed epoch: "
            f"**{metrics.get('Completed Epoch')} / "
            f"{metrics.get('Maximum Epochs')}**"
        )

        best_f1 = metrics.get(
            "Best Validation Macro F1"
        )

        if best_f1 is not None:
            st.write(
                f"Best validation Macro F1: **{best_f1:.4f}**"
            )

# ============================================================
# TEST YOUR OWN SENTENCE
# ============================================================

st.divider()
st.header("Test Your Own Sentence")

sentence = st.text_area(
    "Enter a stock/news sentence:",
    placeholder=(
        "Example: The company reported strong revenue growth "
        "and raised its annual earnings forecast."
    ),
    height=120
)

if st.button(
    f"Predict with {selected_model}",
    type="primary",
    use_container_width=True
):
    sentence = sentence.strip()

    if not sentence:
        st.warning(
            "Please enter a sentence."
        )

    elif model_key == "svm":
        model, vectorizer = load_svm()

        x = vectorizer.transform(
            [sentence]
        )

        prediction = model.predict(x)[0]

        scores = model.decision_function(x)[0]

        predicted_index = list(
            model.classes_
        ).index(prediction)

        st.success(
            f"Predicted Sentiment: **{prediction.upper()}**"
        )

        st.write(
            f"SVM decision score: "
            f"**{scores[predicted_index]:.4f}**"
        )

    elif model_key == "bilstm":
        model, vocab, classes, max_length, device = (
            load_bilstm()
        )

        ids = text_to_ids(
            sentence,
            vocab,
            max_length
        )

        x = torch.tensor(
            [ids],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():
            logits = model(x)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0].cpu().numpy()

        predicted_id = int(
            np.argmax(probabilities)
        )

        prediction = classes[
            predicted_id
        ]

        confidence = float(
            probabilities[predicted_id]
        )

        st.success(
            f"Predicted Sentiment: **{prediction.upper()}**"
        )

        st.write(
            f"Confidence: **{confidence:.2%}**"
        )

        probability_df = pd.DataFrame(
            {
                "Sentiment": classes,
                "Probability": probabilities
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Sentiment"
            )["Probability"]
        )

    else:
        model, tokenizer, classes, device = (
            load_bert()
        )

        encoded = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            output = model(
                **encoded
            )

            probabilities = torch.softmax(
                output.logits,
                dim=1
            )[0].cpu().numpy()

        predicted_id = int(
            np.argmax(probabilities)
        )

        prediction = classes[
            predicted_id
        ]

        confidence = float(
            probabilities[predicted_id]
        )

        st.success(
            f"Predicted Sentiment: **{prediction.upper()}**"
        )

        st.write(
            f"Confidence: **{confidence:.2%}**"
        )

        probability_df = pd.DataFrame(
            {
                "Sentiment": classes,
                "Probability": probabilities
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Sentiment"
            )["Probability"]
        )
