# PyTorch Stock Sentiment Project

This version removes TensorFlow and Keras completely.

## Models

- SVM + TF-IDF -> scikit-learn
- BiLSTM -> PyTorch
- BERT -> PyTorch + Hugging Face Transformers

## Install

```powershell
python -m pip install -r requirements.txt
```

## Pre-train

Open menu:

```powershell
python pretrain_models.py
```

Or directly:

```powershell
python pretrain_models.py svm
python pretrain_models.py bilstm
python pretrain_models.py bert
```

The same saved fixed 20% test dataset is used for every model.

## Start Streamlit GUI

```powershell
python -m streamlit run streamlit_app.py
```

The GUI does not train the models. It loads the saved models/results.
