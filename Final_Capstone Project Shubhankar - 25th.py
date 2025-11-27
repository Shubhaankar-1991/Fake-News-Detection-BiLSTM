# CELL 0 — Imports & Setup
# ===========================================================
import os
import random
import re
import string
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# NLP & utilities
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.style.use("seaborn")
sns.set_context("talk")
print("✅ Imports complete, seed set.")

# ===========================================================
# CELL 1 — Download NLTK stopwords (run once)
# ===========================================================
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
print("✅ NLTK stopwords ready.")

# ===========================================================
# CELL 2 — Load & Label Data
# ===========================================================

fake_path = "Fake.csv"
true_path = "True.csv"

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("Dataset loaded. Shape:", df.shape)
print(df['label'].value_counts())

# ===========================================================
# CELL 3 — Improved Text Cleaning & Normalization
# ===========================================================
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text() if pd.notna(text) else ""

def normalize_text(text, stopwords_set=STOP_WORDS):
    if pd.isna(text): return ""
    text = clean_html(text)
    text = text.lower()
    # URLs -> placeholder
    text = re.sub(r"http\S+|www\.\S+|https\S+", " URL ", text)
    # Emails -> placeholder
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    # Numbers -> placeholder
    text = re.sub(r"\b\d+(\.\d+)?\b", " NUMBER ", text)
    # Normalize quotes and punctuation clusters
    text = re.sub(r"[‘’`´]", "'", text)
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"[^\w\s'']", " ", text)  # keep words and apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords (optional — helps TF-IDF and classical models)
    words = [w for w in text.split() if w not in stopwords_set]
    return " ".join(words)

# create a combined content field: title + text
df['title'] = df['title'].astype(str).fillna("")
df['text'] = df['text'].astype(str).fillna("")
df['content_raw'] = (df['title'] + ". " + df['text']).str.strip()
df['content'] = df['content_raw'].apply(normalize_text)

# Optional: drop very short items
df = df[df['content'].str.len() > 30].reset_index(drop=True)

print("Cleaning done. Remaining records:", len(df))

# ===========================================================
# CELL 4 — Quick EDA (distribution, text length, wordclouds)
# ===========================================================
# Label distribution
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df, palette='Set2')
plt.xticks([0,1], ['FAKE','REAL'])
plt.title('Fake vs Real Distribution')
plt.show()

# Text length histogram
df['text_length'] = df['content'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8,4))
sns.histplot(df['text_length'], bins=40, kde=True)
plt.title('Article length (words)')
plt.show()

print(df['text_length'].describe().to_string())

# ===========================================================
# CELL 5 — Train / Validation / Test split (for deep model)
# ===========================================================

X = df['content']
y = df['label'].values

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=SEED)

print("Splits sizes -> Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

# ===========================================================
# CELL 6 — TF-IDF Vectorization (for baselines)
# ===========================================================
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF shapes:", X_train_tfidf.shape, X_val_tfidf.shape, X_test_tfidf.shape)

# ===========================================================
# CELL 7 — Baseline Models: Train & Evaluate (LR, NB, RF)
# ===========================================================
def eval_model(name, model, X_tr, y_tr, X_te, y_te, return_probs=True):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:,1] if return_probs else None
    stats = {'Model': name,'Accuracy': accuracy_score(y_te, preds),
        'Precision': precision_score(y_te, preds),
        'Recall': recall_score(y_te, preds),
        'F1 Score': f1_score(y_te, preds),
        'ROC-AUC': roc_auc_score(y_te, probs) if probs is not None else np.nan
    }
    print(f"\n{name} Classification Report:\n", classification_report(y_te, preds, target_names=['FAKE','REAL']))
    return model, stats

baselines = {}
results_list = []

# Logistic Regression (baseline)
lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr, stats = eval_model("Logistic Regression (Base)", lr, X_train_tfidf, y_train, X_test_tfidf, y_test)
baselines['lr_base'] = lr
results_list.append(stats)

# Multinomial Naive Bayes
nb = MultinomialNB()
nb, stats = eval_model("Multinomial NB", nb, X_train_tfidf, y_train, X_test_tfidf, y_test)
baselines['nb'] = nb
results_list.append(stats)

# Random Forest (baseline)
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf, stats = eval_model("Random Forest", rf, X_train_tfidf, y_train, X_test_tfidf, y_test)
baselines['rf'] = rf
results_list.append(stats)

baseline_df = pd.DataFrame(results_list).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
print("\nBaseline summary:\n", baseline_df)

# ===========================================================
# CELL 8 — GridSearchCV on Logistic Regression (5-fold CV)
# ===========================================================
param_grid = {'C': [0.01, 0.1, 1.0, 10.0],'penalty': ['l2'],'solver': ['lbfgs']}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=SEED),
    param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
)
grid.fit(X_train_tfidf, y_train)

print("Best params (LR):", grid.best_params_)
print("Best CV F1 (LR):", grid.best_score_)

best_lr = grid.best_estimator_
# evaluate tuned LR on test
y_pred_lr = best_lr.predict(X_test_tfidf)
y_prob_lr = best_lr.predict_proba(X_test_tfidf)[:,1]
print("\nTuned LR test report:\n", classification_report(y_test, y_pred_lr, target_names=['FAKE','REAL']))

# add to comparison list
tuned_lr_stats = {
    'Model': 'Logistic Regression (Tuned)',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1 Score': f1_score(y_test, y_pred_lr),
    'ROC-AUC': roc_auc_score(y_test, y_prob_lr)
}
baseline_df = pd.concat([baseline_df, pd.DataFrame([tuned_lr_stats])], ignore_index=True).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
print("\nUpdated baseline summary:\n", baseline_df)

# ===========================================================
# CELL 10 — Prepare Tokenizer & Sequences for BiLSTM (ensure OOV token set)
# ===========================================================
vocab_size = 12000
maxlen = 250
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)  # fit on training text only

def texts_to_padded_sequences(texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')

X_train_seq = texts_to_padded_sequences(X_train)
X_val_seq = texts_to_padded_sequences(X_val)
X_test_seq = texts_to_padded_sequences(X_test)

y_train_arr = np.array(y_train)
y_val_arr = np.array(y_val)
y_test_arr = np.array(y_test)

print("Sequences shapes:", X_train_seq.shape, X_val_seq.shape, X_test_seq.shape)

embedding_dim = 128

bilstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
bilstm_model.summary()

# ===========================================================
# CELL 12 — Train BiLSTM (with callbacks)
# ===========================================================
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define callbacks
es = EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=1,verbose=1)

# Define safe checkpoint directory and path
checkpoint_path = "./models/best_bilstm_tf"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

mcp = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy',save_best_only=True,mode='max',verbose=1,save_format='tf'

# Train model
history = bilstm_model.fit(X_train_seq, y_train_arr,epochs=8,batch_size=64,validation_data=(X_val_seq, y_val_arr),
    callbacks=[es, rlr, mcp],
    verbose=1)

# ===========================================================
# Plot train/validation curves
# ===========================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('BiLSTM Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('BiLSTM Loss')
plt.legend()

plt.show()

# ===========================================================
# CELL 13 — Evaluate BiLSTM on Test Set
# ===========================================================
bilstm_probs = bilstm_model.predict(X_test_seq).ravel()
bilstm_preds = (bilstm_probs > 0.5).astype(int)

print("BiLSTM Test Metrics:")
print("Accuracy:", accuracy_score(y_test_arr, bilstm_preds))
print("Precision:", precision_score(y_test_arr, bilstm_preds))
print("Recall:", recall_score(y_test_arr, bilstm_preds))
print("F1 Score:", f1_score(y_test_arr, bilstm_preds))
print("ROC-AUC:", roc_auc_score(y_test_arr, bilstm_probs))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_arr, bilstm_preds))

# ROC curve
fpr, tpr, _ = roc_curve(y_test_arr, bilstm_probs)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'BiLSTM (AUC={roc_auc_score(y_test_arr,bilstm_probs):.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('BiLSTM ROC')
plt.legend()
plt.show()

# ===========================================================
# CELL 14 — MODEL COMPARISON (assemble all models' metrics)
# ===========================================================
comparison = []

# Base LR (trained earlier on TF-IDF)
lr_preds_base = baselines['lr_base'].predict(X_test_tfidf)
lr_probs_base = baselines['lr_base'].predict_proba(X_test_tfidf)[:,1]
comparison.append({
    'Model': 'Logistic Regression (Base)',
    'Accuracy': accuracy_score(y_test, lr_preds_base),
    'Precision': precision_score(y_test, lr_preds_base),
    'Recall': recall_score(y_test, lr_preds_base),
    'F1 Score': f1_score(y_test, lr_preds_base),
    'ROC-AUC': roc_auc_score(y_test, lr_probs_base)
})

# Multinomial NB
nb_preds = baselines['nb'].predict(X_test_tfidf)
nb_probs = baselines['nb'].predict_proba(X_test_tfidf)[:,1]
comparison.append({
    'Model': 'Multinomial NB',
    'Accuracy': accuracy_score(y_test, nb_preds),
    'Precision': precision_score(y_test, nb_preds),
    'Recall': recall_score(y_test, nb_preds),
    'F1 Score': f1_score(y_test, nb_preds),
    'ROC-AUC': roc_auc_score(y_test, nb_probs)
})

# Random Forest
rf_preds = baselines['rf'].predict(X_test_tfidf)
rf_probs = baselines['rf'].predict_proba(X_test_tfidf)[:,1]
comparison.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, rf_preds),
    'Precision': precision_score(y_test, rf_preds),
    'Recall': recall_score(y_test, rf_preds),
    'F1 Score': f1_score(y_test, rf_preds),
    'ROC-AUC': roc_auc_score(y_test, rf_probs)
})

# Tuned LR
comparison.append(tuned_lr_stats)

# BiLSTM
comparison.append({
    'Model': 'BiLSTM (Deep Learning)',
    'Accuracy': accuracy_score(y_test_arr, bilstm_preds),
    'Precision': precision_score(y_test_arr, bilstm_preds),
    'Recall': recall_score(y_test_arr, bilstm_preds),
    'F1 Score': f1_score(y_test_arr, bilstm_preds),
    'ROC-AUC': roc_auc_score(y_test_arr, bilstm_probs)
})

comparison_df = pd.DataFrame(comparison).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
print("\n=== MODEL COMPARISON ===\n")
display(comparison_df)

# Bar plot (F1 Score)
plt.figure(figsize=(10,5))
sns.barplot(x='F1 Score', y='Model', data=comparison_df, palette='viridis')
plt.title('Model Comparison by F1 Score')
plt.xlim(0,1)
plt.show()

# ===========================================================
# CELL 15 — Save models & tokenizer
# ===========================================================
# Save TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Save tuned logistic regression
with open('logreg_tuned.pkl', 'wb') as f:
    pickle.dump(best_lr, f)

# Save tokenizer and BiLSTM model
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
bilstm_model.save('bilstm_model.h5')

print("✅ Saved tfidf_vectorizer.pkl, logreg_tuned.pkl, tokenizer.pkl, bilstm_model.h5")
