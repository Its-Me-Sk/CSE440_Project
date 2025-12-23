# main_fixed.py

import gc
import re
import string
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # prevents GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# =========================
# 0. Load dataset
# =========================
print("📌 Loading dataset...")
df = pd.read_csv("train.csv", engine="python", on_bad_lines="skip")
print(f"Dataset loaded. Shape: {df.shape}")

# =========================
# 1. Preprocessing
# =========================
print("📌 Cleaning text...")
stop_words = set(stopwords.words('english'))

def nlp_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['QA Text'] = df['QA Text'].apply(nlp_clean)
print("✅ Text cleaned.")

# =========================
# 2. TF-IDF + Naive Bayes
# =========================
print("\n========================")
print("📌 Training TF-IDF + Naive Bayes...")
X = df['QA Text']
y = df['Class']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=12000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

y_pred = nb.predict(X_val_tfidf)
print("✅ Naive Bayes Classification Report:")
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.savefig("nb_confusion_matrix.png")
plt.close()
print("✅ Naive Bayes confusion matrix saved as nb_confusion_matrix.png")

# =========================
# 3. TF-IDF + Dense NN (Memory-Efficient)
# =========================
print("\n========================")
print("📌 Training TF-IDF + Dense NN (Memory-Efficient)...")

# Reduced features to prevent memory crash
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec_sparse = tfidf.fit_transform(X)
X_vec = X_vec_sparse.astype(np.float32).toarray()  # smaller dtype

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

X_train, X_val, y_train, y_val = train_test_split(X_vec, y_cat, test_size=0.2, random_state=42)

dense_model = Sequential([
    Dense(512, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation="softmax")
])
dense_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(patience=2, restore_best_weights=True)

history = dense_model.fit(
    X_train, y_train, epochs=8, batch_size=64,
    validation_split=0.1, callbacks=[early_stop]
)

y_pred = np.argmax(dense_model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Dense NN (TF-IDF) Confusion Matrix")
plt.savefig("dense_tfidf_cm.png")
plt.close()
print("✅ Dense NN confusion matrix saved as dense_tfidf_cm.png")

dense_val_acc = max(history.history['val_accuracy'])
print(f"✅ Dense NN Best Validation Accuracy: {dense_val_acc:.4f}")

del dense_model
K.clear_session()
gc.collect()

# =========================
# 4. Word2Vec Embeddings
# =========================
print("\n========================")
print("📌 Creating Word2Vec embeddings (Skip-Gram)...")
sentences = df['QA Text'].apply(lambda x: simple_preprocess(str(x))).tolist()

w2v = Word2Vec(sentences=sentences, vector_size=75, window=5, min_count=3, sg=1, workers=4, epochs=5)
word_index = {w: i+1 for i, w in enumerate(w2v.wv.index_to_key)}

embedding_matrix = np.zeros((len(word_index)+1, 75), dtype=np.float32)
for word, i in word_index.items():
    embedding_matrix[i] = w2v.wv[word]

MAX_LEN = 50
def encode(s):
    return [word_index[w] for w in s if w in word_index]
X_seq = pad_sequences([encode(s) for s in sentences], maxlen=MAX_LEN)

y_enc = le.transform(df['Class'])
y_cat = to_categorical(y_enc)
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
print("✅ Word2Vec embeddings created.")

# =========================
# 5. Model Helper with Accuracy Tracking
# =========================
model_accuracies = {}  # Dictionary to store accuracy for each model

def train_evaluate_model(model, name, save_model=True):
    print(f"\n========================")
    print(f"📌 Training {name}...")
    early_stop = EarlyStopping(patience=2, restore_best_weights=True)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    history = model.fit(
        X_train, y_train, 
        epochs=8, batch_size=64, 
        validation_split=0.1, 
        callbacks=[early_stop], 
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    model_accuracies[name] = best_val_acc
    print(f"✅ {name} Best Validation Accuracy: {best_val_acc:.4f}")

    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    print(f"✅ {name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"{name.replace(' ','_').lower()}_cm.png")
    plt.close()
    print(f"✅ {name} confusion matrix saved as {name.replace(' ','_').lower()}_cm.png")

    if save_model:
        model.save(f"{name.replace(' ','_').lower()}.h5")
        print(f"✅ {name} model saved as {name.replace(' ','_').lower()}.h5")

    del model
    K.clear_session()
    gc.collect()

# =========================
# 6. Define & Train Models (Word2Vec)
# =========================

# CNN
cnn_model = Sequential([
    Embedding(embedding_matrix.shape[0], 75, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
    Conv1D(128, kernel_size=5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation="softmax")
])
train_evaluate_model(cnn_model, "CNN")

# Uni-RNN
rnn_model = Sequential([
    Embedding(embedding_matrix.shape[0], 75, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
    SimpleRNN(128),
    Dense(y_cat.shape[1], activation="softmax")
])
train_evaluate_model(rnn_model, "Uni-RNN")

# Uni-LSTM
lstm_model = Sequential([
    Embedding(embedding_matrix.shape[0], 75, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
    LSTM(128),
    Dense(y_cat.shape[1], activation="softmax")
])
train_evaluate_model(lstm_model, "Uni-LSTM")

# Uni-GRU
gru_model = Sequential([
    Embedding(embedding_matrix.shape[0], 75, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
    GRU(128),
    Dense(y_cat.shape[1], activation="softmax")
])
train_evaluate_model(gru_model, "Uni-GRU")

# Bi-RNN
bi_rnn = Sequential([
    Embedding(embedding_matrix.shape[0], 75, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
    Bidirectional(SimpleRNN(128)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation="softmax")
])
train_evaluate_model(bi_rnn, "Bi-RNN")

# Bi-LSTM
bi_lstm = Sequential([
    Embedding(embedding_matrix.shape[0], 75, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
    Bidirectional(LSTM(128)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation="softmax")
])
train_evaluate_model(bi_lstm, "Bi-LSTM")

# Bi-GRU
bi_gru = Sequential([
    Embedding(embedding_matrix.shape[0], 75, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False),
    Bidirectional(GRU(128)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(y_cat.shape[1], activation="softmax")
])
train_evaluate_model(bi_gru, "Bi-GRU")

# =========================
# Save all model accuracies
# =========================
print("\n📌 All Model Accuracies:")
for model_name, acc in model_accuracies.items():
    print(f"{model_name}: {acc:.4f}")

acc_df = pd.DataFrame(list(model_accuracies.items()), columns=["Model", "Validation Accuracy"])
acc_df.to_csv("model_accuracies.csv", index=False)
print("✅ Model accuracies saved to model_accuracies.csv")

print("\n🎉 All models trained, evaluated, and saved successfully!")
