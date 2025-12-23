# =========================
# shared_preprocessing.py
# =========================

import re
import string
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# =========================
# Download stopwords (first run only)
# =========================
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# =========================
# Global configuration (USED BY ALL MODELS)
# =========================
VECTOR_SIZE = 100
MAX_LEN = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Keep important question words
IMPORTANT_WORDS = {"not", "no", "why", "how", "what", "when", "where"}
STOP_WORDS = set(stopwords.words("english")) - IMPORTANT_WORDS

# =========================
# Text cleaning
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

# =========================
# Load & preprocess dataset
# =========================
def load_and_preprocess(csv_path="train.csv"):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    df["QA Text"] = df["QA Text"].apply(clean_text)

    sentences = df["QA Text"].apply(
        lambda x: simple_preprocess(str(x))
    ).tolist()

    return df, sentences

# =========================
# Train Word2Vec (shared)
# =========================
def train_word2vec(sentences):
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=6,
        min_count=3,
        sg=1,               # Skip‑gram
        epochs=10,
        workers=4
    )

    word_index = {w: i + 1 for i, w in enumerate(w2v.wv.index_to_key)}

    embedding_matrix = np.zeros((len(word_index) + 1, VECTOR_SIZE))
    for word, i in word_index.items():
        embedding_matrix[i] = w2v.wv[word]

    return word_index, embedding_matrix

# =========================
# Encode sequences
# =========================
def encode_and_pad(sentences, word_index):
    def encode(sentence):
        return [word_index[w] for w in sentence if w in word_index]

    sequences = [encode(s) for s in sentences]

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return X

# =========================
# Encode labels & split
# =========================
def prepare_labels_and_split(X, labels):
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    y_cat = to_categorical(y_enc)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_cat,
        test_size=TEST_SIZE,
        stratify=y_cat,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, y_train, y_val, le
