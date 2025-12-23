import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, GRU, Dense, Dropout, Bidirectional
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from accuracy_logger import save_accuracy

# =========================
# Load Preprocessed Data
# =========================
print("📌 Loading cleaned dataset...")
df = pd.read_csv("train.csv", engine="python", on_bad_lines="skip")

# IMPORTANT: text must already be cleaned using shared_preprocessing.py
sentences = df["QA Text"].apply(lambda x: simple_preprocess(str(x))).tolist()

# =========================
# Train Word2Vec (Stronger)
# =========================
print("📌 Training Word2Vec...")
w2v = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=7,
    min_count=2,
    sg=1,
    workers=4,
    epochs=10
)

word_index = {w: i + 1 for i, w in enumerate(w2v.wv.index_to_key)}
embedding_dim = 100

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_matrix[i] = w2v.wv[word]

# =========================
# Encode & Pad
# =========================
MAX_LEN = 80

def encode(text):
    return [word_index[w] for w in text if w in word_index]

X_seq = pad_sequences(
    [encode(s) for s in sentences],
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

le = LabelEncoder()
y_enc = le.fit_transform(df["Class"])
y_cat = to_categorical(y_enc)

X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# =========================
# Bi-GRU FINETUNED MODEL
# =========================
print("📌 Building Bi-GRU (Fine-tuned)...")

model = Sequential([
    Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True   # 🔥 key improvement
    ),

    Bidirectional(
        GRU(128, return_sequences=True)
    ),
    Dropout(0.3),

    Bidirectional(
        GRU(64)
    ),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.4),

    Dense(y_cat.shape[1], activation="softmax")
])

optimizer = Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Callbacks
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
    verbose=1
)

# =========================
# Train
# =========================
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# =========================
# Evaluate
# =========================
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

print("\n✅ Bi-GRU (Fine-tuned) Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9,7))
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Greens",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("Bi-GRU Fine-tuned Confusion Matrix")
plt.savefig("bi_gru_finetuned_cm.png")
plt.close()

# =========================
# Save Model
# =========================
model.save("bi_gru_finetuned.keras")
print("✅ Model saved as bi_gru_finetuned.keras")

del model
K.clear_session()
gc.collect()

print("\n🎉 Bi-GRU Fine-tuning Completed Successfully!")
