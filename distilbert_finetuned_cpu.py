import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =====================
# CONFIG (CPU SAFE)
# =====================
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8          # small for CPU
EPOCHS = 4              # DistilBERT converges fast
LR = 2e-5
CSV_OUT = "model_accuracy_distilbert.csv"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("cleaned_dataset.csv")  # change if needed
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

le = LabelEncoder()
labels = le.fit_transform(labels)
num_labels = len(le.classes_)

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# =====================
# TOKENIZATION
# =====================
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )

train_enc = tokenize(X_train)
val_enc = tokenize(X_val)

# =====================
# MODEL
# =====================
model = TFDistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# =====================
# TRAIN
# =====================
history = model.fit(
    train_enc.data,
    y_train,
    validation_data=(val_enc.data, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =====================
# EVALUATE
# =====================
logits = model.predict(val_enc.data).logits
preds = np.argmax(logits, axis=1)
val_acc = accuracy_score(y_val, preds)

# =====================
# SAVE MODEL
# =====================
model.save("distilbert_finetuned.keras")

# =====================
# LOG TO NEW CSV
# =====================
row = {
    "Model": "DistilBERT",
    "Validation Accuracy": round(val_acc, 4),
    "Epochs": EPOCHS,
    "Batch Size": BATCH_SIZE,
    "Max Length": MAX_LEN
}

if os.path.exists(CSV_OUT):
    pd.read_csv(CSV_OUT).append(row, ignore_index=True).to_csv(CSV_OUT, index=False)
else:
    pd.DataFrame([row]).to_csv(CSV_OUT, index=False)

print("✅ DistilBERT training complete")
print("📊 Validation Accuracy:", val_acc)
print("📁 Saved to:", CSV_OUT)
