# =========================
# cnn_finetuned.py
# =========================

import gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from accuracy_logger import save_accuracy

# 🔹 Shared preprocessing
from shared_preprocessing import (
    load_and_preprocess,
    train_word2vec,
    encode_and_pad,
    prepare_labels_and_split,
    VECTOR_SIZE,
    MAX_LEN
)

# =========================
# 1. Load & preprocess data
# =========================
df, sentences = load_and_preprocess("train.csv")

word_index, embedding_matrix = train_word2vec(sentences)
X = encode_and_pad(sentences, word_index)

X_train, X_val, y_train, y_val, label_encoder = prepare_labels_and_split(
    X, df["Class"]
)

num_classes = y_train.shape[1]

# =========================
# 2. Callbacks
# =========================
early_stop = EarlyStopping(
    patience=3,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5
)

# =========================
# 3. Fine‑tuned CNN Model
# =========================
model = Sequential([
    Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=VECTOR_SIZE,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=True
    ),

    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    ),

    GlobalMaxPooling1D(),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 4. Train
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=32,   # CPU‑safe
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

best_val_acc = max(history.history["val_accuracy"])
print(f"\n✅ CNN Best Validation Accuracy: {best_val_acc:.4f}")

# =========================
# 5. Evaluation
# =========================
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

print("\n📊 Classification Report (CNN):")
print(classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_
))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title("CNN Fine‑Tuned Confusion Matrix")
plt.tight_layout()
plt.savefig("cnn_finetuned_cm.png")
plt.close()

print("✅ Confusion matrix saved as cnn_finetuned_cm.png")

# =========================
# 6. Save model
# =========================
model.save("cnn_finetuned.keras")
print("✅ Model saved as cnn_finetuned.keras")

# =========================
# 7. Cleanup
# =========================
K.clear_session()
gc.collect()
