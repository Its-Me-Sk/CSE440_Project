from tensorflow.keras.models import load_model

model = load_model("cnn_finetuned.keras")
model.summary()
