import pandas as pd

INPUT_CSV = "model_accuracies.csv"
OUTPUT_CSV = "finetuned_model_accuracies.csv"

# load existing results
df = pd.read_csv(INPUT_CSV)

# keep only fine‑tuned models
keywords = [
    "Fine", "finetuned",
    "CNN", "Uni-LSTM", "Uni-GRU",
    "Bi-LSTM", "Bi-GRU"
]

mask = df["Model"].str.contains("|".join(keywords), case=False, na=False)
df_finetuned = df[mask]

# save new CSV
df_finetuned.to_csv(OUTPUT_CSV, index=False)

print("✅ Done!")
print("Created:", OUTPUT_CSV)
