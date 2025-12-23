import os
import pandas as pd

CSV_FILE = "model_accuracies.csv"

def save_accuracy(model_name, val_accuracy):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["Model", "Validation Accuracy"])

    # remove old entry if exists
    df = df[df["Model"] != model_name]

    # add new result
    df.loc[len(df)] = [model_name, round(val_accuracy, 4)]

    df.to_csv(CSV_FILE, index=False)
    print(f"📊 Accuracy saved → {CSV_FILE}")
