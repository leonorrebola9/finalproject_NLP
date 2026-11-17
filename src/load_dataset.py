'''
load_dataset.py

Loads the GoEmotions dataset (Reddit comments labeled with 27 emotions
+ neutral) from Hugging Face and saves the training split to a CSV file
for the rest of the project.

Notes:
- Text is kept as-is (punctuation and emojis help the LLM detect sentiment).
- Labels are stored as emotion ids separated by commas (e.g. "17,18").
'''

from datasets import load_dataset
import pandas as pd
import os

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
print(dataset)

EMOTIONS = dataset["train"].features["labels"].feature.names

# data/ folder at the project root (one level above src/)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

for split in ["train", "validation", "test"]:
    df = pd.DataFrame(dataset[split])
    df["text"] = df["text"].str.strip()  # only removes extra spaces at the ends
    df["emotions"] = df["labels"].apply(lambda ids: ";".join(EMOTIONS[i] for i in ids))
    df = df[["id", "text", "emotions"]]
    df.to_csv(os.path.join(DATA_DIR, f"go_emotions_{split}.csv"), index=False)
    print(f"{split}: {len(df)} comments saved")