'''
load_dataset.py

Loads the GoEmotions dataset, cleans the text, explores the structure 
and saves samples for the rest of the project.
'''

from datasets import load_dataset
import pandas as pd
import os

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
print(dataset)

EMOTIONS = dataset["train"].features["labels"].feature.names

os.makedirs("data", exist_ok=True)

for split in ["train", "validation", "test"]:
    df = pd.DataFrame(dataset[split])
    df["text"] = df["text"].str.strip()  # only removes extra spaces at the ends
    df["emotions"] = df["labels"].apply(lambda ids: ";".join(EMOTIONS[i] for i in ids))
    df = df[["id", "text", "emotions"]]
    df.to_csv(f"data/go_emotions_{split}.csv", index=False)
    print(f"{split}: {len(df)} comments saved")