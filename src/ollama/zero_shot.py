'''
ollama/zero_shot.py

Zero-shot prompt: only gives the model the list of labels.

Saves the predictions to results/ollama/zero_shot_validation.csv and shows
the accuracy (overall and per sentiment).
'''

import ollama
import pandas as pd
import os

# Project root = two levels above src/ollama/
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results", "ollama")  # each model has its own folder

df = pd.read_csv(os.path.join(DATA_DIR, "sample_validation.csv"))
MODEL = 'llama3.1:8b'  # runs locally with Ollama (no rate limits)
STRATEGY = "zero_shot"

# Zero-shot: only the list of labels
PROMPT = (
    "Classify the sentiment of this Reddit comment as positive, "
    "negative, ambiguous or neutral. Answer with one word only.\n\n"
    "Comment: {comment}"
)


def classify(comment):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(comment=comment)}],
        options={"temperature": 0},  # same answer every time
    )
    text = response["message"]["content"].strip().lower()
    # Small models sometimes add an explanation: keep only the first word
    return text.split()[0].strip(".*,:!") if text else "invalid"  # "Negative." -> "negative"


print(f"Model: {MODEL} | Prompt: {STRATEGY}\n")

predictions = []  # keep the answers
for i, (_, row) in enumerate(df.iterrows(), start=1):
    print(f"[{i}/{len(df)}] Sending...")
    prediction = classify(row["text"])
    predictions.append(prediction)
    print(f"  Ollama: {prediction}  |  True: {row['sentiment']}")

# Save the results
df["prediction"] = predictions
os.makedirs(RESULTS_DIR, exist_ok=True)
df.to_csv(os.path.join(RESULTS_DIR, f"ollama_{STRATEGY}_validation.csv"), index=False)

# Overall accuracy
correct = df["prediction"] == df["sentiment"]
print(f"\nModel: {MODEL} | Prompt: {STRATEGY}")
print(f"Accuracy: {correct.mean():.1%} ({correct.sum()}/{len(df)})")

# Accuracy per sentiment
print("\nCorrect answers per sentiment:")
for sentiment in ["positive", "negative", "ambiguous", "neutral"]:
    part = correct[df["sentiment"] == sentiment]
    print(f"  {sentiment:<10} {part.sum():>2}/{len(part)}  ({part.mean():.0%})")

# Confusion table: rows = true sentiment, columns = model's answer
print("\nTrue sentiment (rows) vs model's answer (columns):")
print(pd.crosstab(df["sentiment"], df["prediction"]))