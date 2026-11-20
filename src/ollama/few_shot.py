'''
ollama/few_shot.py

Few-shot prompt: the definitions prompt plus 2 real examples per label,
taken from the GoEmotions TRAINING set (never from validation or test).

Saves the predictions to results/ollama/few_shot_validation.csv and shows
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
STRATEGY = "few_shot"

# Few-shot: definitions + 2 labeled examples per sentiment (from the training set)
PROMPT = (
    "Classify the sentiment of this Reddit comment into one of four labels:\n"
    "- positive: admiration, amusement, approval, caring, desire, excitement, "
    "gratitude, joy, love, optimism, pride or relief\n"
    "- negative: anger, annoyance, disappointment, disapproval, disgust, "
    "embarrassment, fear, grief, nervousness, remorse or sadness\n"
    "- ambiguous: confusion, curiosity, realization or surprise "
    "(e.g. questions, wondering, discovering something, being surprised)\n"
    "- neutral: no clear emotion\n\n"
    "Examples:\n"
    "Comment: I stand fully corrected, thank you!\n"
    "Sentiment: positive\n\n"
    "Comment: I went to the exhibit in Chicago some years ago. It's absolutely stunning\n"
    "Sentiment: positive\n\n"
    "Comment: What a dumbass question to start with...\n"
    "Sentiment: negative\n\n"
    "Comment: I stopped worrying about showing vulnerability years ago but there's still nobody there.\n"
    "Sentiment: negative\n\n"
    "Comment: Am I the only one who can't get the sound to work?\n"
    "Sentiment: ambiguous\n\n"
    "Comment: I was flabbergasted when I saw it.\n"
    "Sentiment: ambiguous\n\n"
    "Comment: Also drill man exists for an alternate for Balrog.\n"
    "Sentiment: neutral\n\n"
    "Comment: That's typical Wigs. Hot one game, cold the next three.\n"
    "Sentiment: neutral\n\n"
    "Answer with one word only: positive, negative, ambiguous or neutral.\n\n"
    "Comment: {comment}\n"
    "Sentiment:"
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