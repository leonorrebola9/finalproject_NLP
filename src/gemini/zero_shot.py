'''
gemini/zero_shot.py

Zero-shot prompt: only gives Gemini the list of labels.

Saves the predictions to results/gemini_zero_shot_validation.csv and shows
the accuracy (overall and per sentiment).
'''

from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
import os
import time

# Project root = two levels above src/gemini/
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

df = pd.read_csv(os.path.join(DATA_DIR, "sample_validation.csv"))
load_dotenv()  # Load environment variables from .env file
client = genai.Client()  # finds GEMINI_API_KEY automatically

MODEL = 'gemini-3.5-flash-lite'
PAUSE = 4.5  # seconds between requests (free tier: max 15 requests per minute)
STRATEGY = "zero_shot"

# Zero-shot: only the list of labels
PROMPT = (
    "Classify the sentiment of this Reddit comment as positive, "
    "negative, ambiguous or neutral. Answer with one word only.\n\n"
    "Comment: {comment}"
)


def classify(comment):
    response = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(comment=comment),
        config=types.GenerateContentConfig(temperature=0),  # same answer every time
    )
    return response.text.strip().lower().strip(".")  # "Negative." -> "negative"


print(f"Model: {MODEL} | Prompt: {STRATEGY}\n")

predictions = []  # keep the answers
for i, (_, row) in enumerate(df.iterrows(), start=1):
    print(f"[{i}/{len(df)}] Sending...")
    for attempt in range(3):  # if the API fails (e.g. rate limit),
        try:                  # wait 1 minute and try again (max 3 times)
            prediction = classify(row["text"])
            break
        except Exception as e:
            print(f"  Error ({str(e)[:40]}...). Waiting 60 seconds...")
            time.sleep(60)
    else:
        prediction = "error"  # gave up on this comment
    predictions.append(prediction)
    print(f"  Gemini: {prediction}  |  True: {row['sentiment']}")
    time.sleep(PAUSE)  # respect the requests-per-minute limit

# Save the results
df["prediction"] = predictions
os.makedirs(RESULTS_DIR, exist_ok=True)
df.to_csv(os.path.join(RESULTS_DIR, f"gemini_{STRATEGY}_validation.csv"), index=False)

# Overall accuracy
correct = df["prediction"] == df["sentiment"]
print(f"\nModel: {MODEL} | Prompt: {STRATEGY}")
print(f"Accuracy: {correct.mean():.1%} ({correct.sum()}/{len(df)})")

# Accuracy per sentiment
print("\nCorrect answers per sentiment:")
for sentiment in ["positive", "negative", "ambiguous", "neutral"]:
    part = correct[df["sentiment"] == sentiment]
    print(f"  {sentiment:<10} {part.sum():>2}/{len(part)}  ({part.mean():.0%})")

# Confusion table: rows = true sentiment, columns = Gemini's answer
print("\nTrue sentiment (rows) vs Gemini's answer (columns):")
print(pd.crosstab(df["sentiment"], df["prediction"]))