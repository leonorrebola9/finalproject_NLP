'''
classify_gemini.py

Classifies the comments of a sample with Gemini, using one of several
prompt strategies, saves the predictions to a CSV file and shows the
accuracy (overall and per sentiment).

To test another prompt: add it to PROMPTS and change STRATEGY.
'''

from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
import os
import time

# Project root = one level above src/
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

df = pd.read_csv(os.path.join(DATA_DIR, "sample_validation.csv"))
load_dotenv()  # Load environment variables from .env file
client = genai.Client()  # finds GEMINI_API_KEY automatically

MODEL = 'gemini-3.5-flash-lite'
PAUSE = 4.5  # seconds between requests (free tier: max 15 requests per minute)
STRATEGY = "definitions"  # which prompt to use (a key of PROMPTS)

PROMPTS = {
    # 1) Zero-shot: only the list of labels
    "zero_shot": (
        "Classify the sentiment of this Reddit comment as positive, "
        "negative, ambiguous or neutral. Answer with one word only.\n\n"
        "Comment: {comment}"
    ),

    # 2) Zero-shot with definitions: explains what each label means in GoEmotions
    "definitions": (
        "Classify the sentiment of this Reddit comment into one of four labels:\n"
        "- positive: admiration, amusement, approval, caring, desire, excitement, "
        "gratitude, joy, love, optimism, pride or relief\n"
        "- negative: anger, annoyance, disappointment, disapproval, disgust, "
        "embarrassment, fear, grief, nervousness, remorse or sadness\n"
        "- ambiguous: confusion, curiosity, realization or surprise "
        "(e.g. questions, wondering, discovering something, being surprised)\n"
        "- neutral: no clear emotion\n\n"
        "Answer with one word only: positive, negative, ambiguous or neutral.\n\n"
        "Comment: {comment}"
    ),
}


def classify(comment):
    response = client.models.generate_content(
        model=MODEL,
        contents=PROMPTS[STRATEGY].format(comment=comment),
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