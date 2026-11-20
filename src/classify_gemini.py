'''
classify_gemini.py

Classifies the comments of a sample with Gemini (zero-shot prompt) and
saves the predictions to a CSV file and computes the accuracy.
'''

from dotenv import load_dotenv
from google import genai
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

def classify(comment):                                                
    response = client.models.generate_content(
        model=MODEL,
        # Zero-shot prompt
        contents=f"Classify the sentiment of this Reddit comment as positive, "     # KEPT: same prompt
                 f"negative, ambiguous or neutral. Answer with one word only.\n\n"
                 f"Comment: {comment}",
    )
    return response.text.strip().lower()                              # CHANGED: .lower() -> "negative"


predictions = []            # keep the answers
for i, (_, row) in enumerate(df.iterrows(), start=1):
    print(f"[{i}/{len(df)}] Sending...")                              
    for attempt in range(3):        # if the API fails (e.g. rate limit),
        try:                        # wait 1 minute and try again (max 3 times)
            prediction = classify(row["text"])
            break
        except Exception as e:
            print(f"  Error ({str(e)[:40]}...). Waiting 60 seconds...")
            time.sleep(60)
    else:
        prediction = "error"        # gave up on this comment
    predictions.append(prediction)
    time.sleep(PAUSE)               # respect the requests-per-minute limit
    print(f"  Gemini: {prediction}  |  True: {row['sentiment']}")     # compare with sentiment

df["prediction"] = predictions      # save the results
os.makedirs(RESULTS_DIR, exist_ok=True)
df.to_csv(os.path.join(RESULTS_DIR, "gemini_zero_shot_validation.csv"), index=False)

accuracy = (df["prediction"] == df["sentiment"]).mean()               # accuracy
print(f"\nAccuracy: {accuracy:.1%} ({(df['prediction'] == df['sentiment']).sum()}/{len(df)})")