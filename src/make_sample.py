'''
make_samples.py

Creates balanced samples for evaluating the LLMs:
1) converts each comment's emotions into one sentiment
   (positive / negative / ambiguous / neutral, or mixed);
2) removes mixed comments (emotions from more than one sentiment);
3) samples the same number of comments per sentiment from:
   - validation -> to test and improve the prompts
   - test       -> for the final evaluation (used only at the end)
'''

import pandas as pd
import os

# Project root = one level above src/
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data")

# Official mapping from the GoEmotions authors (+ neutral)
SENTIMENTS = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism",
                 "caring", "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment",
                 "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
    "neutral": ["neutral"],
}
EMOTION_TO_SENTIMENT = {e: s for s, emotions in SENTIMENTS.items() for e in emotions}


def to_sentiment(emotions):
    """'anger;annoyance' -> 'negative'. Emotions from different sentiments -> 'mixed'."""
    sentiments = {EMOTION_TO_SENTIMENT[e] for e in emotions.split(";")}
    if len(sentiments) == 1:
        return sentiments.pop()
    return "mixed"


def make_sample(split, n_per_sentiment):
    df = pd.read_csv(os.path.join(DATA_DIR, f"go_emotions_{split}.csv"))
    df["sentiment"] = df["emotions"].apply(to_sentiment)

    print(f"\n{split}: sentiment of each comment")
    print(df["sentiment"].value_counts())

    df = df[df["sentiment"] != "mixed"]  # keep only comments with a clear sentiment

    sample = (df.groupby("sentiment")
                .sample(n=n_per_sentiment, random_state=42)  # same seed = same sample every time
                .sample(frac=1, random_state=42))            # shuffle the rows

    path = os.path.join(DATA_DIR, f"sample_{split}.csv")
    sample.to_csv(path, index=False)
    print(f"Saved {len(sample)} comments ({n_per_sentiment} per sentiment) to data/sample_{split}.csv")



make_sample("validation", n_per_sentiment=25)  # 100 comments to improve the prompts
make_sample("test", n_per_sentiment=60)        # 240 comments for the final evaluation