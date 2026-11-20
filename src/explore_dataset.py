'''
explore_dataset.py

Explores the emotions in the GoEmotions dataset:
1) lists every emotion found in the training data and how often it appears;
2) groups the emotions into sentiments (official mapping by the GoEmotions authors);
3) saves a bar chart colored by sentiment for the report;
4) checks that the train, validation and test splits have similar distributions.
'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# Project root = one level above src/
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data")
FIGURES_DIR = os.path.join(ROOT, "figures")

df = pd.read_csv(os.path.join(DATA_DIR, "go_emotions_train.csv"))

# 1) List every emotion found in the data and how often it appears
counts = df["emotions"].str.split(";").explode().value_counts()
print(f"Found {len(counts)} different emotions in {len(df)} comments:\n")
print(counts)

# 2) Group emotions into sentiments
# Official mapping from the GoEmotions authors (sentiment_mapping.json in their GitHub repo)
SENTIMENTS = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism",
                 "caring", "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment",
                 "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
    "neutral": ["neutral"] # this is an extra label (not in the official mapping)
}

print("\nEmotions grouped by sentiment:")
for sentiment, emotions in SENTIMENTS.items():
    total = counts[emotions].sum()
    print(f"\n{sentiment.upper()} ({len(emotions)} emotions, {total} occurrences)")
    print("  " + ", ".join(emotions))

# 3) Bar chart colored by sentiment
COLORS = {"positive": "green", "negative": "red", "ambiguous": "gold", "neutral": "gray"}
emotion_color = {e: COLORS[s] for s, emotions in SENTIMENTS.items() for e in emotions}

plt.figure(figsize=(8, 8))
bars = plt.barh(counts.index, counts.values, color=[emotion_color[e] for e in counts.index])
plt.gca().invert_yaxis()  # most frequent at the top
plt.bar_label(bars, padding=3)  # number at the end of each bar
plt.xlim(0, counts.max() * 1.1)  # extra space so the largest number fits
plt.xlabel("Number of comments")
plt.title("Emotion frequency in GoEmotions (train)", fontweight="bold")

# Color legend
legend = [Patch(color=c, label=s.capitalize()) for s, c in COLORS.items()]
plt.legend(handles=legend, title="Sentiment", loc="lower right")
plt.tight_layout()

os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, "emotion_distribution.png"), dpi=200)
print("\nChart saved to figures/emotion_distribution.png")

# 4) Check that the three splits have similar distributions
emotion_sentiment = {e: s for s, emotions in SENTIMENTS.items() for e in emotions}
comparison = {}
for split in ["train", "validation", "test"]:
    d = pd.read_csv(os.path.join(DATA_DIR, f"go_emotions_{split}.csv"))
    emotions = d["emotions"].str.split(";").explode()
    share = emotions.map(emotion_sentiment).value_counts(normalize=True) * 100
    comparison[f"{split} ({len(d)})"] = share.round(1)

print("\nSentiment distribution per split (% of emotion occurrences):")
print(pd.DataFrame(comparison).loc[list(SENTIMENTS)])