'''
ollama/chain_of_thought.py

Chain-of-thought prompt: the same definitions and examples as few_shot.py,
but the model first explains its reasoning in one or two sentences and only
then gives the final label. The reasoning is saved in the CSV (useful for
the error analysis).

Saves the predictions to results/ollama/chain_of_thought_validation.csv and shows
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
STRATEGY = "chain_of_thought"

# Chain-of-thought: few-shot + a short reasoning before the answer
PROMPT = (
    "Classify the sentiment of this Reddit comment into one of four labels:\n"
    "- positive: admiration, amusement, approval, caring, desire, excitement, "
    "gratitude, joy, love, optimism, pride or relief\n"
    "- negative: anger, annoyance, disappointment, disapproval, disgust, "
    "embarrassment, fear, grief, nervousness, remorse or sadness\n"
    "- ambiguous: confusion, curiosity, realization or surprise "
    "(e.g. questions, wondering, discovering something, being surprised)\n"
    "- neutral: no clear emotion\n\n"
    "First explain your reasoning in one or two sentences. Then, on the last line, "
    "write 'Sentiment:' followed by one word only: positive, negative, ambiguous or neutral.\n\n"
    "Examples:\n"
    "Comment: I stand fully corrected, thank you!\n"
    "Reasoning: The author thanks someone for a correction, which shows gratitude.\n"
    "Sentiment: positive\n\n"
    "Comment: I went to the exhibit in Chicago some years ago. It's absolutely stunning\n"
    "Reasoning: Calling the exhibit 'absolutely stunning' expresses admiration.\n"
    "Sentiment: positive\n\n"
    "Comment: What a dumbass question to start with...\n"
    "Reasoning: The author insults the question, which shows annoyance.\n"
    "Sentiment: negative\n\n"
    "Comment: I stopped worrying about showing vulnerability years ago but there's still nobody there.\n"
    "Reasoning: The author feels alone despite opening up, which expresses sadness.\n"
    "Sentiment: negative\n\n"
    "Comment: Am I the only one who can't get the sound to work?\n"
    "Reasoning: The author asks a question because they want to know something, which is curiosity.\n"
    "Sentiment: ambiguous\n\n"
    "Comment: I was flabbergasted when I saw it.\n"
    "Reasoning: 'Flabbergasted' means very surprised, without saying if it was good or bad.\n"
    "Sentiment: ambiguous\n\n"
    "Comment: Also drill man exists for an alternate for Balrog.\n"
    "Reasoning: The author only states a fact about a game, with no emotion.\n"
    "Sentiment: neutral\n\n"
    "Comment: That's typical Wigs. Hot one game, cold the next three.\n"
    "Reasoning: The author describes a player's usual performance without expressing a feeling.\n"
    "Sentiment: neutral\n\n"
    "Comment: {comment}\n"
    "Reasoning:"
)

def classify(comment):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(comment=comment)}],
        options={"temperature": 0},  # same answer every time
    )
    text = response["message"]["content"].strip()

    # The answer is after the last "Sentiment:"; everything before it is the reasoning
    if "sentiment:" not in text.lower():
        return "invalid", text  # the model did not follow the format
    position = text.lower().rfind("sentiment:")
    reasoning = text[:position].strip()
    answer = text[position + len("sentiment:"):].strip().lower()
    prediction = answer.split()[0].strip(".*") if answer else "invalid"  # "Negative." -> "negative"
    return prediction, reasoning


print(f"Model: {MODEL} | Prompt: {STRATEGY}\n")

predictions = []  # keep the answers
reasonings = []   # keep the explanations
for i, (_, row) in enumerate(df.iterrows(), start=1):
    print(f"[{i}/{len(df)}] Sending...")
    prediction, reasoning = classify(row["text"])
    predictions.append(prediction)
    reasonings.append(reasoning)
    print(f"  Ollama: {prediction}  |  True: {row['sentiment']}")

# Save the results
df["prediction"] = predictions
df["reasoning"] = reasonings
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