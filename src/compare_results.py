'''
compare_results.py

Reads every results CSV (results/gemini/ and results/ollama/) and builds
the tables and figures for the report:
1) summary table: accuracy, F1 macro, accuracy per sentiment and number of
   out-of-format answers, for every model x prompt x sample;
2) chart comparing the prompts on the validation sample;
3) confusion matrices of the final evaluation (test sample).
'''

import pandas as pd
import matplotlib.pyplot as plt
import os

# Project root = one level above src/
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

MODELS = {"gemini": "Gemini 3.5 flash-lite", "ollama": "Llama 3.1 8B"}
PROMPTS = ["zero_shot", "definitions", "few_shot", "chain_of_thought"]
SENTIMENTS = ["positive", "negative", "ambiguous", "neutral"]


def f1_macro(df):
    """Average of the F1 score of each sentiment (all sentiments count the same)."""
    scores = []
    for s in SENTIMENTS:
        true_pos = ((df["prediction"] == s) & (df["sentiment"] == s)).sum()
        predicted = (df["prediction"] == s).sum()
        actual = (df["sentiment"] == s).sum()
        precision = true_pos / predicted if predicted else 0
        recall = true_pos / actual if actual else 0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return sum(scores) / len(scores)


# 1) Read every CSV and compute the metrics
rows = []
for folder, model_name in MODELS.items():
    for file in sorted(os.listdir(os.path.join(RESULTS_DIR, folder))):
        if not file.endswith(".csv"):
            continue
        # file names like "few_shot_test.csv" (or "gemini_few_shot_test.csv")
        name = file.removesuffix(".csv").removeprefix(f"{folder}_")
        prompt, sample = name.rsplit("_", 1)

        df = pd.read_csv(os.path.join(RESULTS_DIR, folder, file))
        correct = df["prediction"] == df["sentiment"]
        row = {
            "model": model_name,
            "prompt": prompt,
            "sample": sample,
            "n": len(df),
            "accuracy": round(correct.mean() * 100, 1),
            "f1_macro": round(f1_macro(df), 2),
        }
        for s in SENTIMENTS:  # accuracy per sentiment
            row[s] = round(correct[df["sentiment"] == s].mean() * 100)
        row["out_of_format"] = (~df["prediction"].isin(SENTIMENTS)).sum()
        rows.append(row)

summary = pd.DataFrame(rows)
summary["prompt"] = pd.Categorical(summary["prompt"], PROMPTS, ordered=True)
summary = summary.sort_values(["sample", "model", "prompt"], ascending=[False, True, True])

pd.set_option("display.width", 200)
print("Summary (accuracy and per-sentiment values in %):\n")
print(summary.to_string(index=False))
summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
print("\nSaved results/summary.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)

# 2) Chart: accuracy of each prompt on the validation sample
validation = summary[summary["sample"] == "validation"]
if not validation.empty:
    table = validation.pivot(index="prompt", columns="model", values="accuracy").reindex(PROMPTS)
    ax = table.plot(kind="bar", figsize=(8, 5), color=["#A03AAB", "#73ED89"], rot=0)
    for container, model in zip(ax.containers, table.columns):
        labels = [f"{v:.0f}%" if pd.notna(v) else "" for v in table[model]]  # no label if missing
        ax.bar_label(container, labels=labels, padding=2)
    ax.axhline(25, color="gray", linestyle="--", linewidth=1, label="Random (25%)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prompt comparison (validation, 100 comments)", fontweight="bold")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "prompt_comparison.png"), dpi=200)
    plt.close()
    print("Saved figures/prompt_comparison.png")

# 3) Confusion matrices of the final evaluation (test sample)
for folder, model_name in MODELS.items():
    for file in os.listdir(os.path.join(RESULTS_DIR, folder)):
        if not file.endswith("_test.csv"):
            continue
        df = pd.read_csv(os.path.join(RESULTS_DIR, folder, file))
        # answers outside the 4 labels are grouped as "other"
        df["prediction"] = df["prediction"].where(df["prediction"].isin(SENTIMENTS), "other")
        columns = SENTIMENTS + (["other"] if (df["prediction"] == "other").any() else [])
        matrix = pd.crosstab(df["sentiment"], df["prediction"]).reindex(
            index=SENTIMENTS, columns=columns, fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(matrix.values, cmap="Purples")
        ax.set_xticks(range(len(columns)), columns)
        ax.set_yticks(range(len(SENTIMENTS)), SENTIMENTS)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.values[i, j]
                ax.text(j, i, value, ha="center", va="center",
                        color="white" if value > matrix.values.max() / 2 else "black")
        ax.set_xlabel("Model's answer")
        ax.set_ylabel("True sentiment")
        ax.set_title(f"{model_name} - test ({len(df)} comments)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"confusion_{folder}_test.png"), dpi=200)
        plt.close()
        print(f"Saved figures/confusion_{folder}_test.png")