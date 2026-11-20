'''
test_gemini.py

First test: checks that the Gemini API works and classifies one comment.
'''

from dotenv import load_dotenv
from google import genai
import pandas as pd

df = pd.read_csv("../data/go_emotions_test.csv").sample(5, random_state=42)
load_dotenv()  # Load environment variables from .env file
client = genai.Client()  # finds GEMINI_API_KEY automatically

MODEL = 'gemini-3.5-flash-lite'
for i, (_, row) in enumerate(df.iterrows(), start=1):
    print(f"[{i}/5] Sending...")  # shows that something is happening
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Classify the sentiment of this Reddit comment as positive, "
                 f"negative, ambiguous or neutral. Answer with one word only.\n\n"
                 f"Comment: {row['text']}",
    )
    print(f"Comment: {row['text']}")
    print(f"  Gemini: {response.text.strip()}  |  True emotions: {row['emotions']}")