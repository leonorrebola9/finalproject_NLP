'''
load_dataset.py

Loads the IMDB dataset, cleans the text, explores the structure 
and saves samples for the rest of the project.
'''

from datasets import load_dataset
import pandas as pd
import re
import os

# Load the IMDB dataset from the Hugging Face datasets library
dataset = load_dataset("")

# Check the structure of the dataset
print(dataset)

# Convert the dataset to a pandas DataFrame for easier manipulation
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Display the first few rows of the training DataFrame
print("Training DataFrame:")
print(train_df.head())
print("\nTest DataFrame:")
print(test_df.head())

# Clean the text
def clean_text(dataset):
    # Remove HTML tags (keep punctuation, needed later for sentence splitting)
    dataset['text'] = dataset['text'].apply(lambda x: re.sub(r'<.*?>', ' ', x))
    # Normalize extra whitespace
    dataset['text'] = dataset['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return dataset

# Clean the text in the training and test DataFrames
train_df = clean_text(train_df)
test_df = clean_text(test_df)

# Keep a sample
sample_train = train_df.sample(n=50, random_state=42)
print("Sample Training DataFrame:")
print(sample_train)

sample_test = test_df.sample(n=100, random_state=42)
print("Sample Test DataFrame:")
print(sample_test)

# Apply cleaning to the samples
sample_train = clean_text(sample_train)
sample_test = clean_text(sample_test)

# Save the cleaned samples to CSV files
os.makedirs("data", exist_ok=True)

sample_train.to_csv("data/sample_train.csv", index=False)
sample_test.to_csv("data/sample_test.csv", index=False)