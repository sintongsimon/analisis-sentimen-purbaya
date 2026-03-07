# Seleksi Data Tweet
import os
import pandas as pd
import numpy as np
import re

from google.colab import drive

# Mount Google Drive to access dataset
drive.mount('/content/drive', force_remount=True)

# Install and import language detection library
!pip install langdetect
from langdetect import detect, LangDetectException

# Define dataset path
INPUT_DIR  = "/content/drive/MyDrive/Colab Notebooks/Script"
filepath = f"{INPUT_DIR}/merged_tweets_2025-09-08_to_2025-12-30.csv"

# Load merged tweets dataset
df = pd.read_csv(filepath)

# Print initial number of tweets
print("Jumlah data awal:", len(df))

# Remove duplicate tweets based on text
df = df.drop_duplicates(subset='Text')
print("Setelah hapus duplikat:", len(df))

# Remove raw retweets starting with "RT @"
df = df[~df['Text'].str.startswith('RT @', na=False)]
print("Setelah hapus retweet mentah:", len(df))

# Define function to detect noisy tweets
def is_noise(text):
    text = str(text)

    # Remove tweets that contain only a URL
    if re.fullmatch(r"https?://\S+", text):
        return True

    # Remove tweets that contain only mentions
    if re.fullmatch(r"(@\w+\s*)+", text):
        return True

    # Remove tweets that are too short
    if len(text.strip()) < 5:
        return True

    return False

# Filter out noise tweets
df = df[~df['Text'].apply(is_noise)]
print("Setelah hapus noise:", len(df))

# Define safe language detection function
def detect_lang(text):
    try:
        return detect(text)
    except LangDetectException:
        return None

# Detect language of each tweet
df['lang'] = df['Text'].apply(detect_lang)

# Keep only Indonesian tweets
df = df[df['lang'] == 'id']
df = df.drop(columns=['lang'])

print("Setelah seleksi bahasa:", len(df))

# Define keywords related to economic/political topic
keywords = [
    'purbaya',
    'menteri',
    'ekonomi',
    'keuangan',
    'finansial',
    'pemerintah'
]

# Build regex pattern from keywords
pattern = '|'.join(keywords)

# Keep tweets containing at least one keyword
df = df[df['Text'].str.lower().str.contains(pattern, na=False)]
print("Setelah seleksi relevansi:", len(df))

# Define common spam keywords
spam_keywords = [
    'promo', 'diskon', 'giveaway', 'jual', 'iklan', 'gratis'
]

# Remove tweets containing spam keywords
df = df[~df['Text'].str.lower().str.contains('|'.join(spam_keywords), na=False)]
print("Setelah hapus spam:", len(df))

# Save the cleaned and selected dataset
df.to_csv(INPUT_DIR + "/selected_tweets_2025-09-08_to_2025-12-30.csv", index=False)
print("Dataset bersih berhasil disimpan")

# Final completion message
print(f"✔ Done")
