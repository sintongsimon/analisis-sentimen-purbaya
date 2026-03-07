# Import required libraries
import pandas as pd
import re
import string
from google.colab import drive

# Install and import Indonesian NLP library (Sastrawi)
!pip install Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Mount Google Drive to access dataset files
drive.mount('/content/drive', force_remount=True)

# Define dataset directory and input file path
INPUT_DIR  = "/content/drive/MyDrive/Colab Notebooks/Script"
filepath = f"{INPUT_DIR}/selected_tweets_2025-09-08_to_2025-12-30.csv"

# Load tweet dataset and normalization dictionary
df = pd.read_csv(filepath)
df_normalisasi = pd.read_csv(f"{INPUT_DIR}/normalisasi.csv")

# Convert normalization table into dictionary format
normalisasi_dict = dict(
    zip(
        df_normalisasi.iloc[:, 0],  # kolom tidak baku
        df_normalisasi.iloc[:, 1]   # kolom baku
    )
)

# Basic text cleaning (case folding, remove URL, mention, hashtag, numbers, punctuation)
def basic_cleaning(text):
    text = text.lower()  # case folding
    text = re.sub(r"http\S+|www\S+", "", text)  # hapus URL
    text = re.sub(r"@\w+", "", text)  # hapus mention
    text = re.sub(r"#\w+", "", text)  # hapus hashtag
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = re.sub(r"\s+", " ", text).strip()  # hapus spasi berlebih
    return text

# Normalize informal words using custom dictionary
def normalisasi_text(text, norm_dict):
    words = text.split()
    normalized_words = [
        norm_dict.get(w, w)  # ganti jika ada di kamus
        for w in words
    ]
    return " ".join(normalized_words)

# Full preprocessing pipeline for SVM (cleaning, normalization, stopword removal, stemming)
def preprocess_svm(text):
    text = basic_cleaning(text)
    text = normalisasi_text(text, normalisasi_dict)
    words = text.split()
    words = [w for w in words if w not in stopword]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

# Lighter preprocessing pipeline for IndoBERTweet (minimal cleaning without stemming)
def preprocess_bert(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalisasi_text(text, normalisasi_dict)
    return text

# Initialize stemmer and Indonesian stopword list
stemmer = StemmerFactory().create_stemmer()
stopword = set(StopWordRemoverFactory().get_stop_words())

# Display number of tweets before preprocessing
print("Jumlah data:", len(df))

# Apply preprocessing pipelines for SVM and IndoBERTweet
df["CleanSVM"] = df["Text"].astype(str).apply(preprocess_svm)
df["CleanIndoBERTweet"] = df["Text"].astype(str).apply(preprocess_bert)

# Preview preprocessing result
df[["Text", "CleanSVM", "CleanIndoBERTweet"]].head()

# Remove rows with empty preprocessing results
df = df[df["CleanSVM"].str.strip() != ""]

# Remove duplicate tweets based on processed SVM text
df = df.drop_duplicates(subset=['CleanSVM'])

# Display dataset size after preprocessing
print("Jumlah data setelah preprocessing:", len(df))

# Save final preprocessed dataset
df.to_csv(INPUT_DIR + "/preprocessed_tweets_2_2025-09-08_to_2025-12-30.csv", index=False)
print("Data preprocessing berhasil disimpan")
