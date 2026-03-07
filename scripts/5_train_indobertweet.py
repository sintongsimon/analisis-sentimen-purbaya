# Install required ML and NLP libraries
!pip install transformers datasets torch scikit-learn

# Import required Python libraries
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Mount Google Drive to access dataset and save results
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# =========================
# CONFIG
# =========================

# Define dataset path, model name, and training configuration
INPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/Script"
DATA_PATH = f"{INPUT_DIR}/labeled_tweets_2025-09-08_to_2025-12-30.csv"
MODEL_BASE = "indolem/indobertweet-base-uncased"
N_SPLITS = 5
NUM_LABELS = 2
EPOCHS = 3
BATCH_SIZE = 16

# Define column names used in the dataset
TEXT_COL = "CleanIndoBERTweet"
LABEL_COL = "Label"

# =========================
# LOAD DATA
# =========================

# Load labeled tweet dataset
df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COL, LABEL_COL]].dropna()

# Display original label distribution
print(df[LABEL_COL].value_counts())

# Convert labels into binary format (0 and 1)
df[LABEL_COL] = df[LABEL_COL].map({
    1: 0,
    2: 1
})

# Display label distribution after mapping
print(df[LABEL_COL].value_counts())

# =========================
# TOKENIZER
# =========================

# Load tokenizer from pretrained IndoBERTweet model
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

# =========================
# DATASET CLASS
# =========================

# Custom PyTorch dataset class for tokenized tweets
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels

    # Return tokenized item and label
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    # Return dataset length
    def __len__(self):
        return len(self.labels)

# =========================
# METRICS
# =========================

# Function to calculate evaluation metrics during training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =========================
# K-FOLD TRAIN + EVAL
# =========================

# Initialize stratified K-Fold cross-validation
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

# Container to store evaluation metrics from all folds
metrics_all = []

# Perform training and evaluation for each fold
for fold, (train_idx, val_idx) in enumerate(
    skf.split(df[TEXT_COL], df[LABEL_COL]), 1
):
    print(f"\n===== FOLD {fold} =====")

    # Create training dataset
    train_dataset = TweetDataset(
        df[TEXT_COL].iloc[train_idx].tolist(),
        df[LABEL_COL].iloc[train_idx].tolist()
    )

    # Create validation dataset
    val_dataset = TweetDataset(
        df[TEXT_COL].iloc[val_idx].tolist(),
        df[LABEL_COL].iloc[val_idx].tolist()
    )

    # Load pretrained IndoBERTweet model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE,
        num_labels=NUM_LABELS
    )

    # Configure training parameters
    training_args = TrainingArguments(
        output_dir=f"./results/fold_{fold}",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_dir="./logs",
        report_to="none"
    )

    # Initialize HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # =========================
    # TRAIN
    # =========================

    # Train the model on training dataset
    trainer.train()

    # =========================
    # EVALUATE
    # =========================

    # Evaluate model performance on validation dataset
    eval_metrics = trainer.evaluate()
    eval_metrics["fold"] = fold
    metrics_all.append(eval_metrics)

    # Generate predictions for validation data
    preds = trainer.predict(val_dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    # Print classification report for the fold
    print(classification_report(y_true, y_pred))

    # =========================
    # SAVE MODEL
    # =========================

    # Save trained model and tokenizer for this fold
    model_path = f"{INPUT_DIR}/model/indobertweet_sentiment/{fold}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # =========================
    # SAVE PREDICTIONS
    # =========================

    # Save prediction results for this fold
    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    }).to_csv(
        f"{INPUT_DIR}/results/indobertweet_fold_{fold}_predictions.csv",
        index=False
    )

# =========================
# SAVE ALL METRICS
# =========================

# Convert metrics from all folds into dataframe
metrics_df = pd.DataFrame(metrics_all)

# Save cross-validation metrics to CSV
metrics_df.to_csv(
    f"{INPUT_DIR}/metrics_indobertweet.csv",
    index=False
)

# Print final completion message
print("\n===== ALL FOLDS DONE =====")
