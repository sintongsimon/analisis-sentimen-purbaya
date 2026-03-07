# Import required libraries
import os
import pandas as pd
from datetime import datetime, timedelta
from google.colab import drive

# Mount Google Drive to access files
drive.mount('/content/drive')

# Define input folder and output merged file path
INPUT_DIR  = "/content/drive/MyDrive/Colab Notebooks/Script"
OUTPUT_FILE = f"{INPUT_DIR}/merged_tweets_2025-09-08_to_2025-12-30.csv"

# Define date range for files to merge
start_date = datetime(2025, 9, 8)
end_date   = datetime(2025, 12, 30)

# Specify columns to keep and rename mapping
columns_to_keep = ["full_text", "created_at"]
columns_rename = {
    "full_text": "Text",
    "created_at": "Date"
}

# Initialize containers for dataframes and missing files
dfs = []
missing_files = []

# Iterate through each date within the defined range
current = start_date
while current <= end_date:

    # Build filename based on current date
    date_str = current.strftime("%Y-%m-%d")
    filename = f"tweets_{date_str}.csv"
    filepath = os.path.join(INPUT_DIR, filename)

    # Check if the CSV file exists
    if os.path.exists(filepath):
        try:
            # Read CSV file into dataframe
            df = pd.read_csv(filepath)

            # Keep only the required columns if available
            available_cols = [c for c in columns_to_keep if c in df.columns]
            df = df[available_cols]

            # Rename columns to standardized names
            df = df.rename(columns=columns_rename)

            # Store dataframe for later merging
            dfs.append(df)

            # Log successful load
            print(f"✔ Loaded {filename} ({len(df)} rows)")

        except Exception as e:
            # Log error if file cannot be read
            print(f"⚠️ Error reading {filename}: {e}")
    else:
        # Record missing file
        missing_files.append(filename)

    # Move to the next date
    current += timedelta(days=1)

# Merge all collected dataframes if any were loaded
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicate tweets based on text
    merged_df.drop_duplicates(subset="Text", inplace=True)

    # Save the merged dataset to CSV
    merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Print merge summary
    print("\n✅ MERGE SELESAI")
    print(f"Total tweets: {len(merged_df)}")
    print(f"Saved to: {OUTPUT_FILE}")
else:
    # Handle case where no files were successfully loaded
    print("❌ Tidak ada file CSV yang berhasil dibaca")

# Print list of missing files
print("\n📌 File tidak ditemukan:")
for f in missing_files:
    print("-", f)

# Final completion message
print(f"✔ Done")
