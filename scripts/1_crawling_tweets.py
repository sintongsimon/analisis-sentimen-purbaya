#@title Crawl data

# Auth token from twitter/x
twitter_auth_token = ''


# Import required Python package
!pip install pandas


# Install Node.js (because tweet-harvest built using Node.js)
!sudo apt-get update
!sudo apt-get install -y ca-certificates curl gnupg
!sudo mkdir -p /etc/apt/keyrings
!curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
!NODE_MAJOR=20 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
!sudo apt-get update
!sudo apt-get install nodejs -y
!node -v


# Mount google drive
from google.colab import drive
import os
drive.mount('/content/drive', force_remount=True)
OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/Script"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Install playwright
!npx playwright install-deps
!npx playwright install chromium


# Import library python
import pandas as pd
import time
from datetime import datetime, timedelta
import shutil


# Function to wait
def sleep_with_log(total_seconds, step=60):
    """
    Sleep with periodic log output to prevent Colab idle timeout
    """
    for elapsed in range(0, total_seconds, step):
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Sleeping... {elapsed}/{total_seconds} seconds")
        time.sleep(step)


# Set range date to crawl
start = datetime(2025, 9, 24)
end   = datetime(2025, 9, 30)
dates = []
while start < end:
    dates.append(start.strftime("%Y-%m-%d"))
    start += timedelta(days=1)


# Crawl data
for i in range(len(dates) - 1):
    since = dates[i]
    until = dates[i + 1]

    search = f"purbaya yudhi sadewa since:{since} until:{until} lang:id"
    output = f"tweets_{since}.csv"

    !npx -y tweet-harvest@2.6.1 \
        -o "{output}" \
        -s "{search}" \
        --tab "LATEST" \
        -l 100 \
        --token {twitter_auth_token}

    src = f"/content/tweets-data/{output}"
    dst = f"{OUTPUT_DIR}/{output}"

    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"✔ Saved {dst}")
    else:
        print("⚠️ No file created — possible limit")

    print(f"✔ Done {since}")
    
    
    # Pause between crawling processes
    sleep_with_log(total_seconds=60, step=60)
