import io
import os
import zipfile
import urllib.request

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "spam.csv")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUT_PATH):
        print("Dataset already exists, skipping download.")
        return

    print("Downloading dataset...")
    with urllib.request.urlopen(URL) as response:
        data = response.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("SMSSpamCollection") as f:
            lines = f.read().decode("utf-8").splitlines()

    rows = ["v1,v2"]
    for line in lines:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            label, message = parts
            message = message.replace('"', '""')
            rows.append(f'{label},"{message}"')

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))

    print(f"Saved {len(rows) - 1} messages to {OUT_PATH}")


if __name__ == "__main__":
    download()
