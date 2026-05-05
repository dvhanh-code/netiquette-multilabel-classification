from pathlib import Path
import pandas as pd
from langdetect import detect, DetectorFactory
from collections import Counter
from tqdm import tqdm

DetectorFactory.seed = 42

jigsaw_dir = Path("data/raw/jigsaw")

# ---- Load data (same as loader) ----
train = pd.read_csv(jigsaw_dir / "train.csv")
train["split"] = "train"

test_text = pd.read_csv(jigsaw_dir / "test.csv")
test_labels = pd.read_csv(jigsaw_dir / "test_labels.csv")

test = test_text.merge(test_labels, on="id")
test = test[test["toxic"] != -1]
test["split"] = "test"

raw = pd.concat(
    [
        train[["comment_text", "split"]],
        test[["comment_text", "split"]],
    ],
    ignore_index=True,
)

print(f"Total rows: {len(raw):,}")

# ---- Language detection ----

def safe_detect(text):
    try:
        return detect(str(text))
    except:
        return "unknown"

langs = []

print("\nRunning language detection on FULL dataset...\n")

for text in tqdm(raw["comment_text"], desc="Detecting language"):
    langs.append(safe_detect(text))

counter = Counter(langs)

print("\n=== FULL DATASET LANGUAGE DISTRIBUTION ===")
for lang, count in counter.most_common():
    print(f"{lang}: {count:,} ({count / len(raw) * 100:.2f}%)")