
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Dataset elérési útja
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
TRAIN_DIR = DATA_ROOT / "train"

CLASSES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

# CSV beolvasása
csv_file = list(TRAIN_DIR.glob("*.csv"))[0]

with open(csv_file, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    rows = [
        {
            key.strip(): value.strip()
            for key, value in row.items()
            if key is not None and value is not None
        }
        for row in reader
    ]

# Minden kategóriából kiválasztunk egy véletlen képet
random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, category in zip(axes.flat, CLASSES):

    matching_images = [
        row for row in rows
        if row[category] == "1"
    ]

    sample = random.choice(matching_images)

    image_path = TRAIN_DIR / sample["filename"]
    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)
    ax.set_title(category.capitalize())
    ax.axis("off")

plt.tight_layout()
plt.show()
