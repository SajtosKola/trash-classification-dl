import csv
from pathlib import Path
from collections import Counter
import os

# A dataset helye
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# A hat kategória neve
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# A dataset közvetlenül a raw mappában található
DATASET_DIR = DATA_ROOT

# Ellenőrizzük, hogy mindhárom mappa létezik-e
for split in ["train", "valid", "test"]:
    if not (DATASET_DIR / split).is_dir():
        raise RuntimeError(
            f"Hiányzó mappa: {split}"
        )

# Mindhárom adathalmaz ellenőrzése
for split in ["train", "valid", "test"]:

    folder = DATASET_DIR / split
    csv_files = list(folder.glob("*.csv"))

    if len(csv_files) != 1:
        print(f"\n{split}: Pontosan egy CSV-fájl szükséges!")
        continue

    csv_path = csv_files[0]

    class_counts = Counter()
    invalid_labels = []
    missing_images = []

    total = 0

    with open(csv_path, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row_number, raw_row in enumerate(reader, start=2):

            # Eltávolítjuk a felesleges szóközöket
            row = {
                key.strip(): value.strip()
                for key, value in raw_row.items()
                if key is not None and value is not None
            }

            total += 1

            # Ellenőrizzük a címkéket
            try:
                labels = [int(row[name]) for name in CLASSES]
            except (KeyError, ValueError, TypeError):
                invalid_labels.append(row_number)
                continue

            # Pontosan egy kategória legyen aktív
            if any(value not in (0, 1) for value in labels):
                invalid_labels.append(row_number)
                continue

            if sum(labels) != 1:
                invalid_labels.append(row_number)
                continue

            # Megszámoljuk a kategóriákat
            class_index = labels.index(1)
            class_counts[CLASSES[class_index]] += 1

            # Ellenőrizzük, hogy létezik-e a kép
            filename = row.get("filename", "")
            if not filename or not (folder / filename).is_file():
                missing_images.append(filename or f"Sor: {row_number}")

    print(f"\n===== {split.upper()} =====")
    print(f"Képek száma a CSV-ben: {total}")

    print("\nKategóriánként:")
    for category in CLASSES:
        print(f"{category}: {class_counts[category]}")

    print(f"\nHibás címkék: {len(invalid_labels)}")
    print(f"Hiányzó képfájlok: {len(missing_images)}")

print("\nDataset ellenőrzése befejeződött.")
