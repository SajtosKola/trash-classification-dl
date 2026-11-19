import csv
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt

# A projekt gyökérmappája
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Az adatok helye
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# Kategóriák
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

SPLITS = ["train", "valid", "test"]

# Az eredmények tárolása
split_counts = {}

# Mindhárom CSV beolvasása
for split in SPLITS:

    folder = DATA_ROOT / split
    csv_files = list(folder.glob("*.csv"))

    if len(csv_files) != 1:
        raise RuntimeError(
            f"{split}: Pontosan egy CSV-fájl szükséges!"
        )

    counts = Counter()

    with open(
        csv_files[0],
        "r",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for raw_row in reader:

            row = {
                key.strip(): value.strip()
                for key, value in raw_row.items()
                if key is not None and value is not None
            }

            active_classes = [
                category
                for category in CLASSES
                if row.get(category) == "1"
            ]

            if len(active_classes) != 1:
                raise ValueError(
                    f"Hibás címke a {split} adathalmazban!"
                )

            counts[active_classes[0]] += 1

    split_counts[split] = counts


# Összesített képszámok
total_counts = Counter()

for counts in split_counts.values():
    total_counts.update(counts)

total_images = sum(total_counts.values())


# Szöveges eredmények
print("===== OSZTÁLYELOSZLÁS =====")
print(f"Összes kép: {total_images}\n")

for category in CLASSES:

    count = total_counts[category]
    percentage = count / total_images * 100

    print(
        f"{category}: {count} kép "
        f"({percentage:.2f}%)"
    )


# Diagram készítése
values = [total_counts[category] for category in CLASSES]

plt.figure(figsize=(10, 6))

bars = plt.bar(CLASSES, values)

plt.title("Hulladékkategóriák eloszlása")
plt.xlabel("Kategória")
plt.ylabel("Képek száma")

# Értékek megjelenítése az oszlopok felett
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Diagram mentése
output_dir = PROJECT_ROOT / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "class_distribution.png"

plt.savefig(output_path, dpi=150)

print(f"\nDiagram elmentve ide: {output_path}")

print("\n===== OSZTÁLYELOSZLÁS SPLITENKÉNT =====")

for split in SPLITS:
    counts = split_counts[split]
    split_total = sum(counts.values())

    print(f"\n{split.upper()} ({split_total} kép):")

    for category in CLASSES:
        count = counts[category]
        percentage = count / split_total * 100

        print(f"{category}: {count} kép ({percentage:.2f}%)")

plt.show()