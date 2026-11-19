from pathlib import Path
from collections import Counter
from PIL import Image
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

SPLITS = ["train", "valid", "test"]

image_sizes = Counter()
image_modes = Counter()
total_images = 0

for split in SPLITS:
    folder = DATA_ROOT / split
    csv_file = list(folder.glob("*.csv"))[0]

    with open(csv_file, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = folder / row["filename"]

            with Image.open(image_path) as img:
                image_sizes[img.size] += 1
                image_modes[img.mode] += 1
                total_images += 1

print("===== KÉPTULAJDONSÁGOK =====")
print(f"Vizsgált képek: {total_images}")

print("\n===== KÉPMÉRETEK =====")
for size, count in image_sizes.most_common():
    print(f"{size[0]}x{size[1]}: {count} kép")

print("\n===== KÉPMÓDOK =====")
for mode, count in image_modes.items():
    print(f"{mode}: {count} kép")