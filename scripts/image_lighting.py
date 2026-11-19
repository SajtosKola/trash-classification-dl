from pathlib import Path
import csv
from PIL import Image, ImageStat
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

SPLITS = ["train", "valid", "test"]

brightness_values = []
contrast_values = []

for split in SPLITS:
    folder = DATA_ROOT / split
    csv_file = list(folder.glob("*.csv"))[0]

    with open(csv_file, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = folder / row["filename"]

            with Image.open(image_path) as img:
                grayscale = img.convert("L")
                stats = ImageStat.Stat(grayscale)

                brightness_values.append(stats.mean[0])
                contrast_values.append(stats.stddev[0])

print("===== FÉNYERŐ ÉS KONTRASZT =====")
print(f"Vizsgált képek: {len(brightness_values)}")

print("\nÁtlagos fényerő:")
print(f"{sum(brightness_values) / len(brightness_values):.2f} / 255")

print("\nÁtlagos kontraszt (fényerő szórása):")
print(f"{sum(contrast_values) / len(contrast_values):.2f}")

print("\nFényerő tartománya:")
print(f"{min(brightness_values):.2f} - {max(brightness_values):.2f}")

# Diagramok
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(brightness_values, bins=30, edgecolor="black")
axes[0].set_title("Képek átlagos fényereje")
axes[0].set_xlabel("Fényerő (0-255)")
axes[0].set_ylabel("Képek száma")

axes[1].hist(contrast_values, bins=30, edgecolor="black")
axes[1].set_title("Képek kontrasztja")
axes[1].set_xlabel("Fényerő szórása")
axes[1].set_ylabel("Képek száma")

plt.tight_layout()

output_dir = PROJECT_ROOT / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "image_lighting.png"
plt.savefig(output_path, dpi=150)

print(f"\nDiagram elmentve: {output_path}")

plt.show()