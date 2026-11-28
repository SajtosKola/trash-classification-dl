
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# Elérési útvonalak
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Modellek összesített teszteredményei
models = {
    "Random Baseline": {
        "Accuracy": 17.62,
        "Macro Precision": 17.13,
        "Macro Recall": 17.99,
        "Macro F1": 17.31,
        "Weighted F1": 17.46
    },
    "Saját CNN": {
        "Accuracy": 71.72,
        "Macro Precision": 72.06,
        "Macro Recall": 67.56,
        "Macro F1": 67.34,
        "Weighted F1": 70.43
    },
    "MobileNetV2": {
        "Accuracy": 86.48,
        "Macro Precision": 84.83,
        "Macro Recall": 83.61,
        "Macro F1": 84.08,
        "Weighted F1": 86.30
    }
}


# Osztályonkénti F1-score értékek
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

class_f1 = {
    "Random Baseline": [
    0.2051, 0.1647, 0.1190,
    0.1099, 0.2917, 0.1481
    ],
    "Saját CNN": [
        0.9231, 0.6833, 0.7222,
        0.8214, 0.5455, 0.3448
    ],
    "MobileNetV2": [
        0.9351, 0.8989, 0.9114,
        0.8696, 0.8352, 0.5946
    ]
}


# Összesített eredmények kiírása
print("===== MODELLEK ÖSSZEHASONLÍTÁSA =====")

for model_name, metrics in models.items():

    print(f"\n{model_name}")

    for metric_name, value in metrics.items():

        print(f"{metric_name}: {value:.2f}%")


# Eredmények mentése CSV-fájlba
csv_path = REPORTS_DIR / "model_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Accuracy (%)",
        "Macro Precision (%)",
        "Macro Recall (%)",
        "Macro F1 (%)",
        "Weighted F1 (%)"
    ])

    for model_name, metrics in models.items():

        writer.writerow([
            model_name,
            *metrics.values()
        ])


# Accuracy és Macro F1 összehasonlítása
model_names = list(models.keys())

accuracy_values = [
    models[name]["Accuracy"] for name in model_names
]

macro_f1_values = [
    models[name]["Macro F1"] for name in model_names
]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2,
    accuracy_values,
    width,
    label="Accuracy"
)

bars2 = ax.bar(
    x + width / 2,
    macro_f1_values,
    width,
    label="Macro F1"
)

ax.set_ylabel("Teljesítmény (%)")
ax.set_title("Modellek összehasonlítása")
ax.set_xticks(x)
ax.set_xticklabels(model_names)

ax.set_ylim(0, 100)

ax.legend()
ax.grid(axis="y", alpha=0.3)

ax.bar_label(bars1, fmt="%.1f%%", padding=3)
ax.bar_label(bars2, fmt="%.1f%%", padding=3)

fig.tight_layout()

comparison_path = FIGURES_DIR / "model_comparison.png"

fig.savefig(
    comparison_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# Osztályonkénti F1-score összehasonlítása
x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

for i, (model_name, values) in enumerate(class_f1.items()):

    ax.bar(
        x + (i - 1) * width,
        np.array(values) * 100,
        width,
        label=model_name
    )

ax.set_ylabel("F1-score (%)")
ax.set_title("Osztályonkénti modell-összehasonlítás")

ax.set_xticks(x)
ax.set_xticklabels(class_names)

ax.set_ylim(0, 110)

ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

class_comparison_path = (
    FIGURES_DIR / "class_f1_comparison.png"
)

fig.savefig(
    class_comparison_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# Befejezés
print("\n===== ÖSSZEHASONLÍTÁS BEFEJEZVE =====")

print(f"Eredménytábla: {csv_path}")
print(f"Összesített diagram: {comparison_path}")
print(f"Osztályonkénti diagram: {class_comparison_path}")