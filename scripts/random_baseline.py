from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from dataset import TrashDataset, CLASS_NAMES

# Projekt gyökérmappája
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tesztadatok mappája
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"

# CSV-fájl megkeresése
csv_files = list(TEST_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise FileNotFoundError(
        f"Pontosan egy CSV-fájlt vártam, de ezt találtam: {csv_files}"
    )

# Teszt Dataset létrehozása
test_dataset = TrashDataset(
    csv_file=csv_files[0],
    image_dir=TEST_DIR,
    transform=None
)

# Valódi címkék lekérése
y_true = [label for _, label in test_dataset.samples]

# Véletlenszerű előrejelzések generálása
RANDOM_SEED = 10

rng = np.random.default_rng(RANDOM_SEED)

y_pred = rng.integers(
    low=0,
    high=len(CLASS_NAMES),
    size=len(y_true)
)

# Accuracy kiszámítása
accuracy = accuracy_score(y_true, y_pred)

# Eredmények kiírása
print("===== RANDOM BASELINE =====")

print("Tesztképek száma:", len(y_true))
print("Osztályok száma:", len(CLASS_NAMES))
print("Random seed:", RANDOM_SEED)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Részletes osztályozási jelentés
print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)

# Konfúziós mátrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES)))
)

print("===== CONFUSION MATRIX =====")
print("Sorok: valódi címkék")
print("Oszlopok: előrejelzett címkék")
print(CLASS_NAMES)
print(cm)