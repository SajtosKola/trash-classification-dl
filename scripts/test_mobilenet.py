import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
import csv
from dataset import (
    TrashDataset,
    val_test_transform,
    CLASS_NAMES
)
from mobilenet_model import TrashMobileNetV2


# Elérési útvonalak
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

TEST_DIR = DATA_ROOT / "test"

CHECKPOINT_DIR = PROJECT_ROOT / "models"
MODEL_PATH = CHECKPOINT_DIR / "best_mobilenet_finetuned_extended.pth"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# CSV megkeresése
def find_csv(split_dir):

    csv_files = list(split_dir.glob("*.csv"))

    if len(csv_files) != 1:
        raise FileNotFoundError(
            f"Pontosan egy CSV-fájl szükséges: {split_dir}"
        )

    return csv_files[0]


# Eszköz kiválasztása
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Tesztadathalmaz betöltése
test_dataset = TrashDataset(
    csv_file=find_csv(TEST_DIR),
    image_dir=TEST_DIR,
    transform=val_test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# Modell betöltése
model = TrashMobileNetV2(
    num_classes=len(CLASS_NAMES)
).to(device)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"A modell nem található: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()


# Alapinformációk
print("===== MOBILENETV2 TESZTELÉS =====")
print(f"Eszköz: {device}")
print(f"Tesztképek száma: {len(test_dataset)}")
print(f"Batch-ek száma: {len(test_loader)}")
print(f"Osztályok: {CLASS_NAMES}")
print(f"Betöltött modell: {MODEL_PATH.name}")


# Tesztelés
criterion = torch.nn.CrossEntropyLoss()

all_predictions = []
all_labels = []

running_loss = 0.0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        total += batch_size

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

test_loss = running_loss / total


# Mérőszámok kiszámítása
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    average="weighted",
    zero_division=0
)


# Részletes osztályozási jelentés
report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n===== TESZTEREDMÉNYEK =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print("\n===== CLASSIFICATION REPORT =====")
print(report)

# Confusion Matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES)))
)

print("\n===== CONFUSION MATRIX =====")
print(cm)


# Eredmények mentése szöveges fájlba
report_path = REPORTS_DIR / "mobilenet_test_report.txt"

report_content = f"""
MOBILENETV2 TESZTEREDMÉNYEK

Modell: {MODEL_PATH.name}
Tesztképek száma: {len(test_dataset)}

Test Loss: {test_loss:.4f}
Accuracy: {accuracy * 100:.2f}%

Macro Precision: {precision:.4f}
Macro Recall: {recall:.4f}
Macro F1: {macro_f1:.4f}
Weighted F1: {weighted_f1:.4f}

CLASSIFICATION REPORT

{report}

CONFUSION MATRIX

{cm}
"""

report_path.write_text(
    report_content,
    encoding="utf-8"
)


# Confusion Matrix mentése CSV formátumban
cm_path = REPORTS_DIR / "mobilenet_confusion_matrix.csv"

with open(cm_path, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.writer(file)

    writer.writerow(["Actual / Predicted"] + CLASS_NAMES)

    for class_name, row in zip(CLASS_NAMES, cm):
        writer.writerow([class_name] + row.tolist())


# Confusion Matrix vizualizálása
fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(cm, interpolation="nearest")

fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(len(CLASS_NAMES)),
    yticks=np.arange(len(CLASS_NAMES)),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted label",
    ylabel="True label",
    title="MobileNetV2 - Confusion Matrix"
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor"
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

fig.tight_layout()

confusion_path = FIGURES_DIR / "mobilenet_confusion_matrix.png"

fig.savefig(
    confusion_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("\n===== TESZTELÉS BEFEJEZVE =====")
print(f"Szöveges jelentés: {report_path}")
print(f"Confusion Matrix CSV: {cm_path}")
print(f"Confusion Matrix ábra: {confusion_path}")