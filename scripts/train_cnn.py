import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
from dataset import (
    TrashDataset,
    train_transform,
    val_test_transform,
    CLASS_NAMES
)

from model import TrashCNN

# Projekt és adatok elérési útvonala
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"


# CSV-fájl megkeresése az adott mappában
def find_csv(split_dir):
    csv_files = list(split_dir.glob("*.csv"))

    if len(csv_files) != 1:
        raise FileNotFoundError(
            f"Pontosan egy CSV-fájl szükséges: {split_dir}"
        )

    return csv_files[0]

# GPU vagy CPU kiválasztása
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Dataset objektumok létrehozása
train_dataset = TrashDataset(
    csv_file=find_csv(TRAIN_DIR),
    image_dir=TRAIN_DIR,
    transform=train_transform
)

valid_dataset = TrashDataset(
    csv_file=find_csv(VALID_DIR),
    image_dir=VALID_DIR,
    transform=val_test_transform
)

# DataLoader objektumok
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# CNN modell létrehozása
model = TrashCNN(
    num_classes=len(CLASS_NAMES)
).to(device)


# Loss függvény és optimalizáló
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# Alapinformációk
print("===== CNN TANÍTÁS ELŐKÉSZÍTÉSE =====")
print(f"Eszköz: {device}")
print(f"Tanítóképek száma: {len(train_dataset)}")
print(f"Validációs képek száma: {len(valid_dataset)}")
print(f"Tanító batch-ek száma: {len(train_loader)}")
print(f"Validációs batch-ek száma: {len(valid_loader)}")
print(f"Osztályok: {CLASS_NAMES}")


# Első batch ellenőrzése
images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)

print("\n===== BATCH ELLENŐRZÉS =====")
print(f"Képek alakja: {images.shape}")
print(f"Címkék alakja: {labels.shape}")
print(f"Képek eszköze: {images.device}")
print(f"Címkék eszköze: {labels.device}")

def train_one_epoch(model, loader, criterion, optimizer, device):

    # A modell tanítási módba állítása
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Korábbi gradiensek törlése
        optimizer.zero_grad(set_to_none=True)

        # Előrehaladás a hálózaton
        outputs = model(images)

        # Hiba kiszámítása
        loss = criterion(outputs, labels)

        # Visszaterjesztés
        loss.backward()

        # Súlyok frissítése
        optimizer.step()

        # Statisztikák gyűjtése
        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def validate_one_epoch(model, loader, criterion, device):

    # A modell kiértékelési módba állítása
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    # Validáció közben nincs szükség gradiensszámításra
    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += batch_size

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_accuracy, epoch_f1

# Tanítási beállítások
NUM_EPOCHS = 40

CHECKPOINT_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_cnn_40epochs.pth"

best_val_f1 = -1.0
best_epoch = 0


print("\n===== CNN TANÍTÁS INDUL =====")

for epoch in range(1, NUM_EPOCHS + 1):

    # Tanítás
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # Validáció
    val_loss, val_accuracy, val_f1 = validate_one_epoch(
        model,
        valid_loader,
        criterion,
        device
    )

    scheduler.step(val_loss)

    print(f"\nEpoch [{epoch}/{NUM_EPOCHS}]")

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Valid Loss: {val_loss:.4f} | "
        f"Valid Accuracy: {val_accuracy * 100:.2f}% | "
        f"Valid Macro F1: {val_f1:.4f}"
    )

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Learning Rate: {current_lr:.6f}")

    # Legjobb modell mentése Macro F1 alapján
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        print("Új legjobb modell mentve!")

print("\n===== TANÍTÁS BEFEJEZVE =====")
print(f"Legjobb epoch: {best_epoch}")
print(f"Legjobb validációs Macro F1: {best_val_f1:.4f}")
print(f"Modell mentési helye: {BEST_MODEL_PATH}")