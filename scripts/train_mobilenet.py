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
from mobilenet_model import TrashMobileNetV2

# 1. ELÉRÉSI ÚTVONALAK
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"

CHECKPOINT_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_mobilenet_head.pth"

# 2. TANÍTÁSI BEÁLLÍTÁSOK
NUM_EPOCHS = 20

EARLY_STOPPING_PATIENCE = 6

# 3. CSV MEGKERESÉSE
def find_csv(split_dir):

    csv_files = list(split_dir.glob("*.csv"))

    if len(csv_files) != 1:
        raise FileNotFoundError(
            f"Pontosan egy CSV-fájl szükséges: {split_dir}"
        )

    return csv_files[0]

# 4. ESZKÖZ KIVÁLASZTÁSA
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 5. DATASET ÉS DATALOADER
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

# 6. Mobilenetv2 modell
model = TrashMobileNetV2(
    num_classes=len(CLASS_NAMES)
).to(device)

# 7. LOSS, OPTIMIZER ÉS SCHEDULER
criterion = nn.CrossEntropyLoss()

# Csak a tanítható paramétereket optimalizáljuk
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

# 8. ALAPINFORMÁCIÓK
total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("===== MOBILENETV2 TANÍTÁS ELŐKÉSZÍTÉSE =====")
print(f"Eszköz: {device}")
print(f"Tanítóképek száma: {len(train_dataset)}")
print(f"Validációs képek száma: {len(valid_dataset)}")
print(f"Tanító batch-ek száma: {len(train_loader)}")
print(f"Validációs batch-ek száma: {len(valid_loader)}")
print(f"Összes paraméter: {total_params:,}")
print(f"Tanítható paraméter: {trainable_params:,}")
print(f"Osztályok: {CLASS_NAMES}")

# 9. Tanítási függvény
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    # A befagyasztott feature extractor BatchNorm
    # rétegeinek futó statisztikáit sem módosítjuk
    model.network.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

# 10. Validációs függvény
def validate_one_epoch(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

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

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

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

# 11. Tanítás
best_val_f1 = -1.0
best_epoch = 0

epochs_without_improvement = 0

print("\n===== MOBILENETV2 TANÍTÁS INDUL =====")

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

    # Learning rate módosítása
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

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

    print(f"Learning Rate: {current_lr:.6f}")

    # Legjobb modell mentése Macro F1 alapján
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print("Új legjobb modell mentve!")

    else:

        epochs_without_improvement += 1

        print(
            f"Nincs javulás: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

    # Early Stopping
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEarly Stopping aktiválva!")
        print("A validációs Macro F1 már nem javult.")

        break

print("\n===== MOBILENETV2 TANÍTÁS BEFEJEZVE =====")
print(f"Legjobb epoch: {best_epoch}")
print(f"Legjobb validációs Macro F1: {best_val_f1:.4f}")
print(f"Modell mentési helye: {BEST_MODEL_PATH}")