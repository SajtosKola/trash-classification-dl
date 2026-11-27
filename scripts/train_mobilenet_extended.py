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

# 1. Elérési útvonalak
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"

CHECKPOINT_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Korábbi, 20 epochos modell
INITIAL_MODEL_PATH = CHECKPOINT_DIR / "best_mobilenet_head.pth"

# Továbbtanítás során elért legjobb modell
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_mobilenet_head_extended.pth"

# 2. Tanítási beállítások
START_EPOCH = 20

MAX_ADDITIONAL_EPOCHS = 80

EARLY_STOPPING_PATIENCE = 15

INITIAL_LR = 0.001

# 3. CSV megkeresése
def find_csv(split_dir):

    csv_files = list(split_dir.glob("*.csv"))

    if len(csv_files) != 1:
        raise FileNotFoundError(
            f"Pontosan egy CSV-fájl szükséges: {split_dir}"
        )

    return csv_files[0]

# 4. Eszköz kiválasztása
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 5. Dataset és dataloader
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

# 6. Modell betöltése
model = TrashMobileNetV2(
    num_classes=len(CLASS_NAMES)
).to(device)

if not INITIAL_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"A korábbi modell nem található: {INITIAL_MODEL_PATH}"
    )

checkpoint = torch.load(
    INITIAL_MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

print("Korábbi MobileNetV2 modell sikeresen betöltve!")

# 7. Loss, optimizer és cheduler
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=INITIAL_LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# 8. Alapinfók
print("\n===== MOBILENETV2 TOVÁBBTANÍTÁS ELŐKÉSZÍTÉSE =====")

print(f"Eszköz: {device}")
print(f"Tanítóképek száma: {len(train_dataset)}")
print(f"Validációs képek száma: {len(valid_dataset)}")

print(f"Tanító batch-ek száma: {len(train_loader)}")
print(f"Validációs batch-ek száma: {len(valid_loader)}")

print(f"Kiinduló epoch: {START_EPOCH}")
print(f"Maximális epoch: {START_EPOCH + MAX_ADDITIONAL_EPOCHS}")
print(f"Kezdő learning rate: {INITIAL_LR}")

# 9. Tanítási függvény
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    # A befagyasztott feature extractor BatchNorm
    # rétegeinek statisztikáit változatlanul hagyjuk.
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

# 11. Kiinduló modell kiértékelése
start_val_loss, start_val_accuracy, best_val_f1 = validate_one_epoch(
    model,
    valid_loader,
    criterion,
    device
)

best_epoch = START_EPOCH

# A kiinduló modellt is megőrizzük,
# így akkor is lesz mentett modell, ha nem javul tovább.
torch.save(
    model.state_dict(),
    BEST_MODEL_PATH
)

print("\n===== KIINDULÓ MODELL EREDMÉNYEI =====")

print(f"Valid Loss: {start_val_loss:.4f}")
print(f"Valid Accuracy: {start_val_accuracy * 100:.2f}%")
print(f"Valid Macro F1: {best_val_f1:.4f}")

# 12. Továbbtanítás
epochs_without_improvement = 0

print("\n===== MOBILENETV2 TOVÁBBTANÍTÁS INDUL =====")

for additional_epoch in range(1, MAX_ADDITIONAL_EPOCHS + 1):

    current_epoch = START_EPOCH + additional_epoch

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

    # Learning rate módosítása a validációs Loss alapján
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch [{current_epoch}/{START_EPOCH + MAX_ADDITIONAL_EPOCHS}]")

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
        best_epoch = current_epoch

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

print("\n===== TOVÁBBTANÍTÁS BEFEJEZVE =====")
print(f"Legjobb epoch: {best_epoch}")
print(f"Legjobb validációs Macro F1: {best_val_f1:.4f}")
print(f"Modell mentési helye: {BEST_MODEL_PATH}")