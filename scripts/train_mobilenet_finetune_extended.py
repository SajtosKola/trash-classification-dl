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

# Elérési útvonalak
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"

CHECKPOINT_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_MODEL_PATH = (
    CHECKPOINT_DIR / "best_mobilenet_finetuned.pth"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "best_mobilenet_finetuned_extended.pth"
)


# Tanítási beállítások
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 8

FEATURE_LR = 0.000005
CLASSIFIER_LR = 0.00005


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


# Dataset és DataLoader létrehozása
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


# Modell betöltése
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


# Feature rétegek feloldása
for param in model.network.features.parameters():
    param.requires_grad = False

for param in model.network.features[17:].parameters():
    param.requires_grad = True

for param in model.network.classifier.parameters():
    param.requires_grad = True


# Loss, optimizer és scheduler
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    [
        {
            "params": model.network.features[17:].parameters(),
            "lr": FEATURE_LR
        },
        {
            "params": model.network.classifier.parameters(),
            "lr": CLASSIFIER_LR
        }
    ]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=0.000001
)


# Alapinformációk
total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("MobileNetV2 továbbtanítás előkészítése")

print(f"Eszköz: {device}")
print(f"Tanítóképek: {len(train_dataset)}")
print(f"Validációs képek: {len(valid_dataset)}")

print(f"Összes paraméter: {total_params:,}")
print(f"Tanítható paraméter: {trainable_params:,}")

print(f"Feature LR: {FEATURE_LR}")
print(f"Classifier LR: {CLASSIFIER_LR}")


# Tanítási függvény
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    # A befagyasztott rétegek Batch Normalization statisztikái ne változzanak
    model.network.features[:17].eval()

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


# Validációs függvény
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


# Kiinduló modell kiértékelése
start_loss, start_accuracy, best_val_f1 = validate_one_epoch(
    model,
    valid_loader,
    criterion,
    device
)

best_epoch = "Kiinduló modell"
epochs_without_improvement = 0

# A kiinduló modellt is elmentjük
torch.save(
    model.state_dict(),
    BEST_MODEL_PATH
)

print("\nKiinduló modell eredményei")

print(f"Valid Loss: {start_loss:.4f}")
print(f"Valid Accuracy: {start_accuracy * 100:.2f}%")
print(f"Valid Macro F1: {best_val_f1:.4f}")


# Továbbtanítás
print("\nMobileNetV2 továbbtanítás indul")

for epoch in range(1, NUM_EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_accuracy, val_f1 = validate_one_epoch(
        model,
        valid_loader,
        criterion,
        device
    )

    scheduler.step(val_loss)

    feature_lr = optimizer.param_groups[0]["lr"]
    classifier_lr = optimizer.param_groups[1]["lr"]

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

    print(f"Feature LR: {feature_lr:.7f}")
    print(f"Classifier LR: {classifier_lr:.7f}")

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

    # Early Stopping ellenőrzése
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEarly Stopping aktiválva!")
        break


print("\nTovábbtanítás befejezve")
print(f"Legjobb epoch: {best_epoch}")
print(f"Legjobb validációs Macro F1: {best_val_f1:.4f}")
print(f"Modell mentési helye: {BEST_MODEL_PATH}")