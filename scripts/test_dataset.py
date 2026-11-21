import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import TrashDataset, train_transform, CLASS_NAMES

# Projekt gyökérmappájának meghatározása
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tanítóadatok mappája
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"

# CSV-fájl megkeresése
csv_files = list(TRAIN_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise FileNotFoundError(
        f"Pontosan egy CSV-fájlt vártam, de ezt találtam: {csv_files}"
    )

# Dataset létrehozása
train_dataset = TrashDataset(
    csv_file=csv_files[0],
    image_dir=TRAIN_DIR,
    transform=train_transform
)

print("===== DATASET TESZT =====")
print("Tanítóképek száma:", len(train_dataset))

# Első kép lekérése
image, label = train_dataset[0]

print("\n===== ELSŐ KÉP =====")
print("Tensor alakja:", image.shape)
print("Címke indexe:", label)
print("Címke neve:", CLASS_NAMES[label])
print("Tensor adattípusa:", image.dtype)
print(
    "Normalizált értéktartomány:",
    f"{image.min().item():.3f} - {image.max().item():.3f}"
)

# DataLoader létrehozása
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# Első batch lekérése
images, labels = next(iter(train_loader))

print("\n===== DATALOADER TESZT =====")
print("Képek alakja:", images.shape)
print("Címkék alakja:", labels.shape)
print("Címkék adattípusa:", labels.dtype)
print("Első 8 címke:", labels[:8].tolist())

# Batch áthelyezése a GPU-ra
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images = images.to(device)
labels = labels.to(device)

print("Használt eszköz:", images.device)
print("GPU-ra áthelyezett címkék:", labels.device)