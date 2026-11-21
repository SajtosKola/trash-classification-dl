from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import TrashDataset, train_transform, CLASS_NAMES
from model import TrashCNN

# Projekt gyökérmappája
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tanítóadatok
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
csv_files = list(TRAIN_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise FileNotFoundError("Nem található pontosan egy CSV-fájl!")

# Dataset és DataLoader
dataset = TrashDataset(
    csv_file=csv_files[0],
    image_dir=TRAIN_DIR,
    transform=train_transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# GPU vagy CPU kiválasztása
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Modell létrehozása
model = TrashCNN(num_classes=len(CLASS_NAMES))
model = model.to(device)
model.eval()

# Egy batch lekérése
images, labels = next(iter(loader))

images = images.to(device)
labels = labels.to(device)

# Előrehaladás a hálózaton
with torch.no_grad():
    outputs = model(images)

# Eredmények
print("===== CNN MODELL TESZT =====")

print("Eszköz:", device)
print("Bemenet alakja:", images.shape)
print("Kimenet alakja:", outputs.shape)

print(
    "Tanítható paraméterek száma:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)

print("\nElső kép osztálypontszámai:")
for class_name, score in zip(CLASS_NAMES, outputs[0]):
    print(f"{class_name}: {score.item():.4f}")