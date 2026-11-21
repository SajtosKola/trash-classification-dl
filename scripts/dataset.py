import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Alap transzformációk a tanítóképekhez
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Transzformációk validációs és tesztképekhez
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

class TrashDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = []

        # CSV-fájl beolvasása
        with open(csv_file, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, skipinitialspace=True)

            for row in reader:
                filename = row["filename"]

                # One-hot címke átalakítása egész számmá
                labels = [
                    int(row[class_name])
                    for class_name in CLASS_NAMES
                ]

                if any(label not in (0, 1) for label in labels):
                    raise ValueError(f"Érvénytelen címke: {filename}")

                if sum(labels) != 1:
                    raise ValueError(f"A képhez nem pontosan egy címke tartozik: {filename}")

                label = labels.index(1)

                self.samples.append((filename, label))

        if not self.samples:
            raise ValueError("A CSV-fájl nem tartalmaz képadatokat.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]

        image_path = self.image_dir / filename

        # Kép megnyitása RGB formátumban
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        # Előfeldolgozás alkalmazása
        if self.transform:
            image = self.transform(image)

        return image, label