
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import csv

from pathlib import Path

from dataset import TrashDataset, val_test_transform, CLASS_NAMES
from mobilenet_model import TrashMobileNetV2


# Elérési útvonalak
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

TEST_DIR = DATA_ROOT / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_mobilenet_finetuned_extended.pth"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Képnormalizálás értékei
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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


# Helyes és hibás példák kiválasztása osztályonként
selected_samples = {
    i: {
        "helyes": None,
        "hibas": None
    }
    for i in range(len(CLASS_NAMES))
}

print("===== TESZTKÉPEK KIVÁLASZTÁSA =====")

for index in range(len(test_dataset)):

    image, label = test_dataset[index]
    label = int(label)

    with torch.no_grad():

        inputs = image.unsqueeze(0).to(device)

        outputs = model(inputs)

        probabilities = torch.softmax(outputs, dim=1)

        prediction = outputs.argmax(dim=1).item()

        confidence = probabilities[0, prediction].item()

    # Megnézzük, hogy helyes vagy hibás volt-e az előrejelzés
    if prediction == label:
        category = "helyes"
    else:
        category = "hibas"

    if selected_samples[label][category] is None:

        selected_samples[label][category] = {
            "index": index,
            "image": image.clone().detach(),
            "label": label,
            "prediction": prediction,
            "confidence": confidence
        }

    # Leállunk, ha minden kategóriához van mindkét példánk
    if all(
        all(value is not None for value in samples.values())
        for samples in selected_samples.values()
    ):
        break


# Kiválasztott példák kiírása
for class_index, samples in selected_samples.items():

    print(f"\n{CLASS_NAMES[class_index]}")

    for category, sample in samples.items():

        if sample is None:
            print(f"{category}: nem található példa")
        else:
            print(
                f"{category}: {sample['index']} | "
                f"Előrejelzés: {CLASS_NAMES[sample['prediction']]}"
            )


# Grad-CAM aktivációk és gradiensek tárolása
activations = {}
gradients = {}


# Forward hook az utolsó konvolúciós blokkra
def save_activation(module, inputs, output):

    activations["value"] = output

    output.register_hook(
        lambda grad: gradients.update({
            "value": grad.detach()
        })
    )


# A MobileNetV2 utolsó konvolúciós blokkjának kiválasztása
target_layer = model.network.features[-1]

hook_handle = target_layer.register_forward_hook(
    save_activation
)


# Grad-CAM hőtérkép kiszámítása
def generate_gradcam(image, class_index):

    activations.clear()
    gradients.clear()

    inputs = image.unsqueeze(0).to(device)

    # A bemenethez is engedélyezzük a gradiensszámítást
    inputs.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    outputs = model(inputs)

    # Az előrejelzett osztály pontszámának visszaterjesztése
    score = outputs[0, class_index]

    score.backward()

    activation = activations["value"].detach()
    gradient = gradients["value"]

    # A gradiensek térbeli átlagolása
    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    # A hőtérkép kiszámítása
    cam = torch.relu(
        (weights * activation).sum(dim=1, keepdim=True)
    )

    # A hőtérkép méretének visszaállítása
    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze()

    # Normalizálás 0 és 1 közé
    cam = cam - cam.min()

    cam = cam / (cam.max() + 1e-8)

    return cam.cpu().numpy()


# Normalizálás visszafordítása a megjelenítéshez
def denormalize_image(image):

    image = image.cpu() * STD + MEAN

    image = torch.clamp(image, 0, 1)

    return image.permute(1, 2, 0).numpy()


# Grad-CAM ábra elkészítése
fig, axes = plt.subplots(
    len(CLASS_NAMES),
    4,
    figsize=(16, 21)
)

fig.suptitle(
    "MobileNetV2 - Grad-CAM hibaanalízis",
    fontsize=16
)

print("\n===== GRAD-CAM ÁBRÁK KÉSZÍTÉSE =====")

# Végigmegyünk mind a hat kategórián
for class_index, class_name in enumerate(CLASS_NAMES):

    samples = selected_samples[class_index]

    # Helyesen osztályozott példa
    correct_sample = samples["helyes"]

    if correct_sample is not None:

        image = correct_sample["image"]

        cam = generate_gradcam(
            image,
            correct_sample["prediction"]
        )

        original_image = denormalize_image(image)

        axes[class_index, 0].imshow(original_image)
        axes[class_index, 0].set_title("Helyes előrejelzés")

        axes[class_index, 1].imshow(original_image)
        axes[class_index, 1].imshow(
            cam,
            cmap="jet",
            alpha=0.4
        )

        axes[class_index, 1].set_title(
            f"Grad-CAM: {CLASS_NAMES[correct_sample['prediction']]}"
        )

    else:

        axes[class_index, 0].set_title("Nincs példa")
        axes[class_index, 1].set_title("Nincs példa")

    # Hibásan osztályozott példa
    incorrect_sample = samples["hibas"]

    if incorrect_sample is not None:

        image = incorrect_sample["image"]

        cam = generate_gradcam(
            image,
            incorrect_sample["prediction"]
        )

        original_image = denormalize_image(image)

        axes[class_index, 2].imshow(original_image)

        axes[class_index, 2].set_title(
            f"Hibás: {class_name} → "
            f"{CLASS_NAMES[incorrect_sample['prediction']]}"
        )

        axes[class_index, 3].imshow(original_image)

        axes[class_index, 3].imshow(
            cam,
            cmap="jet",
            alpha=0.4
        )

        axes[class_index, 3].set_title(
            f"Grad-CAM: {CLASS_NAMES[incorrect_sample['prediction']]}"
        )

    else:

        axes[class_index, 2].set_title("Nincs példa")
        axes[class_index, 3].set_title("Nincs példa")

    # Tengelyek elrejtése
    for ax in axes[class_index]:
        ax.axis("off")

    axes[class_index, 0].set_ylabel(
        class_name,
        fontsize=11
    )

# Hook eltávolítása
hook_handle.remove()

fig.tight_layout(rect=[0, 0, 1, 0.98])

# Ábra mentése
output_path = FIGURES_DIR / "mobilenet_gradcam.png"

fig.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


# Kiválasztott példák adatainak mentése
csv_path = REPORTS_DIR / "gradcam_examples.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Actual class",
        "Example type",
        "Test index",
        "Predicted class",
        "Confidence"
    ])

    for class_index, samples in selected_samples.items():

        for category, sample in samples.items():

            if sample is not None:

                writer.writerow([
                    CLASS_NAMES[class_index],
                    category,
                    sample["index"],
                    CLASS_NAMES[sample["prediction"]],
                    round(sample["confidence"], 4)
                ])

# Befejezés
print("\n===== GRAD-CAM BEFEJEZVE =====")
print(f"Ábra: {output_path}")
print(f"Példák listája: {csv_path}")