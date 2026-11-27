import torch
import torch.nn as nn
from torchvision import models

class TrashMobileNetV2(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        # Előre betanított mobilenetv2 betöltése
        self.network = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # A konvolúciós rétegek súlyainak befagyasztása
        for param in self.network.features.parameters():
            param.requires_grad = False

        # Az eredeti osztályozó bemeneti méretének lekérdezése
        in_features = self.network.classifier[1].in_features

        # Az eredeti 1000 osztály helyett 6 osztály
        self.network.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.network(x)

# Modell ellenőrzése
if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TrashMobileNetV2(num_classes=6).to(device)

    model.eval()

    # Két darab 224x224-es RGB kép szimulálása
    sample_input = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(sample_input)

    # Paraméterek száma
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("===== MOBILENETV2 MODELL ELLENŐRZÉS =====")

    print(f"Eszköz: {device}")
    print(f"Bemenet alakja: {sample_input.shape}")
    print(f"Kimenet alakja: {output.shape}")

    print(f"Összes paraméter: {total_params:,}")
    print(f"Tanítható paraméter: {trainable_params:,}")

    print("\nMobileNetV2 sikeresen betöltve!")