import torch
import torch.nn as nn
class TrashCNN(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        # Konvolúciós rétegek
        self.features = nn.Sequential(

            # 1. blokk: 3 -> 32 csatorna
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 2. blokk: 32 -> 64 csatorna
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 3. blokk: 64 -> 128 csatorna
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # A térbeli méretek összefoglalása
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Osztályozó rétegek
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x