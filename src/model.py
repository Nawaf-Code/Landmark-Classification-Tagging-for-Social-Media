import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        self.features = nn.Sequential(
            
            # Convolutional Block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            # Convolutional Block 2
            nn.Conv2d(16, 32, 3, padding=1),   
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),               
            nn.ReLU(),

            # Convolutional Block 3
            nn.Conv2d(32, 64, 3, padding=1),  
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),               
            nn.ReLU(),

            # Convolutional Block 4
            nn.Conv2d(64, 128, 3, padding=1),  
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),               
            nn.ReLU(),

            # Convolutional Block 5
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),               
            nn.ReLU()
    )

        self.classifier = nn.Sequential(
            nn.Flatten(),                     

            nn.Linear(256*7*7, 1024),         
            nn.Dropout(dropout),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, num_classes),
)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from src.data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)  # Fixed `.next()`

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
