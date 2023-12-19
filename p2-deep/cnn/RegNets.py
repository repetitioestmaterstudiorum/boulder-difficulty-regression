import torch
import torch.nn as nn
from torchvision.models.regnet import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights, regnet_y_400mf, RegNet_Y_400MF_Weights


"""
Implementations of RegNets with custom dimensions for input and output

| Model          | Parameters | Acc 1  | Acc 5  |
|----------------|------------|--------|--------|
| regnet_y_400mf | ~4.3M      | 74.046 | 91.716 |
| regnet_y_1_6gf | ~12M       | 77.950 | 93.966 |

https://pytorch.org/vision/main/search.html?q=regnet&check_keywords=yes&area=default
"""


class RegNetY1_6GF(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = False):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        regnet = regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.IMAGENET1K_V2 if pretrained else None)

        self.cnn = nn.Sequential(
                preconv,
                *list(regnet.children())[:-1]
        )

        self.regressor = nn.Sequential(
            nn.Linear(888 + 1, 1024), # + 1 for angle
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)
        features = self.cnn(img)
        # Flatten all dimensions after batch
        features = torch.flatten(features, 1)
        # Concatenate angle to features (batch_size, 888 + 1) --> feautures.shape = (batch_size, 889)
        features = torch.cat((features, angle), dim=1) 
        return self.regressor(features)


class RegNetY400MF(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = False):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        regnet = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2 if pretrained else None)

        self.cnn = nn.Sequential(
                preconv,
                *list(regnet.children())[:-1]
            )

        self.regressor = nn.Sequential(
            nn.Linear(440 + 1, 1024), # + 1 for angle
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)
        features = self.cnn(img)
        # Flatten all dimensions after batch
        features = torch.flatten(features, 1)
        # Concatenate angle to features (batch_size, 440 + 1) --> feautures.shape = (batch_size, 441)
        features = torch.cat((features, angle), dim=1) 
        return self.regressor(features)
