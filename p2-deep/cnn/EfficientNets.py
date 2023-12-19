import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights


"""
Implementations of EfficientNets with custom dimensions for input and output

| Model          | Parameters  | Acc 1  | Acc 5  |
|-----------------|------------|--------|--------|
| efficientnet_b0 | ~5.3M      | 77.692 | 93.532 |

https://pytorch.org/vision/main/search.html?q=efficientnet&check_keywords=yes&area=default#
"""


class EfficientNetB0(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = False):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        
        self.cnn = nn.Sequential(
                preconv,
                *list(efficientnet.children())[:-1]
            )
        
        cnn_out_dim = self._get_cnn_out_dim()

        self.regressor = nn.Sequential(
            nn.Linear(cnn_out_dim + 1, 1024), # + 1 for angle
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

        # self.feature_param = nn.Parameter(torch.tensor(1.0))
        # self.angle_param = nn.Parameter(torch.tensor(100.0))

    def _get_cnn_out_dim(self):
        out = self.cnn(torch.zeros(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)
        features = self.cnn(img)
        # Flatten all dimensions after batch
        features = torch.flatten(features, 1)
        # # Weighting
        # features = features * self.feature_param
        # angle = angle * self.angle_param
        # Concatenate angle to features (batch_size, 1280 + 1) --> feautures.shape = (batch_size, 1281)
        features = torch.cat((features, angle), dim=1)
        return self.regressor(features)


class EfficientNetV2S(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = True, batch_norm: bool = True, activation: str = "PReLU", dropout: float = 0.2, layer_dims: list = [1024, 512, 256, 128, 32]):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        
        self.cnn = nn.Sequential(
                preconv,
                *list(efficientnet.children())[:-1]
            )
        
        cnn_out_dim = self._get_cnn_out_dim()

        layers = []
        for i in range(len(layer_dims)):
            if i == 0:
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(cnn_out_dim + 1, layer_dims[i]))
            else:
                layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i]))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(layer_dims[-1], output_dim))
        
        self.regressor = nn.Sequential(*layers)

    def _get_cnn_out_dim(self):
        out = self.cnn(torch.zeros(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)
        features = self.cnn(img)
        # Flatten all dimensions after batch
        features = torch.flatten(features, 1)
        # Concatenate angle to features (batch_size, 1280 + 1) --> feautures.shape = (batch_size, 1281)
        features = torch.cat((features, angle), dim=1) 
        return self.regressor(features)


def get_activation(activation: str):
        if activation == "ELU":
            return nn.ELU()
        elif activation == "ReLU":
            return nn.ReLU()
        elif activation == "LeakyReLU":
            return nn.LeakyReLU()
        elif activation == "Tanh":
            return nn.Tanh()
        elif activation == "SiLU":
            return nn.SiLU()
        elif activation == "PReLU":
            return nn.PReLU()
        else:
            raise ValueError("Activation function not supported")
