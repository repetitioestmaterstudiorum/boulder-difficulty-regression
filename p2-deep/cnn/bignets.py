import torch
import torch.nn as nn
from torchvision.models import convnext_large, ConvNeXt_Large_Weights, regnet_y_32gf, RegNet_Y_32GF_Weights
from transformers import CvtModel, CvtConfig


class cvt(nn.Module):
    def __init__(self, img_dim: int, output_dim: int):
        super().__init__()
        self.img_dim = img_dim
        
        config = CvtConfig(num_channels=1)
        self.model = CvtModel(config)
        
        cnn_out_dim = self._calculate_cnn_out_dim()
        
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            
            nn.Linear(cnn_out_dim + 1, 1024), # + 1 for angle
            nn.BatchNorm1d(1024),
            nn.PReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.PReLU(),

            nn.Linear(64, output_dim),
        )

    def _calculate_cnn_out_dim(self):
        out = self.model(torch.zeros(1, 1, self.img_dim, self.img_dim))
        ftrs = out.last_hidden_state
        ftrs = torch.flatten(ftrs, 1)
        return int(torch.prod(torch.tensor(ftrs.size())))

    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, image channels, height, width)
        output = self.model(img)
        features = output.last_hidden_state
        features = torch.flatten(features, 1)
        features = torch.cat((features, angle), dim=1)
        return self.regressor(features)


class convnextl(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = True):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False) # 1 channel to 3 channel
        
        model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT if pretrained else None)
        
        self.transformer = nn.Sequential(
                preconv,
                *list(model.children())[:-1]
            )
        
        cnn_out_dim = self._get_cnn_out_dim()
        
        self.regressor = nn.Sequential(
            nn.Linear(cnn_out_dim + 1, output_dim) # + 1 for angle
        )

    def _get_cnn_out_dim(self):
        out = self.cnn(torch.zeros(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))

    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, image channels, height, width)
        features = self.transformer(img)
        features = torch.flatten(features, 1) # Flatten all dimensions after batch
        features = torch.cat((features, angle), dim=1) # Concatenate angle to features
        return self.regressor(features)


class regnety32gf(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = True):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False) # 1 channel to 3 channel
        
        model = regnet_y_32gf(weights=RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1 if pretrained else None)
        
        self.transformer = nn.Sequential(
                preconv,
                *list(model.children())[:-1]
            )
        
        cnn_out_dim = self._get_cnn_out_dim()
        
        self.regressor = nn.Sequential(
            nn.Linear(cnn_out_dim + 1, output_dim) # + 1 for angle
        )

    def _get_cnn_out_dim(self):
        out = self.transformer(torch.rand(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))

    def forward(self, sx):
        img = sx[:,1:]
        angle = sx[:,:1]
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, image channels, height, width)
        features = self.transformer(img)
        features = torch.flatten(features, 1) # Flatten all dimensions after batch
        features = torch.cat((features, angle), dim=1) # Concatenate angle to features
        return self.regressor(features)
