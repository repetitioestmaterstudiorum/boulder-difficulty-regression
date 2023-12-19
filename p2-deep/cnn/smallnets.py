import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights, mobilenet_v2, MobileNet_V2_Weights, mnasnet0_75, MNASNet0_75_Weights, shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights


class shufflenetx1(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = True):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        shufflenet = shufflenet_v2_x1_5(weights=ShuffleNet_V2_X1_5_Weights.DEFAULT if pretrained else None)
        
        self.cnn = nn.Sequential(
                preconv,
                *list(shufflenet.children())[:-1]
            )
        
        cnn_out_dim = self._get_cnn_out_dim()

        # self.regressor = nn.Sequential(
        #     nn.Linear(cnn_out_dim + 1, output_dim) # + 1 for angle
        # )

        self.regressor = nn.Sequential(
            nn.Linear(cnn_out_dim + 1, 1024), # + 1 for angle
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def _get_cnn_out_dim(self):
        out = self.cnn(torch.zeros(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)
        features = self.cnn(img)
        features = torch.flatten(features, 1) # Flatten all dimensions after batch
        features = torch.cat((features, angle), dim=1) # Concatenate angle to features
        return self.regressor(features)
    

class shufflenetx2(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = True):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        shufflenet = shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.DEFAULT if pretrained else None)
        
        self.cnn = nn.Sequential(
            preconv,
            *list(shufflenet.children())[:-1]
        )
        
        cnn_out_dim = self._get_cnn_out_dim()

        self.regressor = nn.Sequential(
            nn.Linear(cnn_out_dim + 1, output_dim) # + 1 for angle
        )

    def _get_cnn_out_dim(self):
        out = self.cnn(torch.zeros(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)
        features = self.cnn(img)
        features = torch.flatten(features, 1) # Flatten all dimensions after batch
        features = torch.cat((features, angle), dim=1) # Concatenate angle to features
        return self.regressor(features)
    

class mobilenet(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = True):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
        
        self.cnn = nn.Sequential(
                preconv,
                *list(mobilenet.children())[:-1]
            )
        
        cnn_out_dim = self._get_cnn_out_dim()

        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cnn_out_dim + 1, output_dim) # + 1 for angle
        )

    def _get_cnn_out_dim(self):
        out = self.cnn(torch.zeros(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)
        features = self.cnn(img)
        features = torch.flatten(features, 1) # Flatten all dimensions after batch
        features = torch.cat((features, angle), dim=1) # Concatenate angle to features)
        return self.regressor(features)
    

class mnasnet(nn.Module):
    def __init__(self, img_dim: int, output_dim: int, pretrained: bool = True):
        super().__init__()
        self.img_dim = img_dim
        
        preconv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        mnasnet = mnasnet0_75(weights=MNASNet0_75_Weights.DEFAULT if pretrained else None)
        
        self.cnn = nn.Sequential(
                preconv,
                *list(mnasnet.children())[:-1]
            )
        
        cnn_out_dim = self._get_cnn_out_dim()

        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cnn_out_dim + 1, output_dim) # + 1 for angle
        )

    def _get_cnn_out_dim(self):
        out = self.cnn(torch.zeros(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)
        features = self.cnn(img)
        features = torch.flatten(features, 1) # Flatten all dimensions after batch
        features = torch.cat((features, angle), dim=1) # Concatenate angle to features
        return self.regressor(features)
