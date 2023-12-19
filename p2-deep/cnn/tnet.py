import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
from torchvision.models import shufflenet_v2_x2_0, shufflenet_v2_x1_5, mobilenet_v2

class tnet(nn.Module):
    def __init__(self, img_dim: int, output_dim: int):
        super().__init__()
        self.img_dim = img_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8 x 8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 4 x 4
        )

        cnn_out_dim = self._get_cnn_out_dim()

        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cnn_out_dim + 1, output_dim), # + 1 for angle
        )

        self.skip_cnn = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 4 x 4
        )

    def _get_cnn_out_dim(self):
        out = self.cnn(torch.zeros(1, 1, self.img_dim, self.img_dim))
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim) # (batch_size, channels, height, width)

        features = self.cnn(img)
        skip_features = self.skip_cnn(img)
        features = features + skip_features
        features = torch.flatten(features, 1) # Flatten all dimensions except batch (batch_size, cnn_out_dim)

        x = torch.cat((features, angle), dim=1) # Concatenate angle to features (batch_size, cnn_out_dim + 1)
        return self.regressor(x)


### ------------------- ###


class tnet2(nn.Module):
    def __init__(self, img_dim: int, output_dim: int):
        super().__init__()
        self.img_dim = img_dim

        self.holds = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
        )

        self.holds_s = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=30, dilation=30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d((16, 16)),
        )

        cnn_out_dim = self._get_cnn_out_dim()

        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cnn_out_dim + 1, output_dim), # + 1 for angle
        )

        self.skip_cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
        )

    def _get_cnn_out_dim(self):
        x = torch.zeros(1, 1, self.img_dim, self.img_dim)
        x1 = self.holds(x)
        x2 = self.holds_s(x)
        x = torch.cat((x1, x2), dim=1)
        out = self.cnn(x)
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim)
        holds = self.holds(img)
        holds_s = self.holds_s(img)
        features = torch.cat((holds, holds_s), dim=1)
        features = self.cnn(features)
        skip_features = self.skip_cnn(img)
        features = features + skip_features
        features = torch.flatten(features, 1)
        x = torch.cat((features, angle), dim=1)
        return self.regressor(x)


### ------------------- ###


class tnet3(nn.Module):
    def __init__(self, img_dim: int, output_dim: int):
        super().__init__()
        self.img_dim = img_dim

        self.holds = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(64),
        )
        self.holds_weight = nn.Parameter(torch.tensor(1.0))

        self.holds_s = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=19, dilation=19),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(64),
        )
        self.holds_s_weight = nn.Parameter(torch.tensor(0.5))
        
        self.preconv = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
        )

        shufflenet = shufflenet_v2_x2_0()
        self.cnn = nn.Sequential(*list(shufflenet.children())[:-1])

        cnn_out_dim = self._get_cnn_out_dim()

        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(cnn_out_dim + 1, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, output_dim), # + 1 for angle
        )

        self.skip_cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.skip_cnn_weight = nn.Parameter(torch.tensor(0.2))

    def _get_cnn_out_dim(self):
        x = torch.zeros(1, 1, self.img_dim, self.img_dim)
        x1 = self.holds(x)
        x2 = self.holds_s(x)
        x = torch.cat((x1, x2), dim=1)
        # print('torch.cat x.shape', x.shape)
        x = self.preconv(x)
        # print('preconv x.shape', x.shape)
        out = self.cnn(x)
        # print('cnn out.shape', out.shape)
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim)
        holds = self.holds(img) * self.holds_weight
        holds_s = self.holds_s(img) * self.holds_s_weight
        features = torch.cat((holds, holds_s), dim=1)
        # print('torch.cat features.shape', features.shape)
        features = self.preconv(features)
        # print('preconv features.shape', features.shape)
        features = self.cnn(features)
        # print('cnn features.shape', features.shape)
        skip_features = self.skip_cnn(img) * self.skip_cnn_weight
        features = features + skip_features
        features = torch.flatten(features, 1)
        x = torch.cat((features, angle), dim=1)
        return self.regressor(x)


### ------------------- ###


class tnet4(nn.Module):
    def __init__(self, img_dim: int, output_dim: int):
        super().__init__()
        self.img_dim = img_dim

        self.holds = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(64),
        )
        self.holds_weight = nn.Parameter(torch.tensor(1.0))

        self.holds_s = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=19, dilation=19),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(64),
        )
        self.holds_s_weight = nn.Parameter(torch.tensor(0.2))
        
        self.preconv = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
        )

        shufflenet = shufflenet_v2_x2_0()
        self.cnn = nn.Sequential(*list(shufflenet.children())[:-1])

        cnn_out_dim = self._get_cnn_out_dim()

        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(cnn_out_dim + 1, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, output_dim), # + 1 for angle
        )

        self.skip_cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.skip_cnn_weight = nn.Parameter(torch.tensor(0.1))

    def _get_cnn_out_dim(self):
        x = torch.zeros(1, 1, self.img_dim, self.img_dim)
        x1 = self.holds(x)
        x2 = self.holds_s(x)
        x = torch.cat((x1, x2), dim=1)
        # print('torch.cat x.shape', x.shape)
        x = self.preconv(x)
        # print('preconv x.shape', x.shape)
        out = self.cnn(x)
        # print('cnn out.shape', out.shape)
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim)
        holds = self.holds(img) * self.holds_weight
        holds_s = self.holds_s(img) * self.holds_s_weight
        features = torch.cat((holds, holds_s), dim=1)
        # print('torch.cat features.shape', features.shape)
        features = self.preconv(features)
        # print('preconv features.shape', features.shape)
        features = self.cnn(features)
        # print('cnn features.shape', features.shape)
        skip_features = self.skip_cnn(img) * self.skip_cnn_weight
        features = features + skip_features
        features = torch.flatten(features, 1)
        x = torch.cat((features, angle), dim=1)
        return self.regressor(x)


### ------------------- ###


class tnet5(nn.Module):
    def __init__(self, img_dim: int, output_dim: int):
        super().__init__()
        self.img_dim = img_dim

        self.holds_cnn = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=1),
            nn.SiLU(),

            nn.Sequential(*list(shufflenet_v2_x1_5().children())[:-1]),
        )
        self.holds_weight = nn.Parameter(torch.tensor(1.0))

        self.holds_surroundings_cnn = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=16, dilation=30),
            nn.SiLU(),
            
            nn.Sequential(*list(mobilenet_v2().children())[:-1]),
        )
        self.holds_s_weight = nn.Parameter(torch.tensor(0.2))

        holds_cnn_out_dim = self._get_cnn_out_dim(self.holds_cnn)
        holds_surroundings_cnn_out_dim = self._get_cnn_out_dim(self.holds_surroundings_cnn)

        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(holds_cnn_out_dim + holds_surroundings_cnn_out_dim + 1, 512),  # + 1 for the angle
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, output_dim), # + 1 for angle
        )

    def _get_cnn_out_dim(self, cnn):
        x = torch.zeros(1, 1, self.img_dim, self.img_dim)
        out = cnn(x)
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim)
        holds = self.holds_cnn(img) * self.holds_weight
        holds_surroundings = self.holds_surroundings_cnn(img) * self.holds_s_weight
        holds = torch.flatten(holds, 1)
        holds_surroundings = torch.flatten(holds_surroundings, 1)
        features = torch.cat((holds, holds_surroundings, angle), dim=1)
        return self.regressor(features)


### ------------------- ###


class tnet6(nn.Module):
    def __init__(self, img_dim: int, output_dim: int):
        super().__init__()
        self.img_dim = img_dim

        self.holds_cnn = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=1),
            nn.SiLU(),

            nn.Sequential(*list(shufflenet_v2_x1_5().children())[:-1]),
        )
        self.holds_weight = nn.Parameter(torch.tensor(1.0))

        self.holds_big_cnn = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2, dilation=2),
            nn.SiLU(),
            
            nn.Sequential(*list(mobilenet_v2().children())[:-1]),
        )
        self.holds_s_weight = nn.Parameter(torch.tensor(0.3))

        holds_cnn_out_dim = self._get_cnn_out_dim(self.holds_cnn)
        holds_big_cnn_out_dim = self._get_cnn_out_dim(self.holds_big_cnn)

        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(holds_cnn_out_dim + holds_big_cnn_out_dim + 1, 512),  # + 1 for the angle
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, output_dim), # + 1 for angle
        )

    def _get_cnn_out_dim(self, cnn):
        x = torch.zeros(1, 1, self.img_dim, self.img_dim)
        out = cnn(x)
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim)
        holds = self.holds_cnn(img) * self.holds_weight
        holds_surroundings = self.holds_big_cnn(img) * self.holds_s_weight
        holds = torch.flatten(holds, 1)
        holds_surroundings = torch.flatten(holds_surroundings, 1)
        features = torch.cat((holds, holds_surroundings, angle), dim=1)
        return self.regressor(features)


### ------------------- ###


class tnet6(nn.Module):
    def __init__(self, img_dim: int, output_dim: int):
        super().__init__()
        self.img_dim = img_dim

        self.holds_cnn = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=1),
            nn.SiLU(),

            nn.Sequential(*list(shufflenet_v2_x1_5().children())[:-1]),
        )
        self.holds_weight = nn.Parameter(torch.tensor(1.0))

        self.holds_big_cnn = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.SiLU(),
            
            nn.Sequential(*list(mobilenet_v2().children())[:-1]),
        )
        self.holds_s_weight = nn.Parameter(torch.tensor(0.3))

        holds_cnn_out_dim = self._get_cnn_out_dim(self.holds_cnn)
        holds_big_cnn_out_dim = self._get_cnn_out_dim(self.holds_big_cnn)

        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(holds_cnn_out_dim + holds_big_cnn_out_dim + 1, 512),  # + 1 for the angle
            nn.BatchNorm1d(512),
            nn.PReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),

            nn.Linear(128, output_dim), # + 1 for angle
        )

    def _get_cnn_out_dim(self, cnn):
        x = torch.zeros(1, 1, self.img_dim, self.img_dim)
        out = cnn(x)
        return int(torch.prod(torch.tensor(out.size())))
    
    def forward(self, img, angle):
        img = img.view(-1, 1, self.img_dim, self.img_dim)
        holds = self.holds_cnn(img) * self.holds_weight
        holds_surroundings = self.holds_big_cnn(img) * self.holds_s_weight
        holds = torch.flatten(holds, 1)
        holds_surroundings = torch.flatten(holds_surroundings, 1)
        features = torch.cat((holds, holds_surroundings, angle), dim=1)
        return self.regressor(features)


### ------------------- ###


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, drop_prob):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.se = SqueezeExcitation(out_channels, out_channels // 4)
        self.stochastic_depth = StochasticDepth(drop_prob, "row")

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = self.se(x)
        x = self.stochastic_depth(x)
        if identity.shape == x.shape:
            x += identity
        return x


class tnextnet(nn.Module):
    def __init__(self, img_dim: int, output_dim=1):
        super().__init__()
        self.img_dim = img_dim

        # Holds pathway
        self.holds = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.SiLU(),
            MBConvS(16, 32, kernel_size=3, stride=1, drop_prob=0.05), # 32 x 32
            MBConvS(32, 64, kernel_size=3, stride=2, drop_prob=0.1), # 16 x 16
        )

        # Holds surroundings pathway
        self.holds_surroundings = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=15, dilation=15), # 32 x 32
            nn.SiLU(),
            MBConvS(32, 64, kernel_size=5, stride=2, drop_prob=0.05), # 16 x 16
        )

        # Initialize importance weights for each feature set
        self.holds_weight = nn.Parameter(torch.tensor(1.0))
        self.holds_surroundings_weight = nn.Parameter(torch.tensor(1.0))
        self.angle_weight = nn.Parameter(torch.tensor(1.0))

        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        holds_out_dim = self._get_conv_out_dim(self.holds)
        holds_surroundings_out_dim = self._get_conv_out_dim(self.holds_surroundings)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(holds_out_dim + holds_surroundings_out_dim + 1, output_dim)  # + 1 for the angle
        )

    def _get_conv_out_dim(self, conv_layers):
        # Apply the convolutional layers to a dummy input and get the output shape
        with torch.no_grad():
            out = conv_layers(torch.zeros(1, 1, self.img_dim, self.img_dim))
            # After avgpool, the spatial dimensions become 1x1, so the output size is just the number of channels
            return out.size(1)  # Number of output channels

    def forward(self, img, angle):
        # Process holds and surroundings
        holds = self.avgpool(self.holds(img))
        holds_surroundings = self.avgpool(self.holds_surroundings(img))

        # Apply importance weights and flatten
        holds_flat = torch.flatten(holds, 1) * self.holds_weight
        holds_surroundings_flat = torch.flatten(holds_surroundings, 1) * self.holds_surroundings_weight

        # Apply weight to angle and unsqueeze to match batch dimension
        angle *= self.angle_weight

        # print(holds_flat.shape, holds_surroundings_flat.shape, angle.shape)
        # output of print: torch.Size([32, 128]) torch.Size([32, 128]) torch.Size([32, 1])

        # Combine features
        combined = torch.cat((holds_flat, holds_surroundings_flat, angle), dim=1)

        return self.classifier(combined)
