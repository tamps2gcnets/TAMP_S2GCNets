import torch
import torch.nn.functional as F
import torch.nn as nn

# MAX DNN #
class MAXCNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MAXCNN, self).__init__()
        self.dim_out = dim_out
        self.features = nn.Sequential(
            nn.Conv2d(dim_in, 8, kernel_size=2, stride=2), #channel of MPG is 1 or window_size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, dim_out, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.maxpool = nn.MaxPool2d(3,3)

    def forward(self, MPG):
        feature = self.features(MPG)
        feature = self.maxpool(feature)
        feature = feature.view(-1, self.dim_out) #B, dim_out
        return feature

# MEAN DNN #
class MEANCNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MEANCNN, self).__init__()
        self.dim_out = dim_out
        self.features = nn.Sequential(
            nn.Conv2d(dim_in, 8, kernel_size=2, stride=2), #channel of MPG is 1 or window_size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, dim_out, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AvgPool2d(3,3)

    def forward(self, MPG):
        feature = self.features(MPG)
        feature = self.avgpool(feature)
        feature = feature.view(-1, self.dim_out) #B, dim_out
        return feature
