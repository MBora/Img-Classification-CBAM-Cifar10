import torch
import torch.nn as nn
# To compute Channel attention
class ChannelAttentionModule(nn.Module): 
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=3, padding=1, bias=False) #kernel size and padding can vary according to requirement
        self.relu = nn.ReLU() #Activation function between convolution layers
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_module = self.fc2(self.relu(self.fc1(self.avg_pool(x)))) #average pool channel
        max_pool_module = self.fc2(self.relu(self.fc1(self.max_pool(x)))) #max pool channel
        
        out = self.sigmoid(avg_pool_module + max_pool_module) #out is the attention map
        refined_feature_map = out * x #out*x is the refined feature map or the channel attention map
        return refined_feature_map

#To compute spatial attention
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        #input channels come from one average channel and one spatial channel
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1) #kernel size and padding can vary according to requirement
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = torch.mean(x, dim=1, keepdim=True) #Average pooling along a dimension
        max_x, _ = torch.max(x, dim=1, keepdim=True) #Max pooling along a dimension
        new_x = torch.cat([avg_x, max_x], dim=1) #Concatenation of both tesnors
        #Layer inside spatial module
        new_x = self.conv1(new_x)
        new_x = self.relu1(new_x)
        new_x = self.conv2(new_x)
        new_x = self.sigmoid(new_x)
        refined_feature_map = new_x * x #new_x*x is the refined feature map or the spatial attention map; x was the channel attention
        return refined_feature_map

#To combine channel and spatial attention modules i.e implement CBAM module
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttentionModule(channels)
        self.sa = SpatialAttentionModule()

    def forward(self, x):
        x = self.ca(x)
        channel_attention = x.clone().detach()
        x = self.sa(x)
        spatial_attention = x.clone().detach()
        return x, channel_attention, spatial_attention #spatial_attention and x is the same

#Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25) #0.25 gave better performance than p=0.5
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)
        self.cbam = CBAM(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 10)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x, channel_attention, spatial_attention = self.cbam(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        return x, channel_attention, spatial_attention