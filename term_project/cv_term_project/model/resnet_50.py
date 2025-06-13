import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """
    ResNet50的瓶颈残差块 (Bottleneck Block)
    expansion = 4 表示输出通道是中间通道的4倍
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1x1卷积：降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷积：特征提取
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1卷积：升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 用于维度匹配的下采样层
        
    def forward(self, x):
        identity = x  # 保存输入用于跳跃连接
        
        # 第一个1x1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 第二个1x1卷积
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 如果需要下采样，对跳跃连接进行处理
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接
        out += identity
        out = self.relu(out)
        
        return out


class ResNet50(nn.Module):
    """
    ResNet50网络结构
    """
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        
        # 输入处理层
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差层 (Stage)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)           # conv2_x: 3个残差块
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2) # conv3_x: 4个残差块
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2) # conv4_x: 6个残差块
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2) # conv5_x: 3个残差块
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建残差层
        
        Args:
            block: 残差块类型 (Bottleneck)
            out_channels: 输出通道数
            blocks: 残差块数量
            stride: 步长
        """
        downsample = None
        
        # 如果步长不为1或输入输出通道数不匹配，需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        # 第一个残差块可能需要下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # 其余残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 输入处理
        x = self.conv1(x)       # [B, 3, 224, 224] -> [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, 112, 112] -> [B, 64, 56, 56]
        
        # 四个残差层
        x = self.layer1(x)      # [B, 64, 56, 56] -> [B, 256, 56, 56]
        x = self.layer2(x)      # [B, 256, 56, 56] -> [B, 512, 28, 28]
        x = self.layer3(x)      # [B, 512, 28, 28] -> [B, 1024, 14, 14]
        x = self.layer4(x)      # [B, 1024, 14, 14] -> [B, 2048, 7, 7]
        
        # 分类头
        x = self.avgpool(x)     # [B, 2048, 7, 7] -> [B, 2048, 1, 1]
        x = torch.flatten(x, 1) # [B, 2048, 1, 1] -> [B, 2048]
        x = self.fc(x)          # [B, 2048] -> [B, num_classes]
        
        return x
