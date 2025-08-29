import torch
import torch.nn as nn
from torchvision import models


class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNN, self).__init__()
        
        # Load pre-trained models
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg19 = models.vgg19(pretrained=True)
        self.resnet50 = models.resnet50(pretrained=True)
        self.densenet121 = models.densenet121(pretrained=True)
        
        # Unfreeze fine-tuned layers as customized
        # VGG16: Unfreeze block4_conv3 and block5 layers
        for param in self.vgg16.parameters():
            param.requires_grad = False
        for param in self.vgg16.features[12:19].parameters():  # block4_conv3 to block5_pool
            param.requires_grad = True
            
        # VGG19: Unfreeze block4_pool to block5 layers
        for param in self.vgg19.parameters():
            param.requires_grad = False
        for param in self.vgg19.features[16:22].parameters():  # block4_pool to block5_pool
            param.requires_grad = True
            
        # ResNet50: Unfreeze conv4_block1_2 to conv5_block3_out
        for param in self.resnet50.parameters():
            param.requires_grad = False
        for param in self.resnet50.layer3.parameters():  # conv4
            param.requires_grad = True
        for param in self.resnet50.layer4.parameters():  # conv5
            param.requires_grad = True
            
        # DenseNet121: Unfreeze conv4_block24_1 to conv5_block16_0
        for param in self.densenet121.parameters():
            param.requires_grad = False
        for param in self.densenet121.features.denseblock3.parameters():  # conv4
            param.requires_grad = True
        for param in self.densenet121.features.denseblock4.parameters():  # conv5
            param.requires_grad = True
        
        # Custom dense layers for feature extraction (D layer with 1024 units)
        self.vgg16_d_layer = nn.Sequential(
            nn.Linear(512*7*7, 1024),  # After global max pooling
            nn.ReLU()
        )
        self.vgg19_d_layer = nn.Sequential(
            nn.Linear(512*7*7, 1024),  # After global max pooling
            nn.ReLU()
        )
        self.resnet50_d_layer = nn.Sequential(
            nn.Linear(2048, 1024),  # After global max pooling
            nn.ReLU()
        )
        self.densenet121_d_layer = nn.Sequential(
            nn.Linear(1024, 1024),  # After global max pooling
            nn.ReLU()
        )
        
        # Custom feature fusion network
        self.fusion_network = nn.Sequential(
            nn.Dropout(0.6),
            nn.BatchNorm1d(4096),  # 4*1024 from combined D layers
            nn.Linear(4096, num_classes)
        )
        nn.init.xavier_uniform_(self.fusion_network[2].weight)
        nn.init.zeros_(self.fusion_network[2].bias)
    
    def forward(self, x):
        # VGG16 feature extraction
        vgg16_features = self.vgg16.features(x)
        vgg16_features = vgg16_features.view(vgg16_features.size(0), -1)  # Flatten
        vgg16_features = self.vgg16_d_layer(vgg16_features)
        
        # VGG19 feature extraction
        vgg19_features = self.vgg19.features(x)
        vgg19_features = vgg19_features.view(vgg19_features.size(0), -1)  # Flatten
        vgg19_features = self.vgg19_d_layer(vgg19_features)
        
        # ResNet50 feature extraction
        resnet50_features = self.resnet50.conv1(x)
        resnet50_features = self.resnet50.bn1(resnet50_features)
        resnet50_features = self.resnet50.relu(resnet50_features)
        resnet50_features = self.resnet50.maxpool(resnet50_features)
        resnet50_features = self.resnet50.layer1(resnet50_features)
        resnet50_features = self.resnet50.layer2(resnet50_features)
        resnet50_features = self.resnet50.layer3(resnet50_features)
        resnet50_features = self.resnet50.layer4(resnet50_features)
        resnet50_features = self.resnet50.avgpool(resnet50_features)
        resnet50_features = resnet50_features.view(resnet50_features.size(0), -1)
        resnet50_features = self.resnet50_d_layer(resnet50_features)
        
        # DenseNet121 feature extraction
        densenet121_features = self.densenet121.features(x)
        densenet121_features = nn.functional.relu(densenet121_features, inplace=True)
        densenet121_features = nn.functional.adaptive_avg_pool2d(densenet121_features, (1, 1))
        densenet121_features = densenet121_features.view(densenet121_features.size(0), -1)
        densenet121_features = self.densenet121_d_layer(densenet121_features)
        
        # Combine features from D layers
        combined_features = torch.cat((vgg16_features, vgg19_features, resnet50_features, densenet121_features), dim=1)
        
        # Pass through fusion network
        out = self.fusion_network(combined_features)
        return out