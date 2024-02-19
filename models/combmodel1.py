import timm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from models.build import EfficientViT_M2

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()

        # Feature extraction (ShuffleNet)
        self.feature_extractor_shufflenet = models.shufflenet_v2_x1_0(weights='ShuffleNet_V2_X1_0_Weights.DEFAULT')
        #print(self.feature_extractor_shufflenet)
        num_features_shufflenet = self.feature_extractor_shufflenet.fc.in_features
        self.feature_extractor_shufflenet.fc = nn.Identity()
        shuff_params = sum(p.numel() for p in self.feature_extractor_shufflenet.parameters())

        # Feature extraction (EfficientViT_M2)
        self.feature_extractor_efficientvit = EfficientViT_M2(pretrained='efficientvit_m2')
        #print(self.feature_extractor_efficientvit)
        num_features_efficientvit = self.feature_extractor_efficientvit.head.l.in_features
        self.feature_extractor_efficientvit.head.l = nn.Identity()
        efficientvit_params = sum(p.numel() for p in self.feature_extractor_efficientvit.parameters())
        # Combine features for classification
        combined_features_size = num_features_shufflenet + num_features_efficientvit
      
        self.classifier = nn.Sequential(
            nn.Linear(combined_features_size, 512),  # Increase width
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.5),  # First dropout layer
            nn.Linear(512, 256),  # Additional hidden layer
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Second dropout layer
            nn.Linear(256, num_classes)
        )
        
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        print("ShuffleNet Parameters:", shuff_params)
        print("EfficientViT Parameters:", efficientvit_params)
        print("Classifier Parameters:", classifier_params)

        total_params = shuff_params + efficientvit_params + classifier_params
        print("Total Parameters:", total_params)

    def forward(self, x):
        # Feature extraction with ShuffleNet
        features_shufflenet = self.feature_extractor_shufflenet(x)

        # Feature extraction with EfficientViT_M2
        features_efficientvit = self.feature_extractor_efficientvit(x)

        # Concatenate the features
        combined_features = torch.cat((features_shufflenet, features_efficientvit), dim=1)
        #print(combined_features.shape)
        # Facial expression classification
        predictions = self.classifier(combined_features)

        return predictions