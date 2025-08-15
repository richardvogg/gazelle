from abc import ABC, abstractmethod
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms

# Abstract Backbone class
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self):
        pass


# Official DINOv2 backbones from torch hub (https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
class DinoV2BackboneTimm(Backbone):
    def __init__(self, model_name):
        super(DinoV2Backbone, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True, features_only=False)
        self.patch_size = getattr(self.model, 'patch_size', 14)  # Default to 14 if not present
        self.embed_dim = getattr(self.model, 'embed_dim', self.model.num_features)

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        # Forward through the model and extract patch tokens
        features = self.model.forward_features(x)
        if isinstance(features, dict) and 'x_norm_patchtokens' in features:
            x = features['x_norm_patchtokens']
        elif hasattr(features, 'x_norm_patchtokens'):
            x = features.x_norm_patchtokens
        else:
            x = features
        x = x.view(b, out_h, out_w, -1).permute(0, 3, 1, 2)
        return x

    def get_dimension(self):
        return self.embed_dim

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.patch_size, w // self.patch_size)

    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize(in_size),
        ])
    
class DinoV2Backbone(Backbone):
    def __init__(self, model_name):
        super(DinoV2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) # "b (out_h out_w) c -> b c out_h out_w"
        return x
    
    def get_dimension(self):
        return self.model.embed_dim
    
    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])