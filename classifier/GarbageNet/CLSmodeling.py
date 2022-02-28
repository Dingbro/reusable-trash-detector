import numpy as np
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from .efficientnet import EfficientNet_Multi_Head

from .rexnet import ReXNetV1, ReXNetV1DropOut


class GarbageNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.__build_model()

    def __build_model(self):
        if "efficientnet" in self.args.model_name:
            if self.args.pretrained:
                self.model = EfficientNet.from_pretrained(
                    model_name = self.args.model_name, 
                    num_classes=self.args.num_classes, 
                    advprop = self.args.advprop)
            else:
                self.model = EfficientNet.from_name(
                    model_name = self.args.model_name, override_params = {"num_classes":self.args.num_classes})

        elif "rexnet" in self.args.model_name:
            width_multi = float(self.args.model_name.replace("rexnet", ""))
            pen_channels = int(1280 * width_multi)
            dropout_ratio = ReXNetV1DropOut[width_multi]

            self.model = ReXNetV1(width_mult = width_multi)

            if self.args.pretrained:
                self.model.load_state_dict(torch.load('/home/ubuntu/2020kaist/Fashion-256-classification/Fashion-256-classification/model/weights/rexnetv1_{}x.pth'.format(width_multi)))

            self.model.output = nn.Sequential(
                nn.Dropout(dropout_ratio),
                nn.Conv2d(pen_channels, self.args.num_classes, 1, bias=True))
            
        else:
            raise ValueError("no supported model name")
        
    def forward(self,x):
        return self.model(x)

    def freeze_backbone(self):
        if "efficientnet" in self.args.model_name:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model._fc.parameters():
                param.requires_grad = True

        elif "rexnet" in self.args.model_name:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.output.parameters():
                param.requires_grad = True
        
        else:
            raise ValueError("no supported model name")

class MultiheadClassifier(nn.Module):
    def __init__(self, args):
        super(MultiheadClassifier, self).__init__()
        self.args = args

        # base model as efficientnet
        self.efficientnet = EfficientNet_Multi_Head.from_pretrained(
                    model_name = self.args.model_name, 
                    advprop = self.args.advprop)
        c = list(self.efficientnet.children())[-6].out_channels
        
        self.fc1 = nn.Linear(in_features=c, out_features=2, bias=True)
        self.fc2 = nn.Linear(in_features=c, out_features=2, bias=True)
        self.fc3 = nn.Linear(in_features=c, out_features=2, bias=True)
        self.fc4 = nn.Linear(in_features=c, out_features=2, bias=True)
        self.fc5 = nn.Linear(in_features=c, out_features=2, bias=True)
        self.fc6 = nn.Linear(in_features=c, out_features=2, bias=True)
        self.fc7 = nn.Linear(in_features=c, out_features=2, bias=True)
        
    def forward(self, inputs):

        bs = inputs.size(0)

        # Convolution layers
        x = self.efficientnet.extract_features(inputs)

        # Pooling and final linear layer
        x = self.efficientnet._avg_pooling(x)
        x = x.view(bs, -1)

        x1 = self.efficientnet._dropout(x)
        x1 = self.fc1(x1)
        
        x2 = self.efficientnet._dropout(x)
        x2 = self.fc2(x2)

        x3 = self.efficientnet._dropout(x)
        x3 = self.fc3(x3)

        x4 = self.efficientnet._dropout(x)
        x4 = self.fc4(x4)

        x5 = self.efficientnet._dropout(x)
        x5 = self.fc5(x5)

        x6 = self.efficientnet._dropout(x)
        x6 = self.fc6(x6)

        x7 = self.efficientnet._dropout(x)
        x7 = self.fc7(x7)
        
        return x1, x2, x3, x4, x5, x6, x7