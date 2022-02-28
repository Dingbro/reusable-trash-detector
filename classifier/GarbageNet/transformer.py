import numpy as np
import torch
import torch.nn as nn, TransformerEncoder, TransformerEncoderLayer

from efficientnet_pytorch import EfficientNet

from .rexnet import ReXNetV1, ReXNetV1DropOut


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GarbageNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.__build_model()

    def __build_model(self):
        if "efficientnet" in self.args.model_name:
            if self.args.pretrained:
                model = EfficientNet.from_pretrained(
                    model_name = self.args.model_name, 
                    num_classes=self.args.num_classes, 
                    advprop = self.args.advprop)
            else:
                model = EfficientNet.from_name(
                    model_name = self.args.model_name, override_params = {"num_classes":self.args.num_classes})
            _layers = list(model.children())[:-3]
            self.backbone = torch.nn.Sequential(*_layers)


        elif "rexnet" in self.args.model_name:
            width_multi = float(self.args.model_name.replace("rexnet", ""))

            model = ReXNetV1(width_mult = width_multi)

            if self.args.pretrained:
                model.load_state_dict(torch.load('/home/ubuntu/2020kaist/Fashion-256-classification/Fashion-256-classification/model/weights/rexnetv1_{}x.pth'.format(width_multi)))

            _layers = list(model.children())[:-1]
            self.backbone = torch.nn.Sequential(*_layers)
            
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