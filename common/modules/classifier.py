from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

__all__ = ['Classifier']
def initialize_layer(layer):
    for m in layer.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

def initialize_layer2(layer):
    for m in layer.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            # nn.init.constant_(m.bias, 0)

class Classifier(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, 
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.bottleneck_dim = bottleneck_dim
        self.parameter_list = [{"params": self.backbone.parameters(), "lr": 1}]
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone.out_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        initialize_layer(self.bottleneck)
        self.parameter_list += [{"params": self.bottleneck.parameters(), "lr": 10}]
        self.contrast_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, self.bottleneck_dim),
        )
        initialize_layer(self.contrast_layer)
        self.parameter_list += [{"params": self.contrast_layer.parameters(), "lr": 10}]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        initialize_layer(self.classifier)
        self.parameter_list += [{"params": self.classifier.parameters(), "lr": 10}]

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        end_points = {}
        features = self.pool_layer(self.backbone(x))
        end_points['features'] = features
        features=self.bottleneck(features) 
        end_points['norm_features'] = features

        contrast_features = self.contrast_layer(features)
        contrast_features = F.normalize(contrast_features, p=2, dim=1)
        end_points['contrast_features'] = contrast_features
        logits = self.classifier(features)
        end_points['logits'] = logits

        return end_points

    def weight_norm(self):
        w = self.classifier[1].weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier[1].weight.data = w.div(norm.expand_as(w))


    def get_parameter_list(self):
        return self.parameter_list
