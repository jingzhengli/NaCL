import torch
import torch.nn as nn
import torch.nn.functional as F

import model.resnet as resnet
from model.utils import BatchNormDomain, initialize_layer
from common.modules.classifier import Classifier as ClassifierBase

class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)
class AdaptiveFeatureNorm(nn.Module):
    r"""
    The `Stepwise Adaptive Feature Norm loss (ICCV 2019) <https://arxiv.org/pdf/1811.07456v2.pdf>`_

    Instead of using restrictive scalar R to match the corresponding feature norm, Stepwise Adaptive Feature Norm
    is used in order to learn task-specific features with large norms in a progressive manner.
    We denote parameters of backbone :math:`G` as :math:`\theta_g`, parameters of bottleneck :math:`F_f` as :math:`\theta_f`
    , parameters of classifier head :math:`F_y` as :math:`\theta_y`, and features extracted from sample :math:`x_i` as
    :math:`h(x_i;\theta)`. Full loss is calculated as follows

    .. math::
        L(\theta_g,\theta_f,\theta_y)=\frac{1}{n_s}\sum_{(x_i,y_i)\in D_s}L_y(x_i,y_i)+\frac{\lambda}{n_s+n_t}
        \sum_{x_i\in D_s\cup D_t}L_d(h(x_i;\theta_0)+\Delta_r,h(x_i;\theta))\\

    where :math:`L_y` denotes classification loss, :math:`L_d` denotes norm loss, :math:`\theta_0` and :math:`\theta`
    represent the updated and updating model parameters in the last and current iterations respectively.

    Args:
        delta (float): positive residual scalar to control the feature norm enlargement.

    Inputs:
        - f (tensor): feature representations on source or target domain.

    Shape:
        - f: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    Examples::

        >>> adaptive_feature_norm = AdaptiveFeatureNorm(delta=1)
        >>> f_s = torch.randn(32, 1000)
        >>> f_t = torch.randn(32, 1000)
        >>> norm_loss = adaptive_feature_norm(f_s) + adaptive_feature_norm(f_t)
    """

    def __init__(self, delta):
        super(AdaptiveFeatureNorm, self).__init__()
        self.delta = delta

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        radius = f.norm(p=2, dim=1).detach()
        assert radius.requires_grad == False
        radius = radius + self.delta
        loss = ((f.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss

class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim = 256, **kwargs):
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck_dim, **kwargs)



class CIFARmodel(nn.Module):
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
                 bottleneck_dim, pool_layer=None):
        super(CIFARmodel, self).__init__()
        self.backbone = backbone
        self.parameter_list = [{"params": self.backbone.parameters(), "lr": 1}]
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        # if bottleneck is None:
        #     self.bottleneck = nn.Identity()
        #     self._features_dim = backbone.out_features
        # else:
        #     self.bottleneck = bottleneck
        #     assert bottleneck_dim > 0
        #     self._features_dim = bottleneck_dim

        # if head is None:
        #     self.head = nn.Linear(self._features_dim, num_classes)
        # else:
        #     self.head = head
        # self.finetune = finetune
        # self.contrast_layer = nn.Sequential(
        #     # nn.Linear(self.base_network.out_dim,self.base_network.out_dim),
        #     # nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.backbone.out_features,bottleneck_dim)
        # )      
        self.contrast_layer = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(128,128)
        )       
         
        initialize_layer(self.contrast_layer)
        self.parameter_list += [{"params": self.contrast_layer.parameters(), "lr": 10}]

        self.classifier = nn.Linear(128, self.num_classes)
        initialize_layer(self.classifier)
        self.parameter_list += [{"params": self.classifier.parameters(), "lr": 10}]

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor):
        """"""
        # f = self.pool_layer(self.backbone(x))
        # f = self.bottleneck(f)
        # predictions = self.head(f)
        # if self.training:
        #     return predictions, f
        # else:
        #     return predictions
        end_points = {}
        features = self.pool_layer(self.backbone(x))
        # features = F.normalize(features, p=2, dim=1)
        # features=self.bottleneck(features) 
        features = F.relu(features) 
        # features = F.normalize(features, p=2, dim=1)
        end_points['features'] = features

        # contrast loss head
        contrast_features = self.contrast_layer(features)
        contrast_features = F.normalize(contrast_features, p=2, dim=1)
        end_points['contrast_features'] = contrast_features

        logits = self.classifier(features)
        end_points['logits'] = logits

        confidences, predictions = torch.max(F.softmax(logits, dim=1), 1)
        end_points['predictions'] = predictions
        end_points['confidences'] = confidences

        return end_points

    def get_parameter_list(self):
        return self.parameter_list
