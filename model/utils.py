import torch
import torch.nn as nn
from torch.nn import init


def weights_init_he(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if 'weight' in m.state_dict().keys():
            m.weight.data.normal_(1.0, 0.02)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)
    else:
        if 'weight' in m.state_dict().keys():
            init.kaiming_normal_(m.weight)
        if 'bias' in m.state_dict().keys():
            m.bias.data.fill_(0)


def init_weights(model, state_dict, num_domains=1, BN2BNDomain=False):
    model.apply(weights_init_he)

    if state_dict is not None:

        model_state_dict = model.state_dict()

        keys = set(model_state_dict.keys())
        trained_keys = set(state_dict.keys())

        shared_keys = keys.intersection(trained_keys)
        new_state_dict = {key: state_dict[key] for key in shared_keys}
        if BN2BNDomain:
            for k in (trained_keys - shared_keys):
                if k.find('fc') != -1:
                    continue
                suffix = k.split('.')[-1]
                for d in range(num_domains):
                    bn_key = k.replace(suffix, 'bn_domain.' + str(d) + '.' + suffix)
                    new_state_dict[bn_key] = state_dict[k]

        model.load_state_dict(new_state_dict)

    return model


class DomainModule(nn.Module):
    def __init__(self, num_domains, **kwargs):
        super(DomainModule, self).__init__()
        self.num_domains = num_domains
        self.domain = 0

    def set_domain(self, domain=0):
        assert (domain < self.num_domains), \
            "The domain id exceeds the range (%d vs. %d)" \
            % (domain, self.num_domains)
        self.domain = domain


class BatchNormDomain(DomainModule):
    def __init__(self, in_size, num_domains, norm_layer, **kwargs):
        super(BatchNormDomain, self).__init__(num_domains)
        self.bn_domain = nn.ModuleDict()
        for n in range(self.num_domains):
            self.bn_domain[str(n)] = norm_layer(in_size, **kwargs)

    def forward(self, x):
        out = self.bn_domain[str(self.domain)](x)
        return out


def tanh_clip(x, clip_val=10.):
    """
    soft clip values to the range [-clip_val, +clip_val]
    """
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip


def initialize_layer(layer):
    for m in layer.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
