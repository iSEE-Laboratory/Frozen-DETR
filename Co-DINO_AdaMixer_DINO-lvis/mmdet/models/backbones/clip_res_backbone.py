from collections import OrderedDict
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, bn=nn.BatchNorm2d):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = bn(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = bn(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = bn(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", bn(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, width=64, bn=nn.BatchNorm2d, args=None):
        super().__init__()

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = bn(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = bn(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = bn(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], bn=bn)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, bn=bn)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, bn=bn)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, bn=bn)

        # embed_dim = width * 32  # the ResNet feature dimension
        # self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, args)

    def _make_layer(self, planes, blocks, stride=1, bn=nn.BatchNorm2d):
        layers = [Bottleneck(self._inplanes, planes, stride, bn)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, bn=bn))

        return nn.Sequential(*layers)

    def forward(self, x, out_indices):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = stem(x)

        outs = []
        for i, res_layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = res_layer(x)
            if i in out_indices:
                outs.append(x)
        return outs




@BACKBONES.register_module()
class CLIPCNNBackbone(nn.Module):
    def __init__(self, pretrained_weight, fronzen_backbone=True, frozen_stages=-1, norm_eval=True, out_indices=(0, 1, 2, 3),):
        super().__init__()
        with open(pretrained_weight, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu")
            state_dict = model.state_dict()
        self.fronzen_backbone = fronzen_backbone
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]

        self.visual = ModifiedResNet(
            layers=vision_layers,
            width=vision_width,
        )

        new_state_dict = dict()
        new_state_dict.update({k.replace('visual.', ''): v for k, v in state_dict.items() if k.startswith('visual.')})
        for key in ["attnpool.positional_embedding", "attnpool.k_proj.weight", "attnpool.k_proj.bias", "attnpool.q_proj.weight", "attnpool.q_proj.bias", "attnpool.v_proj.weight", "attnpool.v_proj.bias", "attnpool.c_proj.weight", "attnpool.c_proj.bias"]:
            if key in new_state_dict:
                del new_state_dict[key]

        self.visual.load_state_dict(new_state_dict)
        if not self.norm_eval:
            self.visual = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.visual)
        if self.fronzen_backbone:
            self._freeze_all_stages()
        else:
            self._freeze_stages()



    def forward(self, x):
        x = self.visual(x, self.out_indices)
        return x
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(CLIPCNNBackbone, self).train(mode)
        if mode and self.fronzen_backbone:
            self._freeze_all_stages()
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
            return
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_all_stages(self):
        self.visual.eval()
        for param in self.visual.parameters():
            param.requires_grad = False
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.visual.bn1.eval()
            self.visual.bn2.eval()
            self.visual.bn3.eval()
            for m in [self.visual.conv1, self.visual.bn1, self.visual.conv2, self.visual.bn2, self.visual.conv3, self.visual.bn3]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self.visual, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        


