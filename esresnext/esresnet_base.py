import abc
import numpy as np
import scipy.signal as sps

import torch
import torch.nn.functional as F

try:
    import torchvision as tv
except ImportError:
    tv = None

from .attention import Attention2d
from .transforms import scale

from typing import cast
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional


def conv3x3(in_planes: int, out_planes: int, stride=1, groups: int = 1, dilation: Union[int, Tuple[int, int]] = 1):
    return torch.nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: Union[int, Tuple[int, int]] = 1):
    return torch.nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(torch.nn.Module):

    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(torch.nn.Module):

    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = torch.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetWithAttention(torch.nn.Module):

    def __init__(self, block, layers, apply_attention=False,
                 num_channels=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):

        super(ResNetWithAttention, self).__init__()
        self.apply_attention = apply_attention

        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = torch.nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if self.apply_attention:
            self.att1 = Attention2d(64, 64 * block.expansion, 1, (3, 1), (1, 0))

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        if self.apply_attention:
            self.att2 = Attention2d(64 * block.expansion, 128 * block.expansion, 1, (1, 5), (0, 2))

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        if self.apply_attention:
            self.att3 = Attention2d(128 * block.expansion, 256 * block.expansion, 1, (3, 1), (1, 0))

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        if self.apply_attention:
            self.att4 = Attention2d(256 * block.expansion, 512 * block.expansion, 1, (1, 5), (0, 2))

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        if self.apply_attention:
            self.att5 = Attention2d(512 * block.expansion, 512 * block.expansion, 1, (3, 5), (1, 2))

        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return torch.nn.Sequential(*layers)

    def _forward_pre_processing(self, x):
        x = x.to(torch.get_default_dtype())
        return x

    def _forward_pre_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def _forward_features(self, x):
        x = self._forward_pre_features(x)

        if self.apply_attention:
            x_att = x.clone()
            x = self.layer1(x)
            x_att = self.att1(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer2(x)
            x_att = self.att2(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer3(x)
            x_att = self.att3(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer4(x)
            x_att = self.att4(x_att, x.shape[-2:])
            x = x * x_att
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x

    def _forward_reduction(self, x):
        if self.apply_attention:
            x_att = x.clone()
            x = self.avgpool(x)
            x_att = self.att5(x_att, x.shape[-2:])
            x = x * x_att
        else:
            x = self.avgpool(x)

        x = torch.flatten(x, 1)
        return x

    def _forward_classifier(self, x):
        x = self.fc(x)
        return x

    def forward(self, x, y=None):
        x = self._forward_pre_processing(x)
        x = self._forward_features(x)
        x = self._forward_reduction(x)
        y_pred = self._forward_classifier(x)

        loss = None
        if y is not None:
            loss = self.loss_fn(y_pred, y).mean()

        return y_pred if loss is None else (y_pred, loss)

    def loss_fn(self, y_pred, y):
        if isinstance(y_pred, tuple):
            y_pred, *_ = y_pred
        if y_pred.shape == y.shape:
            loss_pred = F.binary_cross_entropy_with_logits(
                y_pred, y.to(dtype=y_pred.dtype, device=y_pred.device),
                reduction='sum'
            ) / y_pred.shape[0]
        else:
            loss_pred = F.cross_entropy(y_pred, y.to(y_pred.device))
        return loss_pred


class _ESResNet(ResNetWithAttention):

    @staticmethod
    def loading_func(*args, **kwargs):
        raise NotImplementedError

    def __init__(self, block, layers, apply_attention=False,
                 n_fft=256, hop_length=None, win_length=None, window=None,
                 normalized=False, onesided=True,
                 spec_height=224, spec_width=224,
                 num_classes=1000, pretrained=False, lock_pretrained=None,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):

        super(_ESResNet, self).__init__(
            block=block, layers=layers, apply_attention=apply_attention,
            num_channels=3, num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups, width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer
        )

        self.num_classes = num_classes

        self.fc = torch.nn.Linear(
            in_features=self.fc.in_features,
            out_features=self.num_classes,
            bias=self.fc.bias is not None
        )

        if hop_length is None:
            hop_length = int(np.floor(n_fft / 4))
        if win_length is None:
            win_length = n_fft
        if window is None:
            window = 'boxcar'

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalized = normalized
        self.onesided = onesided
        self.spec_height = spec_height
        self.spec_width = spec_width

        self.pretrained = pretrained
        self._inject_members()
        if pretrained:
            self._load_pretrained_weights()

            unlocked_weights = list()
            for name, p in self.named_parameters():
                unlock = True
                if isinstance(lock_pretrained, bool):
                    if lock_pretrained and name not in self._err_msg:
                        unlock = False
                elif isinstance(lock_pretrained, list):
                    if name in lock_pretrained:
                        unlock = False
                p.requires_grad_(unlock)
                if unlock:
                    unlocked_weights.append(name)

        window_buffer = torch.from_numpy(
            sps.get_window(window=window, Nx=win_length, fftbins=True)
        ).to(torch.get_default_dtype())
        self.register_buffer('window', window_buffer)

        self.log10_eps = 1e-18

    def _inject_members(self):
        pass

    def _load_pretrained_weights(self):
        if isinstance(self.pretrained, bool):
            state_dict = self.loading_func(pretrained=True).state_dict()
        else:
            state_dict = torch.load(self.pretrained, map_location='cpu')

        self._err_msg = ''
        try:
            self.load_state_dict(state_dict=state_dict, strict=True)
        except RuntimeError as ex:
            self._err_msg = str(ex)

    def spectrogram(self, x):
        spec = torch.stft(
            x.view(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            pad_mode='reflect',
            normalized=self.normalized,
            onesided=True,
            return_complex=False
        )

        if not self.onesided:
            spec = torch.cat((torch.flip(spec, dims=(-3,)), spec), dim=-3)

        return spec

    def split_spectrogram(self, spec, batch_size):
        spec_height_3_bands = spec.shape[-3] // 3
        spec_height_single_band = 3 * spec_height_3_bands
        spec = spec[:, :spec_height_single_band]
        spec = spec.reshape(batch_size, -1, spec.shape[-3] // 3, *spec.shape[-2:])
        return spec

    def spectrogram_to_power(self, spec):
        spec_height = spec.shape[-3] if self.spec_height < 1 else self.spec_height
        spec_width = spec.shape[-2] if self.spec_width < 1 else self.spec_width

        pow_spec = spec[..., 0] ** 2 + spec[..., 1] ** 2

        if spec_height != pow_spec.shape[-2] or spec_width != pow_spec.shape[-1]:
            pow_spec = F.interpolate(
                pow_spec, size=(spec_height, spec_width),
                mode='bilinear', align_corners=True
            )

        return pow_spec

    def _forward_pre_processing(self, x):
        x = super(_ESResNet, self)._forward_pre_processing(x)
        x = scale(x, -32768.0, 32767, -1.0, 1.0)

        spec = self.spectrogram(x)
        spec_3ch = self.split_spectrogram(spec, x.shape[0])
        pow_spec_3ch = self.spectrogram_to_power(spec_3ch)
        pow_spec_3ch = torch.where(
            cast(torch.Tensor, pow_spec_3ch > 0.0),
            pow_spec_3ch,
            torch.full_like(pow_spec_3ch, self.log10_eps)
        )
        pow_spec_3ch = pow_spec_3ch.reshape(x.shape[0], -1, self.conv1.in_channels, *pow_spec_3ch.shape[-2:])
        x_db = torch.log10(pow_spec_3ch).mul(10.0)

        return x_db

    def _forward_features(self, x_db):
        outputs = list()
        for ch_idx in range(x_db.shape[1]):
            ch = x_db[:, ch_idx]
            out = super(_ESResNet, self)._forward_features(ch)
            outputs.append(out)
        return outputs

    def _forward_reduction(self, x):
        outputs = list()
        for ch in x:
            out = super(_ESResNet, self)._forward_reduction(ch)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=-1).sum(dim=-1)
        return outputs
