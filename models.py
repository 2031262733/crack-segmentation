"""Two segmentation backbones: a plain U-Net baseline and DeepCrack (VGG16 + deep supervision)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


def conv_bn(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4-level U-Net with bilinear upsampling, trained from scratch."""

    def __init__(self, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.e1 = nn.Sequential(conv_bn(3, c[0]), conv_bn(c[0], c[0]))
        self.e2 = nn.Sequential(conv_bn(c[0], c[1]), conv_bn(c[1], c[1]))
        self.e3 = nn.Sequential(conv_bn(c[1], c[2]), conv_bn(c[2], c[2]))
        self.e4 = nn.Sequential(conv_bn(c[2], c[3]), conv_bn(c[3], c[3]))
        self.bott = nn.Sequential(conv_bn(c[3], c[4]), conv_bn(c[4], c[4]))
        self.d4 = nn.Sequential(conv_bn(c[4] + c[3], c[3]), conv_bn(c[3], c[3]))
        self.d3 = nn.Sequential(conv_bn(c[3] + c[2], c[2]), conv_bn(c[2], c[2]))
        self.d2 = nn.Sequential(conv_bn(c[2] + c[1], c[1]), conv_bn(c[1], c[1]))
        self.d1 = nn.Sequential(conv_bn(c[1] + c[0], c[0]), conv_bn(c[0], c[0]))
        self.head = nn.Conv2d(c[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d4 = self.d4(torch.cat([F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], 1))
        d3 = self.d3(torch.cat([F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        return [self.head(d1)]


class DeepCrack(nn.Module):
    """DeepCrack: VGG16 encoder, one 1x1 side output per stage, deep supervision + fusion.

    Reference: Liu et al., "DeepCrack: A Deep Hierarchical Feature Learning Architecture
    for Crack Segmentation", Neurocomputing 2019. Re-implemented in PyTorch here.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        feats = vgg16(weights=weights).features
        self.stage1 = feats[0:4]    # conv1_1..conv1_2   (64)
        self.stage2 = feats[4:9]    # pool1 .. conv2_2   (128)
        self.stage3 = feats[9:16]   # pool2 .. conv3_3   (256)
        self.stage4 = feats[16:23]  # pool3 .. conv4_3   (512)
        self.stage5 = feats[23:30]  # pool4 .. conv5_3   (512)
        self.side = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in (64, 128, 256, 512, 512)])
        self.fuse = nn.Conv2d(5, 1, 1)
        self._init_side()

    def _init_side(self):
        for m in self.side:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = x
        outs = []
        for stage, side in zip(
            [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5], self.side
        ):
            h = stage(h)
            outs.append(F.interpolate(side(h), size=x.shape[-2:], mode="bilinear", align_corners=False))
        fused = self.fuse(torch.cat(outs, 1))
        return outs + [fused]


def build_model(name, pretrained=True):
    if name == "unet":
        return UNet()
    if name == "deepcrack":
        return DeepCrack(pretrained=pretrained)
    raise ValueError(name)
