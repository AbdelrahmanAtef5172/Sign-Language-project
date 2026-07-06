import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.output1 = nn.Sequential(
            nn.Conv2d(in_channels_list[0], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.output2 = nn.Sequential(
            nn.Conv2d(in_channels_list[1], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.output3 = nn.Sequential(
            nn.Conv2d(in_channels_list[2], out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.merge1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out1 = self.output1(x[0])
        out2 = self.output2(x[1])
        out3 = self.output3(x[2])

        up3 = F.interpolate(out3, size=out2.shape[-2:], mode="nearest")
        out2 = out2 + up3
        out2 = self.merge2(out2)

        up2 = F.interpolate(out2, size=out1.shape[-2:], mode="nearest")
        out1 = out1 + up2
        out1 = self.merge1(out1)

        return [out1, out2, out3]


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        self.conv3X3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
        )
        self.conv5X5_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )
        self.conv5X5_2 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )
        self.conv7X7_2 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )
        self.conv7x7_3 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )

    def forward(self, x):
        c3 = self.conv3X3(x)
        c5 = self.conv5X5_2(self.conv5X5_1(x))
        c7 = self.conv7x7_3(self.conv7X7_2(self.conv5X5_1(x)))
        out = torch.cat([c3, c5, c7], dim=1)
        return F.relu(out)


class ClassHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3):
        super(ClassHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels=256, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_list = [512, 1024, 2048]
        out_channels = cfg['out_channel']

        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, in_channels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, in_channels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, in_channels=out_channels)

    def _make_class_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(in_channels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(in_channels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, in_channels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(in_channels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        fpn = self.fpn(list(out.values()))

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](features[i]) for i in range(len(features))], dim=1)
        classifications = torch.cat([self.ClassHead[i](features[i]) for i in range(len(features))], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](features[i]) for i in range(len(features))], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
