import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import build_ssh
from models.resnest import build_resnest50



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] in ['Resnet50', 'Resnet50_P2', 'Resnet50_P2_SE']:
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        elif cfg['name'] in ['ResNeSt50', 'ResNeSt50_ATSS', 'ResNeSt50_P2', 'ResNeSt50_P2_SE', 'ResNeSt50_P2_ATSS_SE', 'ResNeSt50_P2_ATSS_SE_SoftNMS']:
            backbone = build_resnest50(pretrained=cfg['pretrain'])
        else:
            raise ValueError('Unsupported network name: {}'.format(cfg['name']))

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        if 'in_channels_list' in cfg:
            in_channels_list = cfg['in_channels_list']
        else:
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 4,
                in_channels_stage2 * 8,
            ]
        fpn_num = len(in_channels_list)
        anchor_num_list = cfg.get('anchor_num_list', [2] * fpn_num)
        ssh_type = cfg.get('ssh_type', 'ssh')
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        # 根据配置选择原始 SSH 或 SSH+SE，便于后续做严格消融。
        self.ssh = nn.ModuleList([build_ssh(ssh_type, out_channels, out_channels) for _ in range(fpn_num)])

        self.ClassHead = self._make_class_head(fpn_num=fpn_num, inchannels=cfg['out_channel'], anchor_num_list=anchor_num_list)
        self.BboxHead = self._make_bbox_head(fpn_num=fpn_num, inchannels=cfg['out_channel'], anchor_num_list=anchor_num_list)
        self.LandmarkHead = self._make_landmark_head(fpn_num=fpn_num, inchannels=cfg['out_channel'], anchor_num_list=anchor_num_list)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2,anchor_num_list=None):
        classhead = nn.ModuleList()
        anchor_num_list = anchor_num_list or [anchor_num] * fpn_num
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num_list[i]))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2,anchor_num_list=None):
        bboxhead = nn.ModuleList()
        anchor_num_list = anchor_num_list or [anchor_num] * fpn_num
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num_list[i]))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2,anchor_num_list=None):
        landmarkhead = nn.ModuleList()
        anchor_num_list = anchor_num_list or [anchor_num] * fpn_num
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num_list[i]))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        features = [ssh_layer(feature) for ssh_layer, feature in zip(self.ssh, fpn)]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
