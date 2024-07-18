import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
import numpy as np
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, \
    conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
from einops import rearrange
# from .deepgaitv2 import DeepGaitV2
import sys
sys.path.append("...")
from utils import config_loader

blocks_map = {
    '2d': BasicBlock2D,
    'p3d': BasicBlockP3D,
    '3d': BasicBlock3D
}


class sils_DeepGaitV2(nn.Module):

    def __init__(self, save_name=""):
        super(sils_DeepGaitV2, self).__init__()
        mode = "p3d"
        block = blocks_map[mode]

        in_channels = 1
        layers = [1, 4, 4, 1]
        channels = [64, 128, 256, 512]
        self.inference_use_emb2 = False
        self.device = torch.distributed.get_rank()

        if mode == '3d':
            strides = [
                [1, 1],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            strides = [
                [1, 1],
                [2, 2],
                [2, 2],
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(
            self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=3000)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        
        if save_name:
            checkpoint = torch.load(save_name, map_location=torch.device("cuda", self.device))
            model_state_dict = checkpoint['model']
            self.load_state_dict(model_state_dict)

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=stride,
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                                           nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(1, *stride),
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
            embed = embed_2
        else:
            embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval


class ske_DeepGaitV2(nn.Module):

    def __init__(self):
        super(ske_DeepGaitV2, self).__init__()
        mode = "p3d"
        block = blocks_map[mode]

        in_channels = 2
        layers = [1, 4, 4, 1]
        channels = [64, 128, 256, 512]
        self.inference_use_emb2 = False

        if mode == '3d':
            strides = [
                [1, 1],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            strides = [
                [1, 1],
                [2, 2],
                [2, 2],
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(
            self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=3000)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=stride,
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                                           nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(1, *stride),
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
            embed = embed_2
        else:
            embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval


class sils_Frozen(nn.Module):

    def __init__(self, save_name):
        super(sils_Frozen, self).__init__()
        mode = "p3d"
        block = blocks_map[mode]

        in_channels = 1
        layers = [1, 4, 4, 1]
        channels = [64, 128, 256, 512]
        self.inference_use_emb2 = False
        self.device = torch.distributed.get_rank()

        if mode == '3d':
            strides = [
                [1, 1],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            strides = [
                [1, 1],
                [2, 2],
                [2, 2],
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(
            self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=3000)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        checkpoint = torch.load(save_name, map_location=torch.device("cuda", self.device))
        model_state_dict = checkpoint['model']
        self.load_state_dict(model_state_dict)

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=stride,
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                                           nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(1, *stride),
                              padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        del ipts
        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
            embed = embed_2
        else:
            embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval


class Gait3T(BaseModel):

    def build_network(self, model_cfg):
        self.sil_model = sils_DeepGaitV2()
        # self.sil_model = sils_DeepGaitV2("output/Gait3D/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-60000.pt")
        self.ske_model = ske_DeepGaitV2()
        self.frozen_tower = sils_Frozen("output/Gait3D/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-60000.pt")
        self.non_init_list = ["sil_model", "frozen_tower"]

        final_ch = model_cfg['ske_model']['out_dim']
        self.map = nn.Sequential(
            nn.Linear(256, 1),
            nn.LeakyReLU(),
            nn.Linear(final_ch, final_ch)
        )
        self.map_pose = nn.Linear(final_ch, final_ch)

    def init_parameters(self):
        for name, m in self.named_modules():
            tower_name = name.split(".")
            if tower_name[0] not in self.non_init_list:
                if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0.0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0.0)
                elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                    if m.affine:
                        nn.init.normal_(m.weight.data, 1.0, 0.02)
                        nn.init.constant_(m.bias.data, 0.0)
         
    def inputs_pretreament(self, inputs):
       ### Ensure the same data augmentation for heatmap and silhouette
       pose_sils = inputs[0]
       new_data_list = []
       for pose, sil in zip(pose_sils[0], pose_sils[1]):
           sil = sil[:, np.newaxis, ...] # [T, 1, H, W]
           pose_h, pose_w = pose.shape[-2], pose.shape[-1]
           sil_h, sil_w = sil.shape[-2], sil.shape[-1]
           if sil_h != sil_w and pose_h == pose_w:
               cutting = (sil_h - sil_w) // 2
               pose = pose[..., cutting:-cutting]
           cat_data = np.concatenate([pose, sil], axis=1) # [T, 3, H, W]
           new_data_list.append(cat_data)
       new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
       return super().inputs_pretreament(new_inputs)
    
    def log_grad(self):
        grad_dict = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                if param.grad.abs().sum() > 0 and not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
                    grad = param.grad
                    grad_dict[f'histogram/{name}.grad'] = grad.detach().cpu().numpy().astype(float)
        return grad_dict

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        pose = ipts[0]
        maps = pose[:, :, :2, ...]
        sils = pose[:, :, -1, ...]
        pose = pose.transpose(1, 2).contiguous()

        del ipts
        sil_feat = self.sil_model(([sils], labs, typs, vies, seqL))['training_feat']
        ske_feat = self.ske_model(([maps], labs, typs, vies, seqL))['training_feat']

        sil_logits = sil_feat['softmax']['logits']
        ske_logits = ske_feat['softmax']['logits']
        sil_embed = sil_feat['triplet']['embeddings']
        ske_embed = ske_feat['triplet']['embeddings']
        sil_feat_transpose = sil_embed.transpose(0, 1).contiguous()  # [embed_size, n, separate_fc_cnt]
        ske_feat_transpose = ske_embed.transpose(0, 1).contiguous()  # [embed_size, n, separate_fc_cnt]
        with torch.no_grad():
           sil_anchor_feat = self.frozen_tower(([sils], labs, typs, vies, seqL))['training_feat']
           sil_anchor_feat_transpose = sil_anchor_feat['triplet']['embeddings'].transpose(0, 1).contiguous()
        proj_per_sil = sil_feat_transpose @ ske_feat_transpose.transpose(1, 2).contiguous()  # [embed_size, n, separate_fc_cnt] @ [embed_size, separate_fc_cnt, n]
        proj_per_ske = proj_per_sil.transpose(1, 2).contiguous()
        proj_per_sil_anchor = sil_feat_transpose @ sil_anchor_feat_transpose.transpose(1, 2).contiguous()

        retval = {
            'training_feat': {
                'sil_supcl': {'projections': proj_per_sil, 'targets': labs},
                'ske_supcl': {'projections': proj_per_ske, 'targets': labs},
                'sil_anchor_supcl': {'projections': proj_per_sil_anchor, 'targets': labs},
                'sil_triplet': {'embeddings': sil_embed, 'labels': labs},
                'ske_triplet': {'embeddings': ske_embed, 'labels': labs},
                'sil_softmax': {'logits': sil_logits, 'labels': labs},
                'ske_softmax': {'logits': ske_logits, 'labels': labs},
            },
            'visual_summary': {
                'image/sils': rearrange(pose * 255., 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': sil_anchor_feat['triplet']['embeddings']
            }
        }
        retval['visual_summary'].update(self.log_grad()) # adds grads to tensorboard
        return retval


