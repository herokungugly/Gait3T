import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..backbones.resgcn import ResGCN
from ..modules import Graph
import numpy as np
import torch.nn.functional as F
import copy
from .. modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, \
conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
from einops import rearrange
import sys
sys.path.append(" ... ")
from utils import config_loader


class GaitGraph2(nn.Module):
    """
        GaitGraph2: Towards a Deeper Understanding of Skeleton-based Gait Recognition
        Paper:    https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper
        Github:   https://github.com/tteepe/GaitGraph2
    """
    def build_network(self, model_cfg):
         
        self.joint_format = model_cfg['joint_format']
        self.input_num = model_cfg['input_num']
        self.block = model_cfg['block']
        self.input_branch = model_cfg['input_branch']
        self.main_stream = model_cfg['main_stream']
        self.num_class = model_cfg['num_class']
        self.reduction = model_cfg['reduction']
        self.tta = model_cfg['tta']
        ## Graph Init ##
        self.graph = Graph(joint_format=self.joint_format,max_hop=3)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        ## Network ##
        self.ResGCN = ResGCN(input_num=self.input_num, input_branch=self.input_branch, 
                             main_stream=self.main_stream, num_class=self.num_class,
                             reduction=self.reduction, block=self.block,graph=self.A)

    def forward(self, inputs):

        ipts, labs, type_, view_, seqL = inputs
        x_input = ipts[0] 
        N, T, V, I, C = x_input.size()
        pose  = x_input
        flip_idx = self.graph.flip_idx

        if not self.training and self.tta:
            multi_input = MultiInput(self.graph.connect_joint, self.graph.center)
            x1 = []
            x2 = []
            for i in range(N):
                x1.append(multi_input(x_input[i,:,:,0,:3].flip(0)))
                x2.append(multi_input(x_input[i,:,flip_idx,0,:3]))
            x_input = torch.cat([x_input, torch.stack(x1,0), torch.stack(x2,0)], dim=0)
        
        x = x_input.permute(0, 3, 4, 1, 2).contiguous()

        # resgcn
        x = self.ResGCN(x)

        if not self.training and self.tta:
            f1, f2, f3 = torch.split(x, [N, N, N], dim=0)
            x = torch.cat((f1, f2, f3), dim=1)
             
        embed = torch.unsqueeze(x,-1)
        
        retval = {
            'training_feat': {
                'SupConLoss': {'features': x , 'labels': labs}, # loss
            },
            'visual_summary': {
                'image/pose': pose.view(N*T, 1, I*V, C).contiguous() # visualization
            },
            'inference_feat': {
                'embeddings': embed # for metric
            }
        }
        return retval
    
class MultiInput:
    def __init__(self, connect_joint, center):
        self.connect_joint = connect_joint
        self.center = center

    def __call__(self, data):

        # T, V, C -> T, V, I=3, C + 2
        T, V, C = data.shape
        x_new = torch.zeros((T, V, 3, C + 2), device=data.device)

        # Joints
        x = data
        x_new[:, :, 0, :C] = x
        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]

        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]

        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        bone_length = 0
        for i in range(C - 1):
            bone_length += torch.pow(x_new[:, :, 2, i], 2)
        bone_length = torch.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            x_new[:, :, 2, C+i] = torch.acos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]

        data = x_new
        return data


blocks_map = {
    '2d': BasicBlock2D,
    'p3d': BasicBlockP3D,
    '3d': BasicBlock3D
}


class sils_DeepGaitV2(nn.Module):

    def __init__(self):
        super(sils_DeepGaitV2, self).__init__()
        mode = "p3d"
        block = blocks_map[mode]

        in_channels = 1
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


class Gait3T_coords(BaseModel):

    def build_network(self, model_cfg):
        self.sil_model = sils_DeepGaitV2()
        self.ske_model = GaitGraph2()
        self.frozen_tower = sils_Frozen("output/Gait3D/DeepGaitV2/DeepGaitV2/checkpoints/DeepGaitV2-60000.pt")

        final_ch = model_cfg['ske_model']['out_dim']
        self.map = nn.Sequential(
            nn.Linear(256, 1),
            nn.LeakyReLU(),
            nn.Linear(final_ch, final_ch)
        )
        self.map_pose = nn.Linear(final_ch, final_ch)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        pose = ipts[0]
        print(len(pose))
        maps = pose[0]
        sils = pose[1]
        pose = sils.transpose(1, 2).contiguous()

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
           sil_anchor_feat_transpose = self.frozen_tower(([sils], labs, typs, vies, seqL))['training_feat']['triplet']['embeddings'].transpose(0, 1).contiguous()
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
                'image/sils': rearrange(pose * 255., 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': ske_feat
            }
        }
        return retval
