import torch.nn.functional as F
import copy
import torch
import torch.nn as nn
import numpy as np
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, \
    conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
from einops import rearrange
from .deepgaitv2 import DeepGaitV2
import sys
sys.path.append("...")
from utils import config_loader


class Gait3T_original(BaseModel):

    def build_network(self, model_cfg):
        
        sil_model_cfgs = config_loader(model_cfg["sil_model_cfg"])
        ske_model_cfgs = config_loader(model_cfg["ske_model_cfg"])
        self.sil_model = DeepGaitV2(sil_model_cfgs, training=True)
        self.ske_model = DeepGaitV2(ske_model_cfgs, training=True)

        frozen_cfgs = config_loader(model_cfg["pretrained_model_cfg"])
        self.frozen_tower = DeepGaitV2(frozen_cfgs, training=False)
        self.frozen_tower._load_ckpt(model_cfg['pretrained_model_name'])
 

        final_ch = model_cfg['ske_model']['out_dim']
        self.map = nn.Sequential(
            nn.Linear(256, 1),
            nn.LeakyReLU(),
            nn.Linear(final_ch, final_ch)
        )
        self.map_pose = nn.Linear(final_ch, final_ch)
         
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


    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        pose = ipts[0]
        # print(f'input pose shape: {pose.shape}')
        maps = pose[:, :, :2, ...]
        # print(f'maps shape: {maps.shape}')
        sils = pose[:, :, -1, ...]
        # print(f'sils shape: {sils.shape}')
        pose = pose.transpose(1, 2).contiguous()

        del ipts
        sil_feat = self.sil_model(([sils], labs, typs, vies, seqL))['training_feat']['triplet']['embeddings'].transpose(0, 1).contiguous()  # [embed_size, n, separate_fc_cnt]
        ske_feat = self.ske_model(([maps], labs, typs, vies, seqL))['training_feat']['triplet']['embeddings'].transpose(0, 1).contiguous()
        with torch.no_grad():
           sil_anchor_feat = self.frozen_tower(([sils], labs, typs, vies, seqL))['training_feat']['triplet']['embeddings'].transpose(0, 1).contiguous()
        proj_per_sil = sil_feat @ ske_feat.transpose(1, 2).contiguous()  # [embed_size, n, separate_fc_cnt] @ [embed_size, separate_fc_cnt, n]
        proj_per_ske = proj_per_sil.transpose(1, 2).contiguous()
        proj_per_sil_anchor = sil_feat @ sil_anchor_feat.transpose(1, 2).contiguous()

        retval = {
            'training_feat': {
                'sil_supcl': {'projections': proj_per_sil, 'targets': labs},
                'ske_supcl': {'projections': proj_per_ske, 'targets': labs},
                'sil_anchor_supcl': {'projections': proj_per_sil_anchor, 'targets': labs},
            },
            'visual_summary': {
                'image/sils': rearrange(pose * 255., 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': sil_feat
            }
        }
        return retval

