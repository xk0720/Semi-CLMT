from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .resize_swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, config, img_size=352, num_classes=2, drop_rate=0.5, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.img_size = img_size

        # self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
        #                                     patch_size=config.MODEL.SWIN.PATCH_SIZE,
        #                                     in_chans=config.MODEL.SWIN.IN_CHANS,
        #                                     num_classes=self.num_classes,
        #                                     embed_dim=config.MODEL.SWIN.EMBED_DIM,
        #                                     depths=config.MODEL.SWIN.DEPTHS,
        #                                     num_heads=config.MODEL.SWIN.NUM_HEADS,
        #                                     window_size=config.MODEL.SWIN.WINDOW_SIZE,
        #                                     mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #                                     qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        #                                     qk_scale=config.MODEL.SWIN.QK_SCALE,
        #                                     drop_rate=config.MODEL.DROP_RATE,
        #                                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #                                     ape=config.MODEL.SWIN.APE,
        #                                     patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #                                     use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        self.swin_unet = SwinTransformerSys(img_size=self.img_size,
                                            patch_size=4,
                                            in_chans=3,
                                            num_classes=self.num_classes,
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=11,
                                            mlp_ratio=4.0,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=drop_rate,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_weights(self, pretrained_path='pretrained_swin_weights/model_weights.pth'):
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model" not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            # print(msg)
            return
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")

        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)

        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3 - int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k: v})
        '''
        Check...
        '''
        # print(list(full_dict.keys()))

        for k in list(full_dict.keys()):
            if k in model_dict:
                #                 if k == 'layers.0.blocks.1.attn_mask':
                #                     print('k exists')
                #                     print(k)
                #                     print(full_dict[k])
                #                     raise Exception
                # print('Exception is coming')
                # raise Exception
                if full_dict[k].shape != model_dict[k].shape:
                    # 尝试插值一下???
                    if k.split('.')[-1] == 'relative_position_bias_table':

                        # 'bicubic' 插值 based on swintransformer V2 paper
                        print('bicubic插值')
                        N, nH = full_dict[k].shape
                        resize_feature = torch.transpose(full_dict[k], 0, 1)
                        resize_feature = resize_feature.contiguous().view(1, nH, int(math.sqrt(N)), int(math.sqrt(N)))
                        resize_feature = F.interpolate(input=resize_feature,
                                                       size=(int(math.sqrt(model_dict[k].shape[-2])),
                                                             int(math.sqrt(model_dict[k].shape[-2]))),
                                                       mode='bicubic')
                        full_dict[k] = torch.transpose(resize_feature.view(nH, model_dict[k].shape[-2]), 0, 1)

                        # full_dict[k] = F.interpolate(input=full_dict[k].unsqueeze(0).unsqueeze(0),
                        #                             size=model_dict[k].shape[-2:], mode='bilinear').squeeze(0).squeeze(0)

                    else:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

                    # print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                    # del full_dict[k]

        msg = self.swin_unet.load_state_dict(full_dict, strict=False)

        # raise Exception
        # print(msg)

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
