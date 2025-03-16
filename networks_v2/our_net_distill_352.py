import math
from networks_v2.net_config_unet_skip_expand_decoder_sys import Swinv2Config
import torch
import torch.nn as nn
from networks_v2.net_swin_v2 import SwinUnet as ViT_seg
import torch.nn.functional as F


class Projection_head(nn.Module):
    def __init__(self):
        super(Projection_head, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=(1, 1), stride=(1, 1))
        self.act_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels=768, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.init_weights()

    def forward(self, x):
        # return self.conv(x)
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        return x

    def init_weights(self):
        self.conv_1.apply(init_weights)
        self.conv_2.apply(init_weights)


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    # print("Iinitialing weights now")
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Our_Net_distill(nn.Module):
    def __init__(self, img_size=352, num_classes=2, Glas_data=False, MoNuSeg=False, RITE=False):
        super(Our_Net_distill, self).__init__()

        config = Swinv2Config()

        if Glas_data:
            config.patches_resolution = [128 // 4, 128 // 4]
            config.image_size = 128
            config.window_size = 4
        elif MoNuSeg:
            config.patches_resolution = [224 // 4, 224 // 4]
            config.image_size = 224
            config.window_size = 7
        elif RITE:
            config.patches_resolution = [512 // 4, 512 // 4]
            config.image_size = 512
            config.window_size = 16

        # print(config.image_size)
        # print(config.patches_resolution)

        self.model = ViT_seg(config=config, img_size=config.image_size, num_classes=num_classes).cuda()
        self.model.load_weights()
        # self.projection_head = Projection_head()

        self.conv_list = []
        self.channels = [96, 96, 192, 384]
        # for channel in self.channels:
        #     self.conv = conv_layer(in_channels=channel)
        #     self.conv_list.append(self.conv)

        # self.conv_1 = conv_layer(in_channels=96)
        # self.conv_2 = conv_layer(in_channels=96)
        # self.conv_3 = conv_layer(in_channels=192)
        # self.conv_4 = conv_layer(in_channels=384)

        # self.is_ds = True # flag to control

    # only satisfy adding projection head behind the encoder
    def forward(self, x):
        # encoder and bottleneck
        x, concat_feature_list = self.model.swin_unet.forward_features(x)

        # the output of the bottleneck
        feature_map = x

        B, N, C = x.shape
        # projection = self.projection_head(
        #     torch.transpose(x, 1, 2).contiguous().view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        # )

        # decoder
        x = self.model.swin_unet.forward_up_features(x, concat_feature_list)

        # Final expand 4x
        x = self.model.swin_unet.up_x4(x)

        '''
                return_layer_list
                torch.Size([2, 196, 384])
                torch.Size([2, 784, 192])
                torch.Size([2, 3136, 96])
                torch.Size([2, 3136, 96])    
        '''

        conv_return_layer_list = []
        # size_list = [56, 56, 28, 14]

        #         for idx, conv_ in enumerate(self.conv_list):
        #             B, N, C = return_layer_list[len(self.conv_list) - idx - 1].shape
        #             feature = return_layer_list[len(self.conv_list) - idx - 1].permute(0, 2, 1).contiguous().\
        #                 view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))

        #             output = conv_(feature)
        #             conv_return_layer_list.append(output)

        # B, N, C = return_layer_list[3].shape
        # feature = return_layer_list[3].permute(0, 2, 1).contiguous(). \
        #     view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        # conv_return_layer_list.append(self.conv_1(feature))
        #
        # B, N, C = return_layer_list[2].shape
        # feature = return_layer_list[2].permute(0, 2, 1).contiguous(). \
        #     view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        # conv_return_layer_list.append(self.conv_2(feature))
        #
        # B, N, C = return_layer_list[1].shape
        # feature = return_layer_list[1].permute(0, 2, 1).contiguous(). \
        #     view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        # conv_return_layer_list.append(self.conv_3(feature))
        #
        # B, N, C = return_layer_list[0].shape
        # feature = return_layer_list[0].permute(0, 2, 1).contiguous(). \
        #     view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        # conv_return_layer_list.append(self.conv_4(feature))

        # conv_return_layer_list.append(conv_(feature))

        return x, None, feature_map

        # if self.is_ds:
        #     conv_return_layer_list = []
        #     for idx, conv_ in enumerate(self.conv_list):
        #         conv_return_layer_list.append(conv_(return_layer_list[len(self.conv_list) - idx - 1]))
        #
        #     return x, projection, conv_return_layer_list
        # else:
        #     return x, projection


