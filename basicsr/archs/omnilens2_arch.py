import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import math
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .network_swinir import RSTB, window_partition, window_reverse
from .fema_utils import ResBlock, CombineQuantBlock, ResBlock_wz 
from .vgg_arch import VGGFeatureExtractor
import numbers
from .arch_util import default_init_weights, make_layer, pixel_unshuffle
from .swinir_arch import *
from .arch_util import default_init_weights, make_layer, pixel_unshuffle, patch_shuffle, patch_unshuffle


# Import necessary classes from vqcac_arch

class MultiScaleEncoder_light(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                # ResBlock(out_ch, out_ch, norm_type, act_type),
                # ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        if LQ_stage: 
            self.blocks.append(SwinLayers(**swin_opts))
            # self.blocks.append(PSALayers(**swin_opts))
            # self.blocks.append(RCABLayers(**swin_opts))
            # upsampler = nn.ModuleList()
            # # for i in range(2):
            #     in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            #     upsampler.append(nn.Sequential(
                #     nn.Upsample(scale_factor=2),
                #     nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                #     ResBlock(out_channel, out_channel, norm_type, act_type),
                #     ResBlock(out_channel, out_channel, norm_type, act_type),
                #     )
                # )
            #     res = res * 2
        
            # self.blocks += upsampler

        self.LQ_stage = LQ_stage

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)

        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)

        return outputs


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 **swin_opts,
                 ):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        # if LQ_stage: 
            # self.blocks.append(SwinLayers(**swin_opts))
            # self.blocks.append(NAFLayers(**swin_opts))
            # self.blocks.append(RRDBLayers(**swin_opts))
            # self.blocks.append(PSALayers(**swin_opts))
            # self.blocks.append(RCABLayers(**swin_opts))
            # upsampler = nn.ModuleList()
            # # for i in range(2):
            #     in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            #     upsampler.append(nn.Sequential(
                #     nn.Upsample(scale_factor=2),
                #     nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                #     ResBlock(out_channel, out_channel, norm_type, act_type),
                #     ResBlock(out_channel, out_channel, norm_type, act_type),
                #     )
                # )
            #     res = res * 2
        
            # self.blocks += upsampler

        self.LQ_stage = LQ_stage

    def forward(self, input):
        outputs = []
        x = self.in_conv(input)

        for idx, m in enumerate(self.blocks):
            x = m(x)
            outputs.append(x)

        return outputs


class MultiScaleDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 ):
        super().__init__()
        # self.use_warp = False
        self.upsampler = nn.ModuleList()
        
        # self.warp = nn.ModuleList()
        res =  input_res // (2 ** max_depth)
        self.maxdepth = max_depth
        for i in range(max_depth):
            in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            self.upsampler.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                ResBlock(out_channel, out_channel, norm_type, act_type),
                ResBlock(out_channel, out_channel, norm_type, act_type),
                )
            )
 
            self         
            res = res * 2

    def forward(self, input, enc_feature=None):
        x = input
        for idx, m in enumerate(self.upsampler):
            if enc_feature is not None:
                if idx == (self.maxdepth-1):
                    x = m(x)
                else:
                    x = m(x) + enc_feature[idx+1]
            else:
                x = m(x)
        return x

class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256, 
                blk_depth=6,
                num_heads=8,
                window_size=8,
                num_blk = 4,
                **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(num_blk):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w) 
        return x


@ARCH_REGISTRY.register()
class SwinUnet_psfprediction_cacstage(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 scale_factor=4,
                 use_semantic_loss=False,
                 use_residual=True,
                 **ignore_kwargs):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]


        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder 
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        encode_depth = int(np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0]))
        self.psf_encoder = MultiScaleEncoder(
                                67,     
                                encode_depth,  
                                self.gt_res // self.scale_factor, 
                                channel_query_dict,
                                norm_type, act_type, False
                            )
        
        self.psfpredict_encoder = MultiScaleEncoder(
                                in_channel,     
                                encode_depth,  
                                self.gt_res // self.scale_factor, 
                                channel_query_dict,
                                norm_type, act_type, False
                            )

        
        self.psffusion_encoder = MultiScaleEncoder_light(
                                67,     
                                encode_depth,  
                                self.gt_res // self.scale_factor, 
                                channel_query_dict,
                                norm_type, act_type, False
                            )                            
        #                     )                            

        self.cac_encoder = MultiScaleEncoder(
                                in_channel,     
                                encode_depth,  
                                self.gt_res // self.scale_factor, 
                                channel_query_dict,
                                norm_type, act_type, False
                            )
        # # self.swin_block = PSF_SwinLayers()
        self.swin_block = SwinLayers()


        # build decoder
        # self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            # self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        tt = gt_resolution // 2**self.max_depth
        ch_fuse = channel_query_dict[tt]

        self.fuse_conv = nn.Conv2d(ch_fuse*2, ch_fuse, 1)

        self.cacdecoder = MultiScaleDecoder(
                            in_channel,     
                            self.max_depth,  
                            self.gt_res, 
                            channel_query_dict,
                            norm_type, act_type
        )
        self.psfdecoder = MultiScaleDecoder(
                            in_channel,     
                            self.max_depth,  
                            self.gt_res, 
                            channel_query_dict,
                            norm_type, act_type
        )


        self.psf_out_conv = nn.Conv2d(out_ch, 67, 3, 1, 1)
        

        # build multi-scale vector quantizers 
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()
        self.cac_out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)


    def encode_and_decode_train(self, input):
        img = input
        psf_feats = self.psfpredict_encoder(input.detach())
        cac_feats = self.cac_encoder(input.detach())
        
        psf_feats = psf_feats[::-1]
        
        cac_feats = cac_feats[::-1]

        before_quant_feat = psf_feats[0]

        x_psfp = before_quant_feat
        x_psfp = self.psfdecoder(x_psfp)

        out_psf = self.psf_out_conv(x_psfp)
        psffusion_feats = self.psffusion_encoder(out_psf)
        psffusion_feats = psffusion_feats[::-1]

        x = cac_feats[0]
        x_psf = psffusion_feats[0]
        x = self.fuse_conv(torch.cat([x, x_psf], dim=1))
        x = self.swin_block(x)
        x = self.cacdecoder(x)
        out_img = self.cac_out_conv(x)

        return out_img, out_psf


    def encode_and_decode_test(self, input):
        img = input
        psf_feats = self.psfpredict_encoder(input.detach())
        cac_feats = self.cac_encoder(input.detach())
        
        psf_feats = psf_feats[::-1]
        
        cac_feats = cac_feats[::-1]
        before_quant_feat = psf_feats[0]
        x_psfp = before_quant_feat
        x_psfp = self.psfdecoder(x_psfp)

        out_psf = self.psf_out_conv(x_psfp)
        psffusion_feats = self.psffusion_encoder(out_psf)
        psffusion_feats = psffusion_feats[::-1]

        x = cac_feats[0]
        x_psf = psffusion_feats[0]
        x = self.fuse_conv(torch.cat([x, x_psf], dim=1))
        x = self.swin_block(x)
        x = self.cacdecoder(x)
        out_img = self.cac_out_conv(x)

        return out_img, out_psf

    @torch.no_grad()
    def test(self, input, weight_alpha=None):

        _, _, h_old, w_old = input.shape

        # output, _ = self.encode_and_decode_test(input, None, None)
        output, _ = self.encode_and_decode_test(input)
        if output is not None:
            output = output[..., :h_old, :w_old]
        # if output_vq is not None:
        #     output_vq = output_vq[..., :h_old, :w_old]

        return output

    def forward(self, input, weight_alpha=None):

        dec, dec_psf = self.encode_and_decode_train(input)

        return dec, dec_psf