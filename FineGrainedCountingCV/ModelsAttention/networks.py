import sys
import torch.nn as nn
import torch
import math
from torchvision import models
from ModelsAttention.axial_attention import AxialAttention


class HOUR_GLASS_PROP(nn.Module):
    '''
    '''
    def __init__(self, i_cn, o_cn, _iter=1, att_cn=0, prop=False):
        super(HOUR_GLASS_PROP, self).__init__()
        self._iter = _iter
        self.prop = prop
        self.att_cn = att_cn

        dim = i_cn + att_cn
        
        self.conv1 = nn.Sequential(nn.Conv2d(dim, 64, 5, 1, 2),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 5, 1, 2),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.MaxPool2d(2))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4,stride=2,padding=1,output_padding=0,bias=True),
                                  nn.LeakyReLU(0.1, inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, dim, 4,stride=2,padding=1,output_padding=0,bias=True),
                                  nn.LeakyReLU(0.1, inplace=True))
        self.reg = nn.Conv2d(dim, o_cn, 5, 1, 2)

        self._initialize_weights()

    def forward(self, fea, att=None):
        
        if self.att_cn > 0:
            fea = torch.cat((fea, att), 1)

        if self.prop:
            weight = torch.zeros_like(att)
            weight[att >= 0.001] = 1
            weight[att < 0.001] = 0.2
        for i in range(self._iter):
            if self.prop:
                
                w_fea = fea*weight
                conv1_fea = self.conv1(w_fea)
            else:
                conv1_fea = self.conv1(fea)
            conv2_fea = self.conv2(conv1_fea)
            deconv1_fea = self.deconv1(conv2_fea) + conv1_fea
            fea = self.deconv2(deconv1_fea) + fea

        self.fea = fea
        refined_smap = self.reg(fea)
        return refined_smap

    def _initialize_weights(self):
        initialize(self)


class CSRNet_TWO(nn.Module):
    def __init__(self, i_cn, o_cn):
        super(CSRNet_TWO, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.backend_seg = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.output_layer_seg = nn.Conv2d(64, o_cn, kernel_size=1)
        self._initialize_weights()
        if True:
            mod = models.vgg16(pretrained = True)
            fs = self.frontend.state_dict()
            ms = mod.state_dict()
            for key in fs:
                fs[key] = ms['features.'+key]
            self.frontend.load_state_dict(fs)
        else:
            print("Don't pre-train on ImageNet")

    def forward(self,x):
        
        x = self.frontend(x)
        
        self.smap_fea = self.backend_seg(x)
        
        self.dmap_fea = self.backend(x)
        
        
        x = self.output_layer(self.dmap_fea)
        
        smap = self.output_layer_seg(self.smap_fea)
        
        
        #x = F.interpolate(x, scale_factor=8)
        return x, smap

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,norm=False,dilation = False, dropout=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def initialize(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# Attention modules for CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
            
            
            
#Creating Axial-Attention block (Spatial Aggregator)
class SpatialAggregator(nn.Module):
    def __init__(self, dim, dim_index, heads = 4, num_dimensions = 2, n_att_layers = 16, o_cn=2,_iter = 3):
        super(SpatialAggregator, self).__init__()
        #self.att = AxialAttention(
        #                            dim = 4,               # embedding dimension
        #                            dim_index = 1,         # where is the embedding dimension
        #                            dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
        #                            heads = 1,             # number of heads for multi-head attention
        #                            num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
        #                            sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
        #                            )
        #self.spatial_agg = nn.Sequential(*[AxialAttention(dim=dim, dim_index=dim_index, heads=heads, num_dimensions=2, sum_axial_out=True) for _ in range(n_att_layers)])
        
        self.conv1 = nn.Sequential(nn.Conv2d(dim, 64, 5, 1, 2),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 5, 1, 2),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.MaxPool2d(2))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4,stride=2,padding=1,output_padding=0,bias=True),
                                  nn.LeakyReLU(0.1, inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, dim, 4,stride=2,padding=1,output_padding=0,bias=True),
                                  nn.LeakyReLU(0.1, inplace=True))
        
        self.ca1 = ChannelAttention(in_planes = dim)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(in_planes = 64)
        self.sa2 = SpatialAttention()
        self._iter = _iter
        
        
        
        
        self.reg = nn.Conv2d(dim, o_cn, 5, 1, 2)
        
    def forward(self, fea,att):
        fea = torch.cat((fea, att), 1)
        
        #weight = torch.zeros_like(att)
        #weight[att >= 0.001] = 1
        #weight[att < 0.001] = 0.2
        
        for _ in range(self._iter):
            #CBAM on FEA
            att_fea = self.ca1(fea)*fea
            att_fea = self.sa1(att_fea)*att_fea


            conv1_fea = self.conv1(fea)

            #CBAM on conv1_FEA
            att_conv1_fea = self.ca2(conv1_fea)*conv1_fea
            att_conv1_fea = self.sa2(att_conv1_fea) * att_conv1_fea


            conv2_fea = self.conv2(conv1_fea)
            deconv1_fea = self.deconv1(conv2_fea) + att_conv1_fea
            fea = self.deconv2(deconv1_fea) + att_fea

        self.fea = fea
        refined_smap = self.reg(fea)
        
        return refined_smap