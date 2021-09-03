import os
import torch.nn as nn
import torch.optim as optim
from modules.base_networks import *
from torchvision.transforms import *
import torch.nn.functional as F
from modules.dbpns import Net as DBPNS


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor):
        super(Net, self).__init__()
        # base_filter=256
        # feat=64
        self.nFrames = nFrames

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)  # h*w*3-->h*w*256
        self.feat_all = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        # Further Feature Extraction
        feature_extraction = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(3)]
        self.res_feat_ext = nn.Sequential(*feature_extraction)

        ### SISR
        self.DBPN = DBPNS(base_filter, feat, num_stages, scale_factor)

        # MSD
        self.alignment1 = MSD(nf=64, groups=8, dilation=1)  #

        # Fusion after align
        self.fusion1 = ConvBlock(feat * 3, base_filter, 3, 1, 1, activation='prelu', norm=None)

        # Res-Block1,h*w*256-->H*W*64
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        # Res-Block3，downsample
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)

        # Temporal attention
        self.TAtt = TA_Fusion(nf=64, center_fea=0)

        # Reconstruction
        self.output = ConvBlock(3 * feat, num_channels, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, neigbor):
        B, C, H, W = x.size()  # B * 3 * patchsize * patchsize(ps), LRs
        ### initial feature extraction
        feat_input = self.feat0(x)  # h*w*256
        # B * N * 3 * ps * ps
        feat_all = torch.stack([neigbor[0], neigbor[1], x, neigbor[2], neigbor[3]], dim=1)
        feat_all = self.feat_all(feat_all.view(-1, C, H, W))  # 【N 64 ps ps】
        feat_all = self.res_feat_ext(feat_all)
        feat_all = feat_all.view(B, self.nFrames, -1, H, W)  # [1, N, 64, ps, ps]

        # regroup,(135),(234)
        ref_fea = feat_all[:, 2, :, :, :].clone()
        group1_fea_all = [feat_all[:, 0, :, :, :].clone(), feat_all[:, 2, :, :, :].clone(), feat_all[:, 4, :, :, :].clone()]  # list
        group2_fea_all = [feat_all[:, 1, :, :, :].clone(), feat_all[:, 2, :, :, :].clone(), feat_all[:, 3, :, :, :].clone()]


        # align,(135)
        group1_aligned_fea = []
        for i in range(len(group1_fea_all)):
            group1_neigbor_fea = group1_fea_all[i]
            group1_aligned_fea.append(self.alignment1(group1_neigbor_fea, ref_fea))
        # align,(234)
        group2_aligned_fea = []
        for i in range(len(group2_fea_all)):
            group2_neigbor_fea = group2_fea_all[i]
            group2_aligned_fea.append(self.alignment1(group2_neigbor_fea, ref_fea))

        # fusion,h*w*256
        group1_fus_fea = self.fusion1(torch.cat([group1_aligned_fea[0], group1_aligned_fea[1], group1_aligned_fea[2]], dim=1))
        group2_fus_fea = self.fusion1(torch.cat([group2_aligned_fea[0], group2_aligned_fea[1], group2_aligned_fea[2]], dim=1))

        feat_frame = [group1_fus_fea, group2_fus_fea]

        ####Projection
        Ht = []
        for j in range(2):
            h0 = self.DBPN(feat_input)
            h1 = self.res_feat1(feat_frame[j])

            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            Ht.append(h)
            feat_input = self.res_feat3(h)

        ####Reconstruction
        out = torch.stack([self.DBPN(feat_input), Ht[0], Ht[1]], dim=1)  # [1,4, 64, ps, ps]
        modulated_out = self.TAtt(out)  # 【1 64*4 ps ps】

        output = self.output(modulated_out)

        return output
