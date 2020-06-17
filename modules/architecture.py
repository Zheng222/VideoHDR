import torch.nn as nn
from . import block as B
from torch.nn import functional as F
import torch
from modules import NLmodule

class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()

        n_feats = 32

        modules_head = [B.conv_layer(3, n_feats, 3)]

        modules_body = [B.IMDModule(n_feats) for _ in range(4)]
        modules_body.append(B.conv_layer(n_feats, n_feats, 3))

        modules_tail = [B.conv_layer(n_feats, 3, 3)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x

class RIRN(nn.Module):
    def __init__(self, opt):
        super(RIRN, self).__init__()

        n_resgroups = opt.n_resgroups
        n_resblocks = opt.n_resblocks
        n_feats = opt.nf

        # define head module
        modules_body = [B.ResidualGroup(n_feats, n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(B.conv_layer(n_feats, n_feats, 3))

        # define tail module
        modules_tail = [B.pixelshuffle_block(n_feats, opt.n_colors)]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        res = self.body(x)
        res +=x

        x = self.tail(res)
        return x

class VideoSR(nn.Module):
    def __init__(self, opt):
        super(VideoSR, self).__init__()
        self.opt = opt
        self.center = opt.nframes // 2
        self.nf = opt.nf
        self.nframes = opt.nframes

        ### extract features (for each frame)

        self.conv_first = B.conv_layer(3, self.nf, 3)

        self.feature_extraction = nn.Sequential(B.IMDModule(in_channels=self.nf),
                                                B.IMDModule(in_channels=self.nf),
                                                B.IMDModule(in_channels=self.nf),
                                                B.IMDModule(in_channels=self.nf),
                                                B.IMDModule(in_channels=self.nf))

        self.align = NLmodule.RegionNONLocalBlock(in_channels=opt.nf)

        ### fusion
        self.fusion = B.conv_layer(self.nframes * self.nf, self.nf, 1)

        ### reconstruction
        self.recon = RIRN(self.opt)

        ### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):

        B, T, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        fea = self.feature_extraction(fea)
        fea = fea.view(B, T, -1, H, W)

        ### align
        # ref feature
        ref_fea = fea[:, self.center, :, :, :].clone()
        aligned_fea = []
        for i in range(T):
            aligned_fea.append(self.align(ref_fea, fea[:, i, :, :, :].clone()))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, T, C, H, W]

        aligned_fea = aligned_fea.view(B, -1, H, W)
        aligned_fea = self.fusion(aligned_fea)

        ### recon
        out = self.recon(aligned_fea)
        base = F.interpolate(x_center, scale_factor=4, mode='bicubic', align_corners=False)
        out += base
        return out
