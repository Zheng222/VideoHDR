import torch.nn as nn
from modules import block as B


class VideoColor(nn.Module):

    def __init__(self):
        super(VideoColor, self).__init__()

        n_feats = 32

        modules_head = [B.conv_layer(3, n_feats, 3)]

        modules_body = [B.IMDModule(n_feats) for _ in range(4)]
        modules_body.append(B.conv_layer(n_feats, n_feats, 3))

        modules_tail = [B.conv_layer(n_feats, 3, 3)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        B, T, C, H, W = x.size()  # N video frames
        x = self.head(x.view(-1, C, H, W))

        res = self.body(x)
        res += x

        x = self.tail(res).view(B, T, C, H, W)
        return x
