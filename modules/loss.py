import torch
import torch.nn as nn
import kornia


class L1loss(nn.Module):
    def __init__(self):
        super(L1loss, self).__init__()
        self.criterion = nn.L1Loss()

    def rgb_to_yuv(self, input):
        r, g, b = torch.chunk(input, chunks=3, dim=-3)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b
        yuv_img = torch.cat((y, u, v), -3)
        return yuv_img

    def forward(self, input, reference):
        input_yuv = self.rgb_to_yuv(input)
        reference_yuv = self.rgb_to_yuv(reference)
        return self.criterion(input_yuv, reference_yuv)


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.criterion = kornia.losses.SSIM(window_size=11, max_val=1.0, reduction='mean')

    def forward(self, input, reference):

        return self.criterion(input, reference)
