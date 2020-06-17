import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
        else:
            conv_nd = nn.Conv1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.shared = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                              kernel_size=3, stride=1, padding=1)


    def forward(self, x, x_t):
        """
        :param x: (b, c, t, h, w), current
        :param x_t: (b, c, t, h, w), t
        :return:
        """
        batch_size = x_t.size(0)
        g_x_t = self.g(x_t).view(batch_size, self.inter_channels, -1)  # N, C, (H*W)
        g_x_t = g_x_t.permute(0, 2, 1)  # N, (H * W), C

        theta_x = self.shared(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # N, (H * W), C
        phi_x_t = self.shared(x_t).view(batch_size, self.inter_channels, -1)  # N, C, (H*W)
        f = torch.matmul(theta_x, phi_x_t)  # N, (H*W), (H*W)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x_t)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2)


class RegionNONLocalBlock(nn.Module):
    def __init__(self, in_channels, grid=[2, 2]):  # train: [2, 2];  test: [6, 6]
        super(RegionNONLocalBlock, self).__init__()
        self.non_local_block = NONLocalBlock2D(in_channels)
        self.grid = grid

    def forward(self, x, x_t):
        batch_size, _, height, width = x.size()
        input_row_list = x.chunk(self.grid[0], dim=2)
        input_t_row_list = x_t.chunk(self.grid[0], dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = row.chunk(self.grid[1], dim=3)
            input_t_grid_list_of_a_row = input_t_row_list[i].chunk(self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                grid_t = input_t_grid_list_of_a_row[i]
                grid = self.non_local_block(grid, grid_t)
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)
        return output
