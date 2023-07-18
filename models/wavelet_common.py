import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用哈尔 haar 小波变换来实现二维离散小波
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class WFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(WFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.wavelet_weight = nn.Parameter(torch.ones((hidden_features * 2 *4, 1, 1)))

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.dwt = DWT()
        self.iwt = IWT()
    def forward(self, x):
        x = self.project_in(x)

        dwt_feats = self.dwt(x)
        dwt_feats = self.wavelet_weight * dwt_feats
        iwt_feats = self.iwt(dwt_feats)
        x1, x2 = self.dwconv(iwt_feats).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    

if __name__ == '__main__':
    img_channel = 4
    width = 32


    device = torch.device('cuda:0')

    #net = fftformer(inp_channels=4, out_channels=3, dim=8).to(device=device)
    dwt = DWT().to(device=device)

    iwt = IWT().to(device=device)
    inp_img = torch.randn(1, 4, 1024, 1024).to(device=device)
    dwt_output = dwt(inp_img)
    print(dwt_output.shape)
    iwt_output = iwt(dwt_output)
    print(iwt_output.shape)

    wffn = WFFN(dim=4,ffn_expansion_factor=1, bias=False).to(device=device)
    wffn_res = wffn(inp_img)
    print(wffn_res.shape)
