import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d

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


#注意：使用的时候需要在外层先进行LayerNorm，结束之后不要忘记还有skip connection需要实现。
#从Restormer中修改过来
class WFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(WFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)

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

class WFFN_NAF_V2(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, drop_out_rate=0.):

        super(WFFN_NAF_V2, self).__init__()

        self.hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, self.hidden_features, kernel_size=1, bias=bias)

        self.sg = SimpleGate()

        self.wavelet_ll_weight = nn.Parameter(torch.ones((self.hidden_features, 1, 1)))

        self.project_out = nn.Conv2d(self.hidden_features//2, dim, kernel_size=1, bias=bias)

        self.dwt = DWT()
        self.iwt = IWT()

        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, y):


        x = self.project_in(y)

        dwt_feats = self.dwt(x)
        ll_freq = dwt_feats[:,0:self.hidden_features,:,:]
        ll_freq_weighted = self.wavelet_ll_weight * ll_freq
        dwt_feats[:,0:self.hidden_features,:,:] = ll_freq_weighted
        
        iwt_feats = self.iwt(dwt_feats)
        x = self.sg(iwt_feats)

        x = self.project_out(x)

        x = self.dropout2(x)

        result = x

        return result

class WFFN_NAF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, drop_out_rate=0.):

        super(WFFN_NAF, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.sg = SimpleGate()

        self.wavelet_weight = nn.Parameter(torch.ones((hidden_features *4, 1, 1)))

        self.project_out = nn.Conv2d(hidden_features//2, dim, kernel_size=1, bias=bias)

        self.dwt = DWT()
        self.iwt = IWT()

        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, y):


        x = self.project_in(y)

        dwt_feats = self.dwt(x)
        dwt_feats = self.wavelet_weight * dwt_feats
        iwt_feats = self.iwt(dwt_feats)
        x = self.sg(iwt_feats)

        x = self.project_out(x)

        x = self.dropout2(x)

        result = x

        return result
        
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, drop_out_rate=0.):
        super(NAFFeedForward, self).__init__()

        # SimpleGate
        self.sg = SimpleGate()

        c = dim
        FFN_Expand = ffn_expansion_factor
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.naf_norm2 = LayerNorm2d(c)

        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, y):

        x = self.conv4(self.naf_norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        result = y + x * self.gamma

        return result
    
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

    #wffn = WFFN(dim=4,ffn_expansion_factor=1, bias=False).to(device=device)
    wffn = WFFN_NAF_V2(dim=4,ffn_expansion_factor=2, bias=False).to(device=device)
    wffn_res = wffn(inp_img)
    print(wffn_res.shape)
