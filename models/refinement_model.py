import torch.nn as nn
import torch.nn.functional as F
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class RefineNet(nn.Module):
    def __init__(self, num_residual_blocks):
        super(RefineNet, self).__init__()

        model = [nn.Conv2d(3, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out) #激活函数，输出范围在-1到1之间。
        return out


def test():
    lptn_model = RefineNet(num_residual_blocks=1)
    lptn_model = lptn_model.cuda()
    input_t = torch.randn(1, 3, 1024, 1024).cuda()

    output_t = lptn_model(input_t)
    
    print('output.shape =', output_t.shape)

if __name__ == "__main__":
    test()


