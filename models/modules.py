import torch
from torch import nn
from .model import BasicBlock, Bottleneck

def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    )

def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )

class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        # dilation rate
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        # 标准卷积(3*3)
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]

        out = torch.cat(out, 1)

        # 加融合机制需要开
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)

        return x * (1 - mask) + out * mask

def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )

            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()

        self.fuse = nn.ModuleList()

        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='bilinear', align_corners=True),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

            fuse_dim = (2 ** self.input_branches) - 1

            self.fuse.append(nn.Sequential(
                    nn.Conv2d(c * fuse_dim, c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                    nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                ))

    def forward(self, x):
        assert len(self.branches) == len(x)

        # 四层堆叠的残差模块
        x = [branch(b) for branch, b in zip(self.branches, x)]

        # 支路拼接
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                    torch.cat([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))], dim=1)
            )
        # 卷积融合
        x = [layer(a) for layer, a in zip(self.fuse, x_fused)]

        return x

# 原始
class REM_Net(nn.Module):
    def __init__(self, c=32, bn_momentum=0.1):
        super(REM_Net, self).__init__()

        # encoder
        self.conv1 = convrelu(2, 32, 3, 1, 1)      # 256*256
        self.conv2 = convrelu(32, 64, 5, 2, 2)     # 128*128
        self.conv3 = convrelu(64, 128, 5, 2, 1)    # 64*64
        self.conv4 = convrelu(128, 256, 5, 2, 1)   # 64*64

        self.AOT1 = nn.Sequential(*[AOTBlock(32, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT2 = nn.Sequential(*[AOTBlock(64, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT3 = nn.Sequential(*[AOTBlock(128, [1, 2, 4, 8]) for _ in range(1)])
        self.AOT4 = nn.Sequential(*[AOTBlock(256, [1, 2, 4, 8]) for _ in range(1)])

        # split_layer
        self.transition1 = nn.ModuleList([
            # branch1 (64 * 64)
            nn.Sequential(
                convrelu(256, c * (2 ** 0), 3, 1, 1),
            ),
            # branch2 (32 * 32)
            nn.Sequential(
                convrelu(256, c * (2 ** 1), 3, 1, 2),
            ),
            # branch3 (16 * 16)
            nn.Sequential(
                convrelu(256, c * (2 ** 2), 3, 1, 2),
                convrelu(c * (2 ** 2), c * (2 ** 2), 3, 1, 2),
            ),
            # branch4 (8 * 8)
            nn.Sequential(
                convrelu(256, c * (2 ** 3), 3, 1, 2),
                convrelu(c * (2 ** 3), c * (2 ** 3), 3, 1, 2),
                convrelu(c * (2 ** 3), c * (2 ** 3), 3, 1, 2),
            ),
        ])

        # interaction
        self.interaction = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(input_branches=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(input_branches=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(input_branches=4, output_branches=4, c=c, bn_momentum=bn_momentum),
        )

        # fusion
        self.transition2 = nn.ModuleList([
            nn.Sequential(
                convrelu(c * (2 ** 0), c * (2 ** 0), 3, 1, 1),
            ),
            nn.Sequential(
                convreluT(c * (2 ** 1), c * (2 ** 1), 4, 1),
            ),
            nn.Sequential(
                convreluT(c * (2 ** 2), c * (2 ** 2), 4, 1),
                convreluT(c * (2 ** 2), c * (2 ** 2), 4, 1),
            ),
            nn.Sequential(
                convreluT(c * (2 ** 3), c * (2 ** 3), 4, 1),
                convreluT(c * (2 ** 3), c * (2 ** 3), 4, 1),
                convreluT(c * (2 ** 3), c * (2 ** 3), 4, 1),
            ),
        ])

        # fusion final layer
        self.final_layer = convrelu(c * ((2 ** 4) - 1), 256, 3, 1, 1)

        # decoder
        self.up_conv1 = convrelu(256 + 256, 128, 3, 1, 1)
        self.up_conv2 = convrelu(128 + 128, 64, 3, 1, 1)
        self.up_conv3 = convreluT(64 + 64, 32, 4, 1)
        self.up_conv4 = convrelu(32 + 32, 1, 3, 1, 1)

    def forward(self, build, antenna):
        x = torch.cat([build, antenna], dim=1)

        # encoder
        x1 = self.conv1(x)
        x1 = self.AOT1(x1)
        x2 = self.conv2(x1)
        x2 = self.AOT2(x2)
        x3 = self.conv3(x2)
        x3 = self.AOT3(x3)
        x4 = self.conv4(x3)
        x4 = self.AOT4(x4)

        # split
        x = [
            self.transition1[0](x4),
            self.transition1[1](x4),
            self.transition1[2](x4),
            self.transition1[3](x4),
        ]

        # interaction
        x = self.interaction(x)

        # fusion
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[2]),
            self.transition2[3](x[3])
        ]

        # fusion_final_layer
        x = torch.cat(x, 1)
        x = self.final_layer(x)

        # decoder
        cat1 = torch.cat([x, x4], dim=1)
        up_1 = self.up_conv1(cat1)
        cat2 = torch.cat([up_1, x3], dim=1)
        up_2 = self.up_conv2(cat2)
        cat3 = torch.cat([up_2, x2], dim=1)
        up_3 = self.up_conv3(cat3)
        cat4 = torch.cat([up_3, x1], dim=1)
        up_4 = self.up_conv4(cat4)

        out = up_4

        return out

# Debug
def test():
    x = torch.randn((4, 1, 256, 256))
    y = torch.randn((4, 1, 256, 256))

    model = DeepAE(32,  0.1)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型的参数量为: {params}")
    model.cuda()
    preds = model(x.cuda(), y.cuda())
    print(preds.shape)


if __name__ == "__main__":
    test()
