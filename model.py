import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.embed = nn.parameter.Parameter(
            torch.FloatTensor(256, 26), requires_grad=True)
        nn.init.kaiming_uniform_(
            self.embed, mode='fan_in', nonlinearity='relu')
        self.concat = nn.parameter.Parameter(
            torch.FloatTensor(256, 100), requires_grad=True)
        nn.init.kaiming_uniform_(
            self.concat, mode='fan_in', nonlinearity='relu')

    # (b, 28, 100) => (b, 26, 100) and (b, 2, 100) =>
    # (b, 256, 100) | (b, 256, 100) and (b, 2, 100) => (b, 514, 100)
    def forward(self, sim):

        sim = sim.to(self.device)
        self.embed = self.embed.to(self.device)
        self.concat = self.concat.to(self.device)

        # (batch, 28, 100) => x[0] : (batch, 26, 100) and x[1] : (batch, 2, 100)
        x = torch.split(sim, 26, dim=1)

        x_ori = x[0]
        c_total = x[1]
        x_ori = x[0].to(self.device)
        c_total = x[1].to(self.device)

        x_total = torch.zeros([sim.size(0), 256, 100], dtype=torch.float)

        x_total = x_total.to(self.device, dtype=torch.float)

        for i in range(sim.size(0)):
            # this should be (b, 256, 100)
            x_total[i] = torch.mm(self.embed, x_ori[i])

        # for concat

        concat_batch = torch.zeros([sim.size(0), self.concat.size(
            0), self.concat.size(1)], dtype=torch.float)
        concat_batch = concat_batch.to(self.device)
        for i in range(sim.size(0)):
            concat_batch[i] = self.concat

        x_total = torch.cat((x_total, concat_batch), dim=1)
        x_total = torch.cat((x_total, c_total), dim=1)

        return x_total


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    # conv1d -> InstanceNorm -> Relu -> Conv1d -> IN -> +x -> Relu

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LayerNorm([dim_out, 100], elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_out, dim_out, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LayerNorm([dim_out, 100], elementwise_affine=True),
        )

        ### LayerNorm and Relu
        self.after = nn.Sequential(
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = x + self.main(x)
        out = self.after(out)
        return out


# input : simplex / output : T
class Generator(nn.Module):
    """ Generator block for Conv1d """

    def __init__(self, out_dim=32, c_dim=2):
        super(Generator, self).__init__()

        layers = []

        c_dim = 4
        out_dim = 64

        self.Multiply = Multiply()

        # Concatenate x and c
        # So, the dim. is 258 (hidden_size(default=256) + c dim(default=2))
        layers.append(self.Multiply)

        # Encoding
        layers.append(nn.Conv1d(512+c_dim, out_dim, 1, 1, 0))
        layers.append(nn.LayerNorm([out_dim, 100], elementwise_affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv1d(out_dim, out_dim*2, 1, 1, 0))
        out_dim = out_dim * 2
        layers.append(nn.LayerNorm([out_dim, 100], elementwise_affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv1d(out_dim, out_dim*2, 1, 1, 0))
        out_dim = out_dim * 2
        layers.append(nn.LayerNorm([out_dim, 100], elementwise_affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv1d(out_dim, out_dim*2, 1, 1, 0))
        out_dim = out_dim * 2
        layers.append(nn.LayerNorm([out_dim, 100], elementwise_affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Bottleneck
        for i in range(5):
            layers.append(ResidualBlock(dim_in=out_dim, dim_out=out_dim))

        # Decoding
        # output will be (64, 26, 100)
        layers.append(nn.Conv1d(out_dim, 26, 1, 1, 0))
        layers.append(nn.LayerNorm([26, 100], elementwise_affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Softmax(dim=-2))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c, ):   # x : simplex, c : label
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1)
        c = c.repeat(1, 1, x.size(2))
        x = torch.cat([x, c], dim=1)
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN for Conv1d."""

    def __init__(self, in_size=100, out_dim=32, c_dim=2):
        super(Discriminator, self).__init__()

        out_dim = 64

        layers = []
        layers.append(nn.Dropout(p=0.5, inplace=False))
        layers.append(nn.Conv1d(512, out_dim, 14, 2, 6))
        layers.append(nn.LeakyReLU(negative_slope=0.02, inplace=True))

        layers.append(nn.Conv1d(out_dim, out_dim*2, 14, 2, 6))
        out_dim = out_dim * 2
        layers.append(nn.LayerNorm([out_dim, 25], elementwise_affine=True))
        layers.append(nn.LeakyReLU(negative_slope=0.02, inplace=True))

        layers.append(nn.Conv1d(out_dim, out_dim*2, 15, 2, 7))
        out_dim = out_dim * 2
        layers.append(nn.LayerNorm([out_dim, 13], elementwise_affine=True))
        layers.append(nn.LeakyReLU(negative_slope=0.02, inplace=True))

        layers.append(nn.Conv1d(out_dim, out_dim*2, 11, 2, 5))
        out_dim = out_dim * 2
        layers.append(nn.LayerNorm([out_dim, 7], elementwise_affine=True))
        layers.append(nn.LeakyReLU(negative_slope=0.02, inplace=True))

        self.main = nn.Sequential(*layers)

        # for D_src
        self.conv1 = nn.Conv1d(out_dim, 1, 6, 1, 0, bias=False)

        # for D_cls
        self.conv2 = nn.Conv1d(out_dim, c_dim, 7, 1, 0, bias=False)

    def forward(self, x):
        h = self.main(x)

        out_src = self.conv1(h)
        out_cls = self.conv2(h)

        # out_cls reshape to [4, 2]
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
