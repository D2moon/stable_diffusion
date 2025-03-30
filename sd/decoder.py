import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:   
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BatchSize, in_channels, Height, Width)

        residue = x

        x = self.groupnorm_1(x)

        s = F.silu(x)

        x = self.conv_1(s)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BatchSize, channels, Height, Width)

        residue = x

        n, c, h, w = x.shape

        # (BatchSize, channels, Height, Width) -> (BatchSize, channels, Height*Width)
        x = x.view(n, c, h*w)

        # (BatchSize, channels, Height*Width) -> (BatchSize, Height*Width, channels)
        x = x.transpose(-1, -2)

        # (BatchSize, Height*Width, channels) -> (BatchSize, Height*Width, channels)
        x = self.attention(x)

        # (BatchSize, Height*Width, channels) -> (BatchSize, channels, Height*Width)
        x = x.transpose(-1, -2)
        
        # (BatchSize, channels, Height*Width) -> (BatchSize, channels, Height, Width)
        x = x.view((n, c, h, w))

        x += residue

        return x


class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super.__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/4, Width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (BatchSize, 512, Height/4, Width/4) -> (BatchSize, 512, Height/2, Width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (BatchSize, 256, Height/4, Width/4) -> (BatchSize, 256, Height, Width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLu(),

            # (BatchSize, 128, Height, Width) -> (BatchSize, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BatchSize, 4, Height/8, Width/8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # x: (BatchSize, 3, Height, Width)
        return x