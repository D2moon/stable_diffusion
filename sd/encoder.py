import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (BatchSize, channels, Height, Width) -> (BatchSize, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (BatchSize, 128, Height, Width) -> (BatchSize, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (BatchSize, 128, Height, Width) -> (BatchSize, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (BatchSize, 128, Height, Width) -> (BatchSize, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (BatchSize, 128, Height/2, Width/2) -> (BatchSize, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (BatchSize, 256, Height/2, Width/2) -> (BatchSize, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (BatchSize, 256, Height/2, Width/2) -> (BatchSize, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (BatchSize, 256, Height/4, Width/4) -> (BatchSize, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (BatchSize, 512, Height/4, Width/4) -> (BatchSize, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (BatchSize, 512, Height/4, Width/4) -> (BatchSize, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 512, Height/8, Width/8)
            nn.SiLU(),

            # (BatchSize, 512, Height/8, Width/8) -> (BatchSize, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (BatchSize, 8, Height/8, Width/8) -> (BatchSize, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        # x: (BatchSize, 3, Height, Width)
        # noise: (BatchSize, 4, Height/8, Width/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (PaddingLeft, PaddingRight, PaddingTop, PaddingBottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (BatchSize, 8, Height/8, Width/8) -> two tensors of shape(BatchSize, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (BatchSize, 4, Height/8, Width/8) -> (BatchSize, 4, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (BatchSize, 4, Height/8, Width/8) -> (BatchSize, 4, Height/8, Width/8)
        variance = torch.exp(log_variance)

        # (BatchSize, 4, Height/8, Width/8) -> (BatchSize, 4, Height/8, Width/8)
        stdev = torch.sqrt(variance)

        # Z=N(0, 1) -> Z*stdev + mean = N(mean, variance)
        # print(mean.shape, stdev.shape, noise.shape)
        x = mean + stdev * noise

        # Scale the output to [-1, 1]
        x *= 0.18215
        
        return x
    