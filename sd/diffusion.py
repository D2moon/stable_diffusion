import torch
from torch.nn import functional as F
from torch import nn
from attention import SelfAttention, AcrossAttention


class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embd: int):
        super().__init__()

        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, time: torch.Tensor):
        # time: (1, 320) -> (1, 1280)
        x = self.linear_1(time)

        x = F.silu(x)

        x = self.linear_2(x)

        return x
    

class SwitchSequantial(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x



class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Module([
            # (BatchSize, 4, Height / 8, Width / 8)
            SwitchSequantial(nn.Conv2d(4, 320, kernel_size=3, stride=1, padding=1)),

            SwitchSequantial(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequantial(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (BatchSize, 320, Height / 8, Width / 8) -> (BatchSize, 320, Height / 16, Width / 16)
            SwitchSequantial(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequantial(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequantial(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (BatchSize, 640, Height / 16, Width / 16) -> (BatchSize, 640, Height / 32, Width / 32)
            SwitchSequantial(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequantial(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequantial(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (BatchSize, 1280, Height / 32, Width / 32) -> (BatchSize, 1280, Height / 64, Width / 64)
            SwitchSequantial(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequantial(UNET_ResidualBlock(1280, 1280)),

            # (BatchSize, 1280, Height / 64, Width / 64) -> (BatchSize, 1280, Height / 64, Width / 64)
            SwitchSequantial(UNET_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequantial([
            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280, 1280),
        ])

        self.decoder = nn.ModuleList([
            # (BatchSize, 2560, Height / 64, Width / 64) -> (BatchSize, 1280, Height / 64, Width / 64)
            SwitchSequantial(UNET_ResidualBlock(2560, 1280)),

            SwitchSequantial(UNET_ResidualBlock(2560, 1280)),

            SwitchSequantial(UNET_ResidualBlock(2560, 1280), UpSample(2)),

            SwitchSequantial(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
        ])

        



class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
