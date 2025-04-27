import torch
from torch.nn import functional as F
from torch import nn
from attention import SelfAttention, CrossAttention


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
    

class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # feature: (BatchSize, in_channels, Height, Width)
        # time: (1, 1280)
        
        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)

        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head: int, n_embd: int, d_context: int = 1280):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: (BatchSize, Features, Height, Width)
        # context: (BatchSize, SeqLen, Dim)

        residue_long = x

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (BatchSize, Features, Height, Width) -> (BatchSize, Features, Height*Width)
        x = x.view((n, c, h*w))

        # (BatchSize, Features, Height*Width) -> (BatchSize, Height*Width, Features)
        # why? 把图片的像素点数量当成序列长度，通道数当成特征维度
        x = x.transpose(-1, -2)

        # Normalization + Self Attention with skip connection
        residue_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short

        # Normalization + Cross Attention with skip connection
        residue_short = x

        x = self.layernorm_2(x)

        # Cross Attention
        self.attention_2(x, context)

        x += residue_short

        residue_short = x

        # Normalization + FF with GeGLU and skip connection

        x = self.layernorm_3(x)

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


class UpSample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (BatchSize, channels, Height, Width) -> (BatchSize, channels, Height*2, Width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv(x)

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

            SwitchSequantial(UNET_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequantial(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequantial(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequantial(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequantial(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequantial(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequantial(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequantial(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequantial(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequantial(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

        
class UNET_OutputLayer(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (BatchSize, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)

        x = F.silu(x)

        # (BatchSize, 320, Height / 8, Width / 8) -> (BatchSize, 4, Height / 8, Width / 8)
        x = self.conv(x)

        return x


class Diffusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (BatchSize, 4, Height / 8, Width / 8)
        # context: (BatchSize, SeqLen, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
