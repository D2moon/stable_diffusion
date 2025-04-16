import torch
from torch import nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: False)-> torch.Tensor:
        # x: (BatchSize, SeqLen, Dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        intermin_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (BatchSize, SeqLen, Dim) -> (BatchSize, SeqLen, 3*Dim) -> 3 tensors of (BatchSize, SeqLen, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (BatchSize, SeqLen, Dim) -> (BatchSize, SeqLen, H, Dim/H) -> (BatchSize, H, SeqLen, Dim/H)
        q = q.view(intermin_shape).transpose(1, 2)
        k = k.view(intermin_shape).transpose(1, 2)
        v = v.view(intermin_shape).transpose(1, 2)

        # weight: (BtchSize, H, SeqLen, SeqLen)
        weight = q @ k.transpose(-2, -1)

        if causal_mask:
            # mask where the upper triangle is 1 and lower triangle is 0
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # output: (BatchSize, H, SeqLen, Dim/H)
        output = weight @ v

        # (BatchSize, H, SeqLen, Dim/H) -> (BatchSize, SeqLen, H, Dim/H))
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        # (BatchSize, SeqLen, Dim) -> (BatchSize, SeqLen, Dim)
        output = self.out_proj(output)

        # (BatchSize, SeqLen, Dim)
        return output




