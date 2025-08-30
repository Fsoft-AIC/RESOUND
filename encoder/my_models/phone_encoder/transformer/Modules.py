import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
 

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, x_input):
        return self.affine(x_input)
class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, input_dim, style_dim):
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.style = AffineLinear(style_dim, input_dim * 2)
        self.style.affine.bias.data[:input_dim] = 1
        self.style.affine.bias.data[input_dim:] = 0

    def forward(self, input, style_code):
        # style
        style = self.style(style_code).unsqueeze(1)
        gamma, beta = style.chunk(2, dim=-1)
        
        out = self.norm(input)
        out = gamma * out + beta
        return out
