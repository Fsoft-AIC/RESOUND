import torch
import torch.nn as nn
from .Modules import Mish, Conv1dGLU, MultiHeadAttention, LinearNorm
import numpy as np
import torch.nn.functional as F

class MelStyleEncoder(nn.Module):
    ''' MelStyleEncoder '''
    def __init__(self, config):
        super(MelStyleEncoder, self).__init__()
        self.in_dim = config.n_mel_channels
        self.hidden_dim = config.style_hidden
        self.out_dim = config.style_vector_dim
        self.kernel_size = config.style_kernel_size
        self.n_head = config.style_head
        self.dropout = config.dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            len_ = torch.clamp(len_, min=1e-9) 
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        
        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1,2)
        x = self.temporal(x)
        x = x.transpose(1,2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)
        if w is None:
            print("Warning: MelStyleEncoder output is None")
        return w
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class AffineLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AffineLinear, self).__init__()
        affine = nn.Linear(in_dim, out_dim)
        self.affine = affine

    def forward(self, input):
        return self.affine(input)

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
    
def Config():
    config = {}
    return config
def main():
    config = Config()
    config.n_mel_channels = 80
    config.style_hidden = 512
    config.style_vector_dim = 256
    config.style_kernel_size = 5
    config.style_head = 8
    config.dropout = 0.1
    model = MelStyleEncoder(config)
    
    mel = np.load('mel.npy')
    
    mel = torch.from_numpy(mel).unsqueeze(0).float()
    w = model(mel)
    print(w.shape)
    norm = StyleAdaptiveLayerNorm(512, 256)
    input = torch.randn(1, 100, 512)
    w = norm(input, w)
    print(w.shape)

if __name__ == '__main__':
    main()
