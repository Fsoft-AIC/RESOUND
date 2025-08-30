import torch 
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Dict, Iterable, Optional
import torch.nn.functional as F
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)
    
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, None if self.bias is None else self.bias)
class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(in_dim, hid_dim),
            nn.GELU(), 
            Linear(hid_dim, out_dim),
        )
        self.mlp_norm = LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return self.dropout(self.mlp_norm(self.mlp(x)))


class ResidualAttentionBlocks(nn.Module):
    def __init__(self, embed_dim: int,
                  num_heads: int, 
                  dropout: float = 0.1,
                  cross_attention: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, dropout = dropout)
        self.attn_norm = LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_norm = LayerNorm(embed_dim)

        self.mlp_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            Linear(embed_dim, self.mlp_dim),
            nn.GELU(), 
            Linear(self.mlp_dim, embed_dim),
        )
        self.mlp_norm = LayerNorm(embed_dim)

        self.cross_attention = cross_attention

    def forward(self, x: Tensor, x_q: Tensor = None, key_padding_mask=None) -> Tensor:
        x = self.attn_norm(x)
        x = x + self.attn(x, x, x, key_padding_mask=key_padding_mask)[0]

        if self.cross_attention:
            #key_padding_mask = key_padding_mask.transpose(1, 0)
            x_q = self.cross_attn_norm(x_q)
            x_q_r, attn_weight = self.cross_attn(query=x_q, key=x, value=x, key_padding_mask=key_padding_mask)
            x_q = x_q + x_q_r
            x_q = x_q + self.mlp(self.mlp_norm(x_q))
            return x_q, attn_weight

        x = x + self.mlp(self.mlp_norm(x))
        return x
    

class GatedResidualAttentionBlocks(nn.Module):
    def __init__(self, embed_dim: int,
                  num_heads: int, 
                  dropout: float = 0.1,
                  cross_attention: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads, dropout = dropout)
        self.attn_norm = LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_norm = LayerNorm(embed_dim)

        self.mlp_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            Linear(embed_dim, self.mlp_dim),
            nn.GELU(), 
            Linear(self.mlp_dim, embed_dim),
        )
        self.mlp_norm = LayerNorm(embed_dim)

        self.cross_attention = cross_attention

        # Add Gated
        self.gated_x_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.gated_x_attn_ln = LayerNorm(embed_dim)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        
        self.ff_ln = LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            Linear(embed_dim, self.mlp_dim),
            nn.GELU(), 
            Linear(self.mlp_dim, embed_dim),
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.]))  
    def apply_gated_x_attn(self, x_p, x_v, vid_padding_mask=None, phone_padding_mask=None):
        x_r, attn_weight = self.cross_attn(query=self.gated_x_attn_ln(x_p), key=x_v, value=x_v, key_padding_mask=vid_padding_mask)
        x_p = x_p + x_r * self.attn_gate.tanh()
        x_p = x_p + self.ff(self.ff_ln(x_p)) * self.ff_gate.tanh()
        return x_p
    def forward(self, x: Tensor, x_q: Tensor = None, key_padding_mask=None, query_padding_mask=None) -> Tensor:
        x = self.apply_gated_x_attn(x, x_q, vid_padding_mask=query_padding_mask, phone_padding_mask=key_padding_mask)

        x_norm = self.attn_norm(x)
        x = x + self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)[0]

        if self.cross_attention:
            #key_padding_mask = key_padding_mask.transpose(1, 0)
            x_q_norm = self.cross_attn_norm(x_q)
            x_q_r, attn_weight = self.cross_attn(query=x_q_norm, key=x, value=x, key_padding_mask=key_padding_mask)
            x_q = x_q + x_q_r
            x_q = x_q + self.mlp(self.mlp_norm(x_q))
            return x_q, attn_weight

        x = x + self.mlp(self.mlp_norm(x))
        return x
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class VideoEncoder(nn.Module):
    def __init__(self, cfg, pre_encoder_embed_dim):
        super().__init__()
        self.blocks: Iterable[ResidualAttentionBlocks] = nn.ModuleList(
            [
                ResidualAttentionBlocks(embed_dim=cfg.encoder_hidden, num_heads=cfg.encoder_head, dropout=cfg.encoder_dropout) 
                for _ in range(cfg.encoder_layer)
            ]
        )
        self.register_buffer("positional_embedding", sinusoids(cfg.max_seq_len, cfg.encoder_hidden))

        self.post_ln = LayerNorm(cfg.encoder_hidden)
        self.video_projection = Linear(pre_encoder_embed_dim, cfg.encoder_hidden)
        self.video_projection_scalar = nn.Parameter(torch.tensor(1.))
    
    def forward(self, x, padding_mask=None):
        #x = self.video_projection(x)
        x = self.video_projection_scalar * x
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)
        
        for layer, block in enumerate(self.blocks):
            x = block(x, padding_mask)

        x = self.post_ln(x)

        return x



class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks: Iterable[ResidualAttentionBlocks] = nn.ModuleList(
            [
                ResidualAttentionBlocks(embed_dim=cfg.encoder_hidden, num_heads=cfg.encoder_head, dropout=cfg.encoder_dropout) 
                for _ in range(cfg.encoder_layer)
            ]
        )
        self.register_buffer("positional_embedding", sinusoids(cfg.max_seq_len, cfg.encoder_hidden))

        self.post_ln = LayerNorm(cfg.encoder_hidden)

    def forward(self, x, padding_mask=None):

        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)
        
        for layer, block in enumerate(self.blocks):
            x = block(x, padding_mask)

        x = self.post_ln(x)

        return x


class TextEncoder(nn.Module):
    def __init__(
        self, cfg
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg.max_seq_len, cfg.encoder_hidden)
        self.positional_embedding = nn.Parameter(torch.empty(cfg.max_seq_len, cfg.encoder_hidden))

        self.blocks: Iterable[ResidualAttentionBlocks] = nn.ModuleList(
            [
                ResidualAttentionBlocks(embed_dim=cfg.encoder_hidden, num_heads=cfg.encoder_head, dropout=cfg.encoder_dropout) 
                for _ in range(cfg.encoder_layer)
            ]
        )
        self.ln = LayerNorm(cfg.encoder_hidden)



    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        x = (
        self.token_embedding(x)
        + self.positional_embedding[: x.shape[1]]  
    )
        x = x.to(xa.dtype)

        for layer, block in enumerate(self.blocks):
            x = block(x)

        x = self.ln(x)
        

        return x

class CrossAttention(nn.Module):
    def __init__(
        self, cfg 
    ):
        super().__init__()


        self.blocks: Iterable[ResidualAttentionBlocks] = nn.ModuleList(
            [
                ResidualAttentionBlocks(embed_dim=cfg.encoder_hidden, num_heads=cfg.encoder_head, dropout=cfg.encoder_dropout, cross_attention=True)
                for _ in range(cfg.encoder_layer)
            ]
        )
        self.ln = LayerNorm(cfg.encoder_hidden)

        self.dropout = torch.nn.Dropout(cfg.encoder_dropout)      


    def forward(self, x: Tensor, xq: Tensor,
                key_padding_mask):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        
        x = x.to(xq.dtype)

        for layer, block in enumerate(self.blocks):
            xq, attn_weight = block(x, xq, key_padding_mask=key_padding_mask)
            
        xq = self.ln(xq)


        return xq, attn_weight
class GatedCrossAttention(nn.Module):
    def __init__(
        self, cfg 
    ):
        super().__init__()


        self.blocks: Iterable[GatedResidualAttentionBlocks] = nn.ModuleList(
            [
                GatedResidualAttentionBlocks(embed_dim=cfg.encoder_hidden, num_heads=cfg.encoder_head, dropout=cfg.encoder_dropout, cross_attention=True)
                for _ in range(cfg.encoder_layer)
            ]
        )
        self.ln = LayerNorm(cfg.encoder_hidden)

        self.dropout = torch.nn.Dropout(cfg.encoder_dropout)      


    def forward(self, x: Tensor, xq: Tensor,
                key_padding_mask, query_padding_mask):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        
        x = x.to(xq.dtype)

        for layer, block in enumerate(self.blocks):
            xq, attn_weight = block(x, xq, key_padding_mask=key_padding_mask, query_padding_mask=query_padding_mask)
            
        xq = self.ln(xq)


        return xq, attn_weight
    

class MaskFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cross_attention = CrossAttention(cfg)

        self.mask = nn.Sequential(
            nn.Conv1d(cfg.encoder_hidden, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1)   
        )
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Linear(cfg.encoder_hidden, 512)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, text_feature, video_feature, video_padding_mask):
        attn_output, attn_output_weights = self.cross_attention(xq=text_feature.transpose(0, 1),
                                                                 x=video_feature.transpose(0, 1),
                                                                 key_padding_mask=video_padding_mask)
        
        mask = self.mask(attn_output.permute(0, 2, 1).contiguous()).permute(2, 0, 1).contiguous()
        
        mask = self.sigmoid(mask)

        enhanced_text = text_feature + mask * text_feature

        enhanced_feat = torch.cat([video_feature, enhanced_text], dim=1)
        enhanced_feat = self.fusion(self.dropout(enhanced_feat))
        return enhanced_feat
def __main__():
    x = torch.randn(5, 10, 512)
    x_q = torch.randn(5, 10, 512)
    key_padding_mask = torch.randn(5, 10)
    model = ResidualAttentionBlocks(embed_dim=512, num_heads=8, dropout=0.1, cross_attention=True)
    print(model(x, x_q, key_padding_mask))

if __name__ == "__main__":
    __main__()  
