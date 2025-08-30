# Mel Generator
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List
import numpy as np
import sys,logging
DBG=True if len(sys.argv) == 1 else False
from .Modules import *
if DBG:
    pass
else:
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertAsrConfig, AVHubertSeq2SeqConfig, Linear, Embedding
@dataclass
class GeneratorConfig:
    max_seq_len: int = field(default=512)  
    n_layers: int = field(default=3)
    dropout: float = field(default=0.1) 
    encoder_hidden: int = field(default=512)
    encoder_head: int = field(default=8)
    conv_filter_size: int = field(default=2048)
    conv_kernel_size: List[int] = field(default_factory=lambda: [3, 3])  
    # f0, energy units
    f0_n_units: int = field(default=64)
    energy_n_units: int = field(default=64)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class FilterGenerator(nn.Module):
    def __init__(self, config):
        super(FilterGenerator, self).__init__()

        n_position = config.max_seq_len + 1
        d_word_vec = config.encoder_hidden
        n_layers = config.n_layers
        n_head = config.encoder_head
        d_k = d_v = (
            config.encoder_hidden
            // config.encoder_head
        )
        d_model = config.encoder_hidden
        d_inner = config.conv_filter_size
        kernel_size = config.conv_kernel_size
        dropout = config.dropout

        self.max_seq_len = config.max_seq_len
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
    def forward(self, hidden, mask, hidden_query=None):

    
        batch_size, max_len = hidden.shape[0], hidden.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and hidden.shape[1] > self.max_seq_len:
            
            output = hidden + get_sinusoid_encoding_table(
                hidden.shape[1], self.d_model
            )[: hidden.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                hidden.device
            )
        else:
            output = hidden + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for i, enc_layer in enumerate(self.layer_stack):
            output, enc_slf_attn = enc_layer(
                output, mask=mask, slf_attn_mask=slf_attn_mask, hidden_query=hidden_query if i==0 else None
            )

        return output

class SourceGenerator(FilterGenerator):
    def __init__(self, generator_cfg: GeneratorConfig):
        super().__init__(generator_cfg)
        self.layer_norm = nn.LayerNorm(1)
        #self.f0_embeding = nn.Embedding(generator_cfg.f0_n_units, generator_cfg.encoder_hidden)
        #self.energy_embeding = nn.Embedding(generator_cfg.energy_n_units, generator_cfg.encoder_hidden)
    def forward(self, f0_emb, energy_emb, mask, hidden_query=None):
        #f0_emb = self.f0_embeding(f0_tokens)
        #energy_emb = self.energy_embeding(energy_tokens)
        assert f0_emb.shape == energy_emb.shape
        prosody_emb = (f0_emb + energy_emb) * 0.5
        prosody_emb = self.layer_norm(prosody_emb)
        return super().forward(prosody_emb, mask, hidden_query)

class CoarsedMelGenerators(nn.Module):
    def __init__(self, generator_cfg: GeneratorConfig, cfg: AVHubertAsrConfig):
        super().__init__()

        self.cfg = cfg

        self.filter_generator = FilterGenerator(generator_cfg)
        self.source_generator = SourceGenerator(generator_cfg)
        self.mel_conv = nn.Sequential(
            nn.Conv1d(in_channels=cfg.conformer_embed_dim,out_channels=cfg.conformer_embed_dim,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=cfg.conformer_embed_dim,out_channels=cfg.conformer_embed_dim,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=cfg.conformer_embed_dim,out_channels=cfg.conformer_embed_dim,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.dropout),
            nn.GELU(),
        )
        #cfg.decoder_embed_dim = len(tgt_dict)
        if getattr(cfg, "decoder_embed_dim", generator_cfg.encoder_hidden) != generator_cfg.encoder_hidden:
            self.proj_out = Linear(generator_cfg.encoder_hidden, cfg.decoder_embed_dim)
        else:
            self.proj_out = None

        # mel generation
        self.mel_proj = Linear(generator_cfg.encoder_hidden, 160)
    def forward(self, f0_tokens, energy_tokens, filter_input, energy_padding_mask, vid_padding_mask, tbc=True, spk_emb=None, hidden_query=None):
       
        min_len = min(f0_tokens.size(1), filter_input.size(1))

        source_input = self.source_generator(
            f0_tokens[:, :min_len], 
            energy_tokens[:, :min_len], 
            mask=energy_padding_mask[:, :min_len], 
            hidden_query=hidden_query[:, :min_len] if hidden_query is not None else None
        )
        filter_input = self.filter_generator(
            filter_input[:, :min_len], 
            mask=vid_padding_mask[:, :min_len], 
            hidden_query=hidden_query[:, :min_len] if hidden_query is not None else None
        )

        assert source_input.shape == filter_input.shape, \
        f"Shape mismatch: source_input {source_input.shape} != filter_input {filter_input.shape}"
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        torch.set_printoptions(threshold=float('inf'),  precision=4)
        #assert 1==0, f"{source_input} \n {filter_input}"
        coarse_mel = source_input
        #+ filter_input
     
      
       
        # x just for calculate CE loss with Hubert
        if self.proj_out:
            x = self.proj_out(coarse_mel) 
            # B x T x 204


        # for mel generation
        if spk_emb is not None:
            assert spk_emb.size(-1) == 256
            spk_x = torch.cat([spk_emb.unsqueeze(1).repeat(1,coarse_mel.size(1),1), coarse_mel], dim=-1)
        else:
            spk_x = coarse_mel

        encoder_out_mel = self.mel_proj(self.mel_conv(spk_x.transpose(1,2)).transpose(1,2))

        B, T, D = encoder_out_mel.shape
        encoder_out_mel = encoder_out_mel.reshape(B, T, D//2, 2).transpose(-1,-2).reshape(B, T*2, D//2)


        
        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": vid_padding_mask,  # B x T
            "padding_mask": vid_padding_mask,
            "encoder_out_mel": encoder_out_mel,
        }
