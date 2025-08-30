import sys,logging
import contextlib
from argparse import Namespace
from hydra.core.config_store import ConfigStore
import torch


from .my_models.phone_encoder import *
from .my_models.cross_attention import *
from .my_models.variance_adaptor import *
from .my_models.mel_style_encoder import *
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model
from fairseq.models.transformer_lm import TransformerLanguageModelConfig
import os
import matplotlib.pyplot as plt
from fairseq.dataclass import ChoiceEnum
DEFAULT_MAX_TARGET_POSITIONS = 1024
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])
DBG=True if len(sys.argv) == 1 else False

if DBG:
    pass
else:
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertAsrConfig, AVHubertSeq2SeqConfig, Linear, Embedding
    from pathlib import Path
    sys.path.insert(0, Path(__file__).resolve().parent.parent)
    from espnet.nets.pytorch_backend.transformer.encoder import Encoder, EncoderConformerFusion
    sys.path.pop(0)

logger = logging.getLogger(__name__)
import os
import matplotlib.pyplot as plt
import torch
from datetime import datetime
def save_attention_map(attention_probs, head_index, folder_path="attention_maps_"):
    """
    Saves the attention map for a specific head without requiring a batch index.
    
    Parameters:
    attention_probs: torch.Tensor of shape (batch_size, seq_len, attention_dim)
    head_index: Index of the attention head
    folder_path: Path where attention maps will be saved
    """
    # Create a unique folder for each batch using timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch_folder = os.path.join(folder_path, f'batch_{timestamp}')
    os.makedirs(batch_folder, exist_ok=True)

    # attention_probs shape: (batch_size, seq_len, attention_dim) = [6, 290, 69]
    attention_probs_np = attention_probs.detach().cpu().numpy()

    # Visualize and save each attention map for each sample in the batch
    for sample_idx in range(attention_probs_np.shape[0]):  # Loop over batch_size
        fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure width for x-axis
        cax = ax.matshow(attention_probs_np[sample_idx], cmap='viridis', aspect='auto')  # Set aspect to 'auto'
        plt.colorbar(cax)

        # Set axis labels and titles
        plt.title(f'Attention Map  - Sample {sample_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')

        # Save the plot in the unique folder for the current sample in the batch
        file_path = os.path.join(batch_folder, f'attention_map_head_sample_{sample_idx}.png')
        plt.savefig(file_path)
        plt.close()

@dataclass
class SpeechUnitLanguageModelConfig(TransformerLanguageModelConfig):
    layernorm_embedding: bool = field(
        default=True, metadata={"help": "add layernorm to embedding"}
    )
    max_target_positions: int = field(
            default=1024, metadata={"help": "max target positions"}
        )
    mask_dur_prob: float = field(
        default=0.0, metadata={"help": "probability to mask entire duration sequence"}
    )
    mask_dur_seg_prob: float = field(
        default=0.0,
        metadata={"help": "probability to mask a segment of duration sequence"},
    )
    mask_dur_seg_leng: int = field(
        default=5, metadata={"help": "length of duration segment mask"}
    )
    mask_dur_seg_type: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", 
        metadata={"help": "how to choose duration mask length"}
    )

    mask_f0_prob: float = field(
        default=0.0, metadata={"help": "probability to mask entire f0 sequence"}
    )
    mask_f0_seg_prob: float = field(
        default=0.0, metadata={"help": "probability to mask a segment of f0 sequence"}
    )
    mask_f0_seg_leng: int = field(
        default=5, metadata={"help": "length of f0 segment mask"}
    )
    mask_f0_seg_type: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose f0 mask length"}
    )


    mask_energy_prob: float = field(
        default=0.0, metadata={"help": "probability to mask entire energy sequence"}
    )
    mask_energy_seg_prob: float = field(
        default=0.0, metadata={"help": "probability to mask a segment of energy sequence"}
    )
    mask_energy_seg_leng: int = field(
        default=5, metadata={"help": "length of energy segment mask"}
    )
    mask_energy_seg_type: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose energy mask length"}
    )

    ignore_duration_input: bool = field(
        default=False, metadata={"help": "whether token durations should be zeroed out"}
    )

    ignore_f0_input: bool = field(
        default=False, metadata={"help": "whether F0 should be zeroed out"}
    )
    ignore_energy_input: bool = field(
        default=False, metadata={"help": "whether F0 should be zeroed out"}
    )

    # use duration or not
    use_duration_as_input: bool = field(
        default=False, metadata={"help": "use duration or not, not using duration means frame-level prediction"}
    )
@dataclass
class StyleEncoderConfig:
    n_mel_channels: int = field(default=80)
    style_hidden: int = field(default=128)
    style_vector_dim: int = field(default=128)
    style_kernel_size: int = field(default=5)
    style_head: int = field(default=2)
    dropout: float = field(default=0.1)
    
@dataclass
class LipEncoderConfig:
    max_seq_len: int = field(default=1000)
    encoder_layer: int = field(default=6)
    encoder_head: int = field(default=4)
    encoder_hidden: int = field(default=512)
    decoder_hidden: int = field(default=512)
    conv_filter_size: int = field(default=1024)
    conv_kernel_size: tuple = field(default=(9, 1))
    encoder_dropout: float = field(default=0.2)

@dataclass
class VideoEncoderConfig:
    max_seq_len: int = field(default=2000)
    encoder_layer: int = field(default=3)
    encoder_head: int = field(default=8)
    encoder_hidden: int = field(default=512)
    encoder_dropout: float = field(default=0.1)

@dataclass
class SpeakerCrossConfig:
    max_seq_len: int = field(default=2000)
    encoder_layer: int = field(default=3)
    encoder_head: int = field(default=8)
    encoder_hidden: int = field(default=512)
    encoder_dropout: float = field(default=0.1)
@dataclass
class SpeechPromptConfig:
    # Mel-spectrogram parameters
    mel_channels: int = field(default=80)
    
    # Conv1D parameters
    conv_channels: int = field(default=512)
    conv_kernel: int = field(default=5)
    dropout: float = field(default=0.1)
    hidden_embed_dim: int = field(default=512)
    
    # FFT Block parameters
    nb_block: int = field(default=4)
    max_seq_len: int = field(default=4000)
    
    # Attention parameters
    encoder_dim: int = field(default=512)
    encoder_n_head: int = field(default=8)
    encoder_dropout: float = field(default=0.1)
    attn_dropout: float = field(default=0.1)
    attn_nb_heads: int = field(default=8)
    conv_dropout: float = field(default=0.1)
@dataclass
class GeneratorConfig:
    max_seq_len: int = field(default=4000)  
    n_layers: int = field(default=3)
    dropout: float = field(default=0.1) 
    encoder_hidden: int = field(default=512)
    encoder_head: int = field(default=8)
    conv_filter_size: int = field(default=2048)
    conv_kernel_size: tuple = field(default=(3, 3))
    # f0, energy units
    f0_n_units: int = field(default=68) # 64 + 4 special tokens
    energy_n_units: int = field(default=68)

@dataclass
class NARProsodyConfig:
    # Input/Output dimensions
    input_dim: int = field(default=512)      # Input embedding dimension
    output_dim: int = field(default=1)      # Number of prosody features
    
    # Convolutional layer parameters
    conv_channels: int = field(default=256)
    conv_kernel: int = field(default=3)
    dropout: float = field(default=0.2)
    
    # FFT Block parameters
    nb_block: int = field(default=4)         # Number of FFT blocks
    max_seq_len: int = field(default=4000)   # Maximum sequence length
    hidden_embed_dim: int = field(default=256)  # Hidden embedding dimension

@dataclass
class MultiTargetEncoderModelConfig(AVHubertSeq2SeqConfig):
    use_conformer: bool = field(default=False)
    conformer_layers: int = field(default=8)  #12  - 24/10/2024
    conformer_embed_dim: int = field(default=512)
    conformer_ffn_embed_dim: int = field(default=2048)
    conformer_attention_heads: int = field(default=8)
    conformer_dropout: float = field(default=0.1)
    conformer_attention_dropout: float = field(default=0.1)
    conformer_layer_norm_first: bool = field(default=True)
    
    vid_conformer_layers: int = field(default=6)

    cross_num_attention_heads: int = field(default=8)
    cross_num_attention_heads: int = field(default=8)
    cross_hidden_size: int = field(default=512)
    cross_dropout_rate: float = field(default=0.1)
    #lip_encoder
    lip_encoder: LipEncoderConfig = field(default_factory=LipEncoderConfig)
    #variance adaptor
    
    video_encoder: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    #style encoder
    style_encoder: StyleEncoderConfig = field(default_factory=StyleEncoderConfig)
    #speaker_cross encoder
    speaker_cross: SpeakerCrossConfig = field(default_factory=SpeakerCrossConfig)

    #Prompt Encoder
    prompt_encoder: SpeechPromptConfig = field(default=SpeechPromptConfig())

    #Prosody Predictor
    speech_unit_modeling: SpeechUnitLanguageModelConfig = field(default=SpeechUnitLanguageModelConfig())

    #Generator
    generator: GeneratorConfig = field(default=GeneratorConfig())

    #NAR Prosody Predictor
    nar_prosody_model: NARProsodyConfig = field(default=NARProsodyConfig())



    
        

@register_model("multi_target", dataclass=MultiTargetEncoderModelConfig)
class MultiTargetEncoderModel(FairseqEncoderModel):
    def __init__(self, conformer, tgt_dict, cfg):
        super().__init__(conformer)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates


    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        cfg.decoder_embed_dim = len(tgt_dict)
        conformer = None
        if cfg.use_conformer:
            conformer = Conformer(cfg)

        return cls(conformer, tgt_dict, cfg)


    def forward(self, **kwargs):
        if self.cfg.use_conformer:
            output = self.encoder(**kwargs)

        output['encoder_out'] = output['encoder_out'].transpose(0,1).contiguous()

        return output

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class Conformer(FairseqEncoder):
    def __init__(self, cfg,  tgt_dict=None):
        super().__init__(None)

        self.encoder = Encoder(
            idim=-1,
            attention_dim=cfg.conformer_embed_dim, # adim
            attention_heads=cfg.conformer_attention_heads, # aheads
            linear_units=cfg.conformer_ffn_embed_dim, # eunits
            num_blocks=cfg.conformer_layers, #elayers
            dropout_rate=cfg.conformer_dropout, # dropout_rate
            positional_dropout_rate=cfg.conformer_dropout, # dropout_rate
            attention_dropout_rate=cfg.conformer_attention_dropout, # transformer_attn_dropout_rate
            input_layer="conv3d", # transformer_input_layer
            normalize_before=cfg.conformer_layer_norm_first,
            macaron_style=1, # macaron_style
            encoder_attn_layer_type="rel_mha", # transformer_encoder_attn_layer_type
            use_cnn_module=1, # use_cnn_module
            zero_triu=False, # zero_triu
            cnn_module_kernel=31, # cnn_module_kernel
            relu_type="swish", # relu_type
            a_upsample_ratio=1, # a_upsample_ratio,
            
        )
        self.num_block = cfg.conformer_layers

        self.phon_encoder = PhoneEncoder()
        self.vid_encoder = VideoEncoder(cfg.video_encoder, cfg.video_encoder.encoder_hidden)


        
        self.mlp_dim = cfg.conformer_embed_dim * 4      

        self.vid_mlp = nn.Sequential(
            Linear(cfg.conformer_embed_dim, self.mlp_dim),
            nn.ReLU(), 
            Linear(self.mlp_dim, cfg.conformer_embed_dim),
        )
        self.vid_norm = nn.LayerNorm(cfg.conformer_embed_dim)


        self.phone_mlp = nn.Sequential(
            Linear(cfg.conformer_embed_dim, self.mlp_dim),
            nn.ReLU(), 
            Linear(self.mlp_dim, cfg.conformer_embed_dim),
        )
        self.phoneme_norm = nn.LayerNorm(cfg.conformer_embed_dim)

        
         #Cross attention
        self.cross_attn = CrossAttention(cfg.video_encoder)


        self.final_dropout = nn.Dropout(cfg.final_dropout)
        

        d = cfg.conformer_embed_dim
        if 512 != d:
            self.proj_in = Linear(cfg.w2v_args.model.encoder_embed_dim, d)
        else:
            self.proj_in = None
        

        if tgt_dict is not None:
            # self.proj_out = Linear(d, len(tgt_dict))
            self.proj_out = MLP(
                d, [d, d, len(tgt_dict)], cfg.final_dropout, nn.GELU
            )
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj_out = Linear(d, cfg.decoder_embed_dim)
          
        else:
            self.proj_out = None

        self.mel_conv = nn.Sequential(
            nn.Conv1d(in_channels=d+256,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=d,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=d,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.final_dropout),
            nn.GELU(),
        )
        self.mel_proj = Linear(d, 160)


    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, spk_emb=None, tbc=True, **kwargs):
        """Lip Video embedding input
        """
        x_vid = source["video"]
        x_vid = self.encoder.frontend(x_vid.squeeze(1))
    

        """ Phoneme embedding input
        """
        phone_emb = source["phone_emb"]
        phone_emb_mask = source["phone_mask"].bool()

        
        """ Encode phoneme input
        """
        x_phone = self.phon_encoder(phone_emb, phone_emb_mask)
        x_phone_mask = phone_emb_mask
        x_phone = x_phone + self.phone_mlp(self.phoneme_norm(x_phone))

     


        #after front end [148, 512]
        x_vid = x_vid.repeat_interleave(2, dim=1)
        x_vid_padding_mask = padding_mask.repeat_interleave(2, dim=1)

        """Encode video embedding
        """
        
        x_vid = x_vid + self.vid_mlp(self.vid_norm(x_vid))
        x_vid = self.vid_encoder(x_vid, x_vid_padding_mask)
        x_vid_mask = x_vid_padding_mask

        """ Fusion Module
        """

        ###-------------------###
        """ Concatenatation
        """

        # x = torch.cat((x_vid, x_phone), dim=1)
        # x = self.fuse_proj(x)
        # x = self.fuse_norm(x)
        # padding_mask = torch.cat((x_vid_mask, x_phone_mask), dim=1)

        ###-------------------###

        ###-------------------###
        """  Cross attention
        """

        output, attention_weights = self.cross_attn(xq=x_vid.transpose(0, 1), x=x_phone.transpose(0, 1),
                                        key_padding_mask=x_phone_mask)

        output = output.transpose(0,1)
        output_mask = x_vid_mask


        """ End fusion module
        """

        x = output
        padding_mask = output_mask

        # To device
        device = x.device
        padding_mask = padding_mask.to(device)


        if self.proj_in:
            x = self.proj_in(x)


        # Conformer
        x, masks = self.encoder.forward_after_frontend(
            x,
            masks = ~padding_mask.unsqueeze(-2),
        )

        padding_mask = ~masks.squeeze(-2)


        if spk_emb is not None:
            assert spk_emb.size(-1) == 256
            spk_x = torch.cat([spk_emb.unsqueeze(1).repeat(1,x.size(1),1), x], dim=-1)
        else:
            spk_x = x

        encoder_out_mel = self.mel_proj(self.mel_conv(spk_x.transpose(1,2)).transpose(1,2))

        B, T, D = encoder_out_mel.shape
        encoder_out_mel = encoder_out_mel.reshape(B, T, D//2, 2).transpose(-1,-2).reshape(B, T*2, D//2)

        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
        x = self.final_dropout(x)
        if self.proj_out:
            x = self.proj_out(x)
        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            "encoder_out_mel": encoder_out_mel,
            "attention": attention_weights
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    

import math
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims,
        dropout: float = 0.1,
        nonlinearity = nn.ReLU,
        normalization = None, #nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super(MLP, self).__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        return x

