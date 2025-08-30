"""
Cross attention
Video Encoder = Transformer Encoder + MLP
"""
import sys,logging
import contextlib
from argparse import Namespace

from .my_models.phone_encoder import *
from .my_models.cross_attention import *
from .my_models.mel_style_encoder import *

from .multiverse.nar_prosody import NARProsodyPredictor
from .multiverse.generator import CoarsedMelGenerators
from .multiverse.speech_prompt import SpeechPromptEncoder
import torch
import math
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model
import os
import matplotlib.pyplot as plt
import torch
from datetime import datetime
DBG=True if len(sys.argv) == 1 else False

if DBG:
    pass
else:
    from avhubert.hubert_asr import HubertEncoderWrapper, AVHubertAsrConfig, AVHubertSeq2SeqConfig, Linear, Embedding
    from pathlib import Path
    sys.path.insert(0, Path(__file__).resolve().parent.parent)
    from espnet.nets.pytorch_backend.transformer.encoder import Encoder
    sys.path.pop(0)
    from .model import MultiTargetEncoderModelConfig, MLP


logger = logging.getLogger(__name__)
def save_attention_map(attention_probs, head_index, folder_path="attention_maps_avhubert"):
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

@register_model("multi_target_avhubert", dataclass=MultiTargetEncoderModelConfig)
class MultiTargetAVHubertEncoderModel(FairseqEncoderModel):
    def __init__(self, encoder, tgt_dict, cfg, conformer, prompt_encoder, prosody_model, coarsed_mel_generators, fuser="cross"):
        super().__init__(encoder)
        self.conformer = conformer.float()
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

        self.prompt_encoder = prompt_encoder.float()
        self.fuser = nn.MultiheadAttention(cfg.conformer_embed_dim, cfg.conformer_attention_heads) \
            if fuser == "cross" else None
        self.prosody_model = prosody_model.float()
        self.coarsed_mel_generators = coarsed_mel_generators.float()
        
        # Linear Prosody
        self.pitch_linear = nn.Linear(cfg.nar_prosody_model.output_dim, cfg.generator.encoder_hidden)
        self.energy_linear = nn.Linear(cfg.nar_prosody_model.output_dim, cfg.generator.encoder_hidden)


    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )
        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        cfg.decoder_embed_dim = len(tgt_dict)

        conformer = None
        if cfg.use_conformer:
            conformer = Conformer(cfg)

        prompt_encoder = SpeechPromptEncoder(cfg.prompt_encoder)
        prosody_model = NARProsodyPredictor(cfg.nar_prosody_model)
        coarsed_mel_generators = CoarsedMelGenerators(cfg.generator, cfg)
        return cls(encoder, tgt_dict, cfg, conformer, prompt_encoder, prosody_model, coarsed_mel_generators)


    def forward(self, **kwargs):
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)
        
        if self.cfg.use_conformer:
            output_conformer = self.conformer(
                source=output['encoder_out'].repeat_interleave(2, dim=0),
                padding_mask=output['encoder_padding_mask'].repeat_interleave(2, dim=1),
                spk_emb=kwargs['spk_emb'],
                full_source=kwargs['source']
            )

        """
        speech_prompt: B x S x 512
        mel_specs:     B x S x 80
        mel_mask:      B x S x 80
        """

        # Move prompt_encoder to the same device as the inputs
        #self.prompt_encoder = self.prompt_encoder.to(device)
        with torch.amp.autocast('cuda'):
            speech_prompt = self.prompt_encoder(mel_specs=kwargs["source"]["mel_prompt"],
                                            mel_mask=~kwargs["source"]["mel_prompt_mask"].bool())
        
        content_output = output_conformer['content_out']
        content_mask = output_conformer["content_padding_mask"] 

        
        
        speech_prompt_mask = ~kwargs["source"]["mel_prompt_mask"].bool()
        if self.fuser is not None:
            speech_prompt, _ = self.fuser(content_output.transpose(0, 1), # Q
                                          speech_prompt.transpose(0, 1),  # K
                                          speech_prompt.transpose(0, 1),  # V
                                          key_padding_mask=speech_prompt_mask)
        else:
            speech_prompt = torch.cat([content_output, speech_prompt], dim=1)
        # Speech Prompt is a condition for prosody model
        ### T, B, V
        speech_prompt = speech_prompt.transpose(0, 1)
        with torch.amp.autocast('cuda'):
            prosody_output = self.prosody_model(mel_specs=speech_prompt, mask=content_mask)
           
            #assert 1==0, prosody_output["f0"].shape
            f0_pred = kwargs["source"]["f0_target"] 
            energy_pred = kwargs["source"]["energy_target"] 

            

            output_debug = self.coarsed_mel_generators(f0_tokens=f0_pred, \
                                                    energy_tokens=energy_pred, \
                                                    filter_input=content_output, \
                                                    energy_padding_mask=content_mask, \
                                                    vid_padding_mask=content_mask
                                                    )
        
        return {
            "encoder_out": output_conformer["encoder_out"], #encoder_out from conformer
            "encoder_out_mel": output_debug["encoder_out_mel"], #encoder_out_mel from mel_generator
            "prosody_out": prosody_output, #p/e    from prosdy model

        }
    def get_normalized_probs(self, net_output, log_probs, type_loss="hubert", sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if type_loss in ["f0", "energy", "duration"]:
            # Map type_loss to the corresponding key in prosody_out
            #key = "duration" if type_loss == "duration" else type_loss
            output = net_output["prosody_out"][type_loss]
            
            if not torch.is_tensor(output):
                raise NotImplementedError
                
            logits = output.float()
            return F.log_softmax(logits, dim=-1) if log_probs else F.softmax(logits, dim=-1)
        
        return super().get_normalized_probs(net_output, log_probs, sample)
    def get_targets(self, sample, net_output, type_loss="hubert"):
        """Get targets for loss calculation."""
        if type_loss == "f0":
            return sample["net_input"]["tokens_target"]["f0_target"]
        elif type_loss == "energy":
            return sample["net_input"]["tokens_target"]["energy_target"]
        elif type_loss == "duration":
            return sample["net_input"]["tokens_target"]["duration_target"]
        else:
            return super().get_targets(sample, net_output)

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

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

class Conformer(FairseqEncoder):
    def __init__(self, cfg: AVHubertAsrConfig, tgt_dict=None):
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

        """ Encoders
        """
        self.phon_encoder = PhoneEncoder()
        self.vid_encoder = VideoEncoder(cfg.video_encoder, cfg.w2v_args.model.encoder_embed_dim)




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


        self.encoder.frontend = None

        d = cfg.conformer_embed_dim

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.num_updates = 0

        if getattr(cfg.w2v_args.model, "encoder_embed_dim", d) != d:
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


        self.content_out = nn.Sequential(
            Linear(cfg.conformer_embed_dim +256 , self.mlp_dim),
            nn.GELU(), 
            Linear(self.mlp_dim, cfg.conformer_embed_dim,),
        )
        self.content_out_norm = LayerNorm(cfg.conformer_embed_dim + 256)

        

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, spk_emb=None, full_source=None, tbc=True, **kwargs):
        """ Phoneme embedding input
        """
        phone_emb = full_source["phone_emb"]
        phone_emb_mask = full_source["phone_mask"].bool()


        """ Encode phoneme input
        """
        x_phone = self.phon_encoder(phone_emb, phone_emb_mask)
        x_phone_mask = phone_emb_mask
        x_phone = x_phone + self.phone_mlp(self.phoneme_norm(x_phone))


 
        """Lip Video embedding input
        """
        x = source
        vid_padding_mask = padding_mask
 

        if tbc:
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

        if self.proj_in:
            x = self.proj_in(x)

        """Encode video embedding
        """
        
        x = x + self.vid_mlp(self.vid_norm(x))
        x_vid = self.vid_encoder(x, vid_padding_mask)
        x_vid_mask = vid_padding_mask

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
        
        #save_attention_map(attention_weights, head_index=0)
        ###-------------------###
        """  Gated Cross attention
        """
        # output, attention_weights = self.cross_attn(xq=x_vid.transpose(0, 1), x=x_phone.transpose(0, 1),
        #                         key_padding_mask=x_phone_mask, query_padding_mask=x_vid_mask)
        
        output = output.transpose(0,1)
        output_mask = x_vid_mask
       
        ###-------------------###


        x = output
        padding_mask = output_mask

        # To device
        device = x.device
        padding_mask = padding_mask.to(device)

        # Conformer
        x, masks = self.encoder.forward_after_frontend(
            x,
            masks = ~padding_mask.unsqueeze(-2),
        )

        padding_mask = ~masks.squeeze(-2)
        codes = self.final_dropout(x)

        if self.proj_out:
            codes = self.proj_out(codes)

        # For content output
        B, T, D = x.shape
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)  #Check need this ?
        content_out = x.reshape(B, T, D//2, 2).transpose(-1,-2).reshape(B, T*2, D//2) # B x T_mel x D // 2
        
        if spk_emb is not None:
#            assert spk_emb.size(-1) == 256
            spk_x = torch.cat([spk_emb.unsqueeze(1).repeat(1,content_out.size(1),1), content_out], dim=-1)
        else:
            spk_x = content_out

        content_out = self.content_out(self.content_out_norm(spk_x)) # B x T_mel x D 
        content_mask = padding_mask.repeat_interleave(2, dim=1)


        
        
   
        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        content_out = self.final_dropout(content_out)

        # if self.proj_out:
        #     x = self.proj_out(x)
        return {
            "content_out": content_out,
            "content_padding_mask": content_mask,
            "encoder_out": codes
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