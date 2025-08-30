# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from avhubert.sequence_generator import SequenceGenerator
def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_size(model, model_name="Model"):
    """Print detailed parameter information for a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_name} Parameter Count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming 32-bit floats)")

class MultiTargetSequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=0,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        super().__init__(
            models, tgt_dict, beam_size, max_len_a, max_len_b, max_len, min_len,
            normalize_scores, len_penalty, unk_penalty, temperature, match_source_len,
            no_repeat_ngram_size, search_strategy, eos, symbols_to_strip_from_output,
            lm_model, lm_weight
        )
        
        # Initialize temperature parameters
        # Initialize temperature parameters
        print_model_size(self.model, "MultiTarget Model")
        print_model_size(self.model.models[0].encoder, "MultiTarget Model")
        
        print_model_size(self.model.models[0].conformer, "MultiTarget Model")
        1/0

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:                     
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input. input keys: " + str(net_input.keys()))

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        if src_tokens['audio'] is not None:
            bsz, src_len = src_tokens['audio'].size()[:2]
            src_device = src_tokens['audio'].device
        else:
            bsz, src_len = net_input['padding_mask'].size()
            src_device = src_tokens['video'].device

        ###
        sample['target_lengths'] = src_lengths * 2
        max_len = sample['target_lengths'].max().item()
        sample['target'] = sample['target'][:, :max_len]
        if sample['target'].size(-1) < max_len:
            sample['target'] = F.pad(sample['target'], (0, max_len-sample['target'].size(-1)), "constant", self.pad)
        for i, target_length in enumerate(sample['target_lengths']):
            sample['target'][i][target_length:] = self.pad
            pos_eos = (sample['target'][i]==self.eos).nonzero()
            if pos_eos.nelement() > 0 and pos_eos[0].item() > 0:
                pos_eos = pos_eos[0].item()
                sample['target'][i][pos_eos:] = sample['target'][i][pos_eos-1]
        ###

        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # if hasattr(self.model.single_model, 'conformer'):
        #     encoder_outs = [
        #         model.conformer(encoder_outs[i]['encoder_out'].repeat_interleave(2, dim=0),
        #                         encoder_outs[i]['encoder_padding_mask'].repeat_interleave(2, dim=1),
        #                         net_input['spk_emb'],
        #                         full_source=net_input["source"])
        #         for i, model in enumerate(self.model.models)
        #     ]
        if hasattr(self.model.single_model, 'conformer'):
            for i, model in enumerate(self.model.models):
                output_conformer = model.conformer( \
                    encoder_outs[i]['encoder_out'].repeat_interleave(2, dim=0), \
                    encoder_outs[i]['encoder_padding_mask'].repeat_interleave(2, dim=1), \
                    net_input['spk_emb'],
                    full_source=net_input["source"]
                )
                with torch.amp.autocast('cuda'):
                    speech_prompt = model.prompt_encoder(mel_specs=net_input["source"]["mel_prompt"],
                                                mel_mask=~net_input["source"]["mel_prompt_mask"].bool())
                    
                content_output = output_conformer['content_out']
                content_mask = output_conformer["content_padding_mask"]

                speech_prompt_mask = ~net_input["source"]["mel_prompt_mask"].bool()
                if model.fuser is not None:
                    speech_prompt, _ = model.fuser(content_output.transpose(0, 1), 
                                          speech_prompt.transpose(0, 1), 
                                          speech_prompt.transpose(0, 1),
                                          key_padding_mask=speech_prompt_mask)
                else:
                    speech_prompt = torch.cat([content_output, speech_prompt], dim=1)


                # Auto-regressive pitch and energy prediction
                """
                net_input["tokens"]["prefix_prompts"] = speech_prompt.transpose(0, 1)
                prosody_output = model.prosody_model(**net_input["tokens"])

                
                """
                
                #assert 1==0, f"predict: {f0},\n target: {f0_GT}"
              
                
                speech_prompt = speech_prompt.transpose(0, 1)
                with torch.amp.autocast('cuda'):
                    prosody_output = model.prosody_model(mel_specs=speech_prompt, mask=content_mask)
       
                
               
                energy_pred = prosody_output["energy"]
                f0_pred = prosody_output["f0"]
                # f0_pred = net_input["source"]["f0_target"] 
                # energy_pred = net_input["source"]["energy_target"] 
               # assert 1 == 0, "Check F0 and Energy predictions above."
                with torch.amp.autocast('cuda'):
                    output_mel = model.coarsed_mel_generators(f0_tokens=f0_pred, \
                                                    energy_tokens=energy_pred, \
                                                    filter_input=content_output, \
                                                    energy_padding_mask=content_mask, \
                                                    vid_padding_mask=content_mask
                                                    )


        encoder_out = output_conformer['encoder_out'].transpose(0,1).contiguous()
        encoder_out_mel = output_mel["encoder_out_mel"]

        if encoder_out_mel is not None:
            mels = encoder_out_mel.cpu().numpy()
            mels = [mel[:target_length*2] for mel, target_length in zip(mels, sample['target_lengths'])]
            sample['mels'] = mels

        encoder_out[..., self.tgt_dict.bos()] = -math.inf
        encoder_out[..., self.pad] = -math.inf
        encoder_out[..., self.eos] = -math.inf
        encoder_out[..., self.unk] = -math.inf

        output = encoder_out.argmax(dim=-1).transpose(0,1)
        finalized = [[{"tokens": tokens}] for tokens in output]
        

        return finalized, sample
