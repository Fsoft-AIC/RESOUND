# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, glob

import sys
from typing import List
import yaml
from fairseq import search
from dataclasses import dataclass, field
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from omegaconf import MISSING, DictConfig
import torch
from typing import List, Optional, Tuple, Literal
DBG=True if len(sys.argv) == 1 else False
from fairseq.models.transformer_lm import TransformerLanguageModelConfig
if DBG:
    from dataset import MultiTargetDataset
    from ..avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask, LabelEncoder
else:
    from .dataset import MultiTargetDataset
    from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask, LabelEncoder

logger = logging.getLogger(__name__)

class LabelEncoderUnit(LabelEncoder):
    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=True, add_if_not_exist=False,
        ).long()

    def decode(self, tok, symbols_ignore=None):
        return self.dictionary.string(tok, extra_symbols_to_ignore=symbols_ignore)




class UnitDictionary(Dictionary):
    """
    A fixed-sized Dictionary that operates on integer-valued tokens
    wth a trivial (identity) token <-> id mapping.
    Special symbols (bos, eos, ...) have ids above n_units.
    """

    def __init__(
        self,
        *,  # begin keyword-only arguments
        n_units,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
        clip=False,
    ):
        self.n_units = n_units
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.clip = clip

        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        for i in range(n_units):
            self.add_symbol(str(i + 1))
      
        self.pad_index = self.add_symbol(pad) 
        # self.eos_index = self.add_symbol(eos) 
        # self.unk_index = self.add_symbol(unk) 

        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def encode_line(self, line, append_eos=False, prepend_bos=False) -> torch.IntTensor:
        words = [int(float(x)) for x in line.split()]
        if self.clip:
            words = [max(1, min(64, word)) for word in words]
        if prepend_bos:
            words = [self.bos_index] + words
        if append_eos:
            words.append(self.eos_index)
        ids = torch.IntTensor(words)
        return ids
    def print_tokens(self):
        print("Index | Symbol")
        print("-" * 20)
        for i, symbol in enumerate(self.symbols):
            print(f"{i:5d} | {symbol}")

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx 
            self.symbols.append(word)
            self.count.append(n)
            return idx 

@dataclass
class Lip2SpeechConfig(AVHubertPretrainingConfig):
    time_mask: bool = field(default=False)
    random_erase: bool = field(default=False)
    is_training: bool = field(default=True)

    # config for the dur/f0/energy dataset 
    config_path: str = field(default="config.yaml", metadata={"help": "Path to data config.json"})
    
    #It will contain

    # SpeechUnitModelingConfig
    
    max_token_duration: int = field(
        default=200, metadata={"help": "all token durations are capped to this value"}
    )
    tokens_per_sample: int = field(
        default=1024, metadata={"help": "tokens in a sample"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max target positions"}
    )
# use duration or not
    use_duration_as_input: bool = field(
        default=False, metadata={"help": "use duration or not, not using duration means frame-level prediction else phoneme-level"}
    )
    


@register_task("lip2speech", dataclass=Lip2SpeechConfig)
class Lip2SpeechTask(AVHubertPretrainingTask):
    def __init__(self, cfg: Lip2SpeechConfig) -> None:

        """AvhubertPretrainingTask
        """
        super().__init__(cfg)



        """SpeechUnitLanguageModelingTask
        """

        with open(cfg.config_path, "r") as file:
            self.data_config = yaml.load(file, Loader=yaml.SafeLoader)


        self._source_duration_dictionary = self._target_duration_dictionary = (
            UnitDictionary(n_units=self.cfg.max_token_duration + 1, clip=True)
        )
        self._source_f0_dictionary = self._target_f0_dictionary = (
            UnitDictionary(n_units=self.data_config["f0_vq_n_units"])
        )
        self._source_energy_dictionary = self._target_energy_dictionary = (
            UnitDictionary(n_units=self.data_config["energy_vq_n_units"])
        )

        if self.cfg.use_duration_as_input:
            self._channel_names = ["duration", "f0", "energy"]
            self._channel_sizes = [
                len(self._target_duration_dictionary),
                len(self._target_f0_dictionary),
                len(self._target_energy_dictionary),
            ] 
        else:
            self._channel_names = ["f0", "energy"]
            self._channel_sizes = [
            len(self._target_f0_dictionary),
            len(self._target_energy_dictionary),
            ]
    """Properties for the duration, f0, energy dictionaries
    """
    @property
    def source_duration_dictionary(self) -> Optional[Dictionary]:
        return self._source_duration_dictionary

    @property
    def source_f0_dictionary(self) -> Optional[Dictionary]:
        return self._source_f0_dictionary
    
    @property
    def source_energy_dictionary(self) -> Optional[Dictionary]:
        return self._source_energy_dictionary

    @property
    def channel_names(self) -> List[str]:
        return self._channel_names

    @property
    def channel_sizes(self) -> List[int]:
        return self._channel_sizes


    @property
    def target_duration_dictionary(self) -> Optional[Dictionary]:
        return self._target_duration_dictionary

    @property
    def target_f0_dictionary(self) -> Optional[Dictionary]:
        return self._target_f0_dictionary

    @property
    def target_energy_dictionary(self) -> Optional[Dictionary]:
        return self._target_energy_dictionary


    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        """
        target_dictionary = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]

        labels = ["unt"]
        """
        dictionaries = [self.target_dictionary] if self.fine_tuning else self.dictionaries
        pad_list = [dictionary.pad() for dictionary in dictionaries]
        eos_list = [dictionary.eos() for dictionary in dictionaries]

        if not self.cfg.is_s2s:
            procs = [LabelEncoderUnit(dictionary) for dictionary in dictionaries]
        else:
            logger.info(f"Using tokenizer")
            bpe_tokenizer = self.s2s_tokenizer
            procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]
        image_aug = self.cfg.image_aug if split == 'train' else False
        noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr)
        noise_num = self.cfg.noise_num # 


        self.is_training = split == 'train' or split == 'valid'
        self.datasets[split] = MultiTargetDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=False,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            noise_fn=noise_fn,
            noise_prob=self.cfg.noise_prob,
            noise_snr=noise_snr,
            noise_num=noise_num,
            time_mask=self.cfg.time_mask,
            random_erase=self.cfg.random_erase,
            is_training=self.is_training,
            dur_dictionary=self.source_duration_dictionary,
            f0_dictionary=self.source_f0_dictionary,
            energy_dictionary=self.source_energy_dictionary,
        )

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        if seq_gen_cls is None:
            from .sequence_generator import MultiTargetSequenceGenerator
            seq_gen_cls = MultiTargetSequenceGenerator
        return super().build_generator(models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None)