#from text import *
import base64
import os
import string
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Tuple

import tiktoken




@dataclass
class Tokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens"""

    encoding: tiktoken.Encoding
    language: str = "en"
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        sot_sequence = [sot]
        if self.task is not None:
            sot_sequence.append(transcribe)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        return self.to_language_token(self.language)

    def to_language_token(self, language):
        if language != "en":
            raise ValueError("This tokenizer only supports English")
        return self.special_tokens.get("<|en|>")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        return (self.to_language_token("en"),)

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        return ("en",)

    def split_to_word_tokens(self, tokens: List[int]):
        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_spaces(self, tokens: List[int]):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens

@lru_cache(maxsize=None)
def get_tokenizer(
    *,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> Tokenizer:
    


    encoding_name = "gpt2"
    language = None
    task = None

    encoding = get_encoding(name=encoding_name)

    return Tokenizer(
        encoding=encoding, language=language, task=task
    )

@lru_cache(maxsize=None)
def get_encoding(name: str = "gpt2"):
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )
"""================================================================
"""
from ..text import *
from g2p_en import G2p
import numpy as np
import torch
text_cleaners = ["english_cleaners"]
def clean_text(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text
def process_text(text):
    text_cleaned = clean_text(text)
    g2p = G2p()
    phone = g2p(text_cleaned)
    phone = [p for p in phone if p != " "]
    phone = "{" + "}{".join(phone) + "}"
    phone = re.sub(r"\{[^\w\s]?\}", "{sp}", phone)
    phone = phone.replace("}{", " ")
    text = text_to_sequence(phone, text_cleaners)
    return text

def process_text_BPE(text):
    text_cleaned = clean_text(text)
    tokenizer = get_tokenizer()
    text = tokenizer.encode(text_cleaned)
    return text
def process_text_texgrid(phones):
    phonemes = '{'+  ' '.join(phones) + '}'
    phonemes = text_to_sequence(phonemes , text_cleaners)
    return phonemes
import numpy as np

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        # Convert to NumPy array if it's not already
        x = np.array(x) if not isinstance(x, np.ndarray) else x
        
        # Pad the array
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    def create_mask(x, length):
        # Convert to NumPy array if it's not already
        x = np.array(x) if not isinstance(x, np.ndarray) else x
        
        mask = np.ones(length)
        mask[x.shape[0]:] = 0
        return mask

    max_len = max([len(x) for x in inputs])
    
    # Pad all inputs
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    
    # Create masks
    masks = np.stack([create_mask(x, max_len) for x in inputs])

    return padded, masks


    return padded, mask
def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask