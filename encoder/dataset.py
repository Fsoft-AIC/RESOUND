import itertools
import logging
import os
import sys
import time
import tgt
from typing import Any, List, Optional, Union
from .my_models.phone_encoder import *
import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from scipy.io import wavfile
 
DBG=True if len(sys.argv) == 1 else False
 
if DBG:
    import utils_aug as custom_utils
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
        stream=sys.stdout,
    )
    from ..avhubert.hubert_dataset import load_audio_visual, load_label, load_label_offset, verify_label_lengths, AVHubertDataset
else:
    from . import utils_aug as custom_utils
    from avhubert.hubert_dataset import load_audio_visual, load_label, load_label_offset, verify_label_lengths, AVHubertDataset
 
logger = logging.getLogger(__name__)
 
def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

 
class Paddings(object):
    def __init__(self, dur_val=0, energy_val=-2.0, f0_val=-2.0):
        self.dur = dur_val
        self.energy = energy_val
        self.f0 = f0_val


class Shifts(object):
    def __init__(self, shifts_str, pads):
        self._shifts = list(map(int, shifts_str.split(",")))
        assert len(self._shifts) == 2, self._shifts
        assert all(s >= 0 for s in self._shifts)
        self.extra_length = max(s for s in self._shifts)
        self.pads = pads

    @property
    def dur(self):
        return self._shifts[0]

    @property
    def f0(self):
        return self._shifts[1]

    @staticmethod
    def shift_one(seq, left_pad_num, right_pad_num, pad):
        assert seq.ndim == 1
        bos = seq.new_full((left_pad_num,), pad)
        eos = seq.new_full((right_pad_num,), pad)
        seq = torch.cat([bos, seq, eos])
        mask = torch.ones_like(seq).bool()
        mask[left_pad_num : len(seq) - right_pad_num] = 0
        return seq, mask

    def __call__(self, dur, f0, energy):
        if self.extra_length == 0:
            dur_mask = torch.zeros_like(dur).bool()
            f0_mask = torch.zeros_like(f0).bool()
            energy_mask = torch.zeros_like(energy).bool()
            return dur, dur_mask, f0, f0_mask, energy, energy_mask

        return dur, dur_mask, f0, f0_mask, energy, energy_mask
class MultiTargetDataset(AVHubertDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            label_paths: List[str],
            label_rates: Union[List[float], float],  # -1 for sequence labels
            pad_list: List[str],
            eos_list: List[str],
            label_processors: Optional[List[Any]] = None,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            store_labels: bool = True,
            random_crop: bool = False,
            single_target: bool = False,
            stack_order_audio: int=1,
            skip_verify: bool=False,
            image_mean: float=0,
            image_std: float=1,
            image_crop_size: int=88,
            image_aug: bool=False,
            modalities: Optional[List[str]]=None,
            is_s2s=False,
            noise_fn=None,
            noise_prob=0,
            noise_snr=0,
            noise_num=1,
            time_mask: bool = False,
            random_erase: bool = False,
            is_training: bool = True,
            dur_dictionary: Optional[Any] = None,
            f0_dictionary: Optional[Any] = None,
            energy_dictionary: Optional[Any] = None,
            shifts="0,0",
            use_duration: bool = False,
    ):
        # self.label_rates = (
        #     [label_rates for _ in range(len(label_paths))]
        #     if isinstance(label_rates, int)
        #     else label_rates
        # )
        self.label_rates = [-1 for _ in range(len(label_paths))]
        self.modalities = set(modalities)
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(manifest_path, max_keep_sample_size, min_keep_sample_size, frame_rate=sample_rate, label_paths=label_paths, label_rates=self.label_rates)
        self.sample_rate = sample_rate
        self.stack_order_audio = stack_order_audio
        self.shuffle = shuffle
        self.random_crop = random_crop
 
        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.store_labels = store_labels
        self.is_s2s = is_s2s
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], noise_prob, noise_snr, noise_num
 
        self.is_training = is_training
        self.sil_phones = ["sil", "sp", "spn", ""]
        # assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert (
            label_processors is None
            or len(label_processors) == self.num_labels
        )
        if not skip_verify:
            for label_path, label_rate in zip(label_paths, self.label_rates):
                verify_label_lengths(self.sizes, self.sample_rate, label_path, label_rate, inds, tot)
        else:
            logger.info(f"Skip label alignment verifying")
 
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(image_mean, image_std) ]
                + ([custom_utils.RandomErase(0.5)] if random_erase else [])
                + ([custom_utils.TimeMask()] if time_mask else []) )
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        logger.info(f"image transform: {self.transform}")
 
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )

        """For duration, f0, energy
        """
        self.use_duration = use_duration
        self.dur_dictionary = dur_dictionary
        self.f0_dictionary = f0_dictionary
        self.energy_dictionary = energy_dictionary
        self.pads = Paddings(
            dur_val=0,
            f0_val=self.f0_dictionary.pad(),  # use 0 for duration padding
            energy_val=self.energy_dictionary.pad(),
        )
        self.shifts = Shifts(shifts, pads=self.pads)
 
    def load_additional_feature(self, mix_name):
        video_fn, audio_fn = mix_name
 
        mel_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/mel/')[:-4]+'.npy'
        if os.path.exists(mel_fn):
            mel = np.load(mel_fn)
        else:
            raise FileNotFoundError(f"{mel_fn} does not exist")
                                                                                    #0.5s_mel_prompt
                                                                                    #3s_mel_prompt
        mel_prompt_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/3s_mel_prompt/')[:-4]+'.npy'
        if os.path.exists(mel_prompt_fn):
            mel_prompt = np.load(mel_prompt_fn)
        else:
            raise FileNotFoundError(f"{mel_prompt_fn} does not exist")
        #spk_emb_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/spk_emb/')[:-4]+'.npy'
                                                                                    #spk_emb_ytts
                                                                                    #spk_emb_metavoice
        spk_emb_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/spk_emb_ytts/')[:-4]+'.npy'
        if os.path.exists(spk_emb_fn):
            spk_emb = np.load(spk_emb_fn)
        else:
            raise FileNotFoundError(f"{spk_emb_fn} does not exist")
       

 
        phonenme_emb_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/g2p_phone/')[:-4]+'.npy'
        if os.path.exists(phonenme_emb_fn):
            phonenme_emb = np.load(phonenme_emb_fn)
        else:
            raise FileNotFoundError(f"{phonenme_emb_fn} does not exist")

 

        """ Loading f0, duration, energy
        """
        dur, f0, energy = self.load_duration_f0_energy(video_fn=os.path.join(self.audio_root, video_fn))

        return mel_fn, mel, mel_prompt,  spk_emb, phonenme_emb, dur, f0, energy
 
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        #self.f0_dictionary.print_tokens()
        path, mel, mel_prompt, spk_emb, phonenme_emb, dur, f0, energy = self.load_additional_feature(self.names[index])
        
        
        mel = torch.from_numpy(mel.astype(np.float32))
        mel_prompt = torch.from_numpy(mel_prompt.astype(np.float32))
        spk_emb /= np.linalg.norm(spk_emb)
        spk_emb = torch.from_numpy(spk_emb.astype(np.float32))
        # style_emb /= np.linalg.norm(style_emb)
        # style_emb = torch.from_numpy(style_emb.astype(np.float32))
 
        sample["path"] = path
        sample["mel"] = mel
        sample["mel_prompt"] = mel_prompt
        sample["spk_emb"] = spk_emb.squeeze(0)
        sample["phone_emb"] = phonenme_emb

        """
        Loading duration, f0, energy
        """
        #dur, f0, energy = self.get_raw_item(dur, f0, energy)
        
    
        # Add prosody to sample
        sample["prosody"] = {
            "f0": torch.from_numpy(f0.astype(np.float32)).unsqueeze(-1),
            "energy": torch.from_numpy(energy.astype(np.float32)).unsqueeze(-1)
        }
                
       
        
       
        return sample
 
    def collater(self, samples):
        batch = super().collater(samples)
        if len(samples) == 0:
            return {}
 
        batch["net_input"]["source"]["path"] = [s["path"] for s in samples]
        max_mel_len = max(len(s["mel"]) for s in samples)
        batch["mel"] = torch.stack([torch.nn.functional.pad(s["mel"], [0, 0, 0, max_mel_len - len(s["mel"])]) for s in samples])
       
        batch["net_input"]["spk_emb"] = torch.stack([s["spk_emb"] for s in samples])

        max_mel_prompt_len = max(len(s["mel_prompt"]) for s in samples)
        batch["net_input"]["source"]["mel_prompt"] = torch.stack([torch.nn.functional.pad(s["mel_prompt"], [0, 0, 0, max_mel_prompt_len - len(s["mel_prompt"])]) for s in samples])
        batch["net_input"]["source"]["mel_prompt_mask"] = self.create_padded_tensor(samples, max_mel_prompt_len)
       

 
        
   
        if self.is_training:
        # #phone_emb lay tu liptext su dung g2p
            phone_list = [s["phone_emb"] for s in samples]
            phones, phone_mask = self.process_phone_embedding(phone_list)
            batch["net_input"]["source"]["phone_emb"] = torch.stack([s for s in phones])
            batch["net_input"]["source"]["phone_mask"] = torch.stack([s for s in phone_mask])
 
        else:
           
            phone_list = [s["phone_emb"] for s in samples]
            phones, phone_mask = self.process_phone_embedding(phone_list)
            batch["net_input"]["source"]["phone_emb"] = torch.stack([s for s in phones])
            batch["net_input"]["source"]["phone_mask"] = torch.stack([s for s in phone_mask])


        """ For duration, f0, energy
        """
        max_f0_len = max(len(s["prosody"]["f0"]) for s in samples)
        batch["f0_target"] = torch.stack([torch.nn.functional.pad(s["prosody"]["f0"], [0, 0, 0, max_f0_len - len(s["prosody"]["f0"])]) for s in samples])
        batch["f0_mask"] = self.create_padded_tensor(samples, max_f0_len)

        max_energy_len = max(len(s["prosody"]["energy"]) for s in samples)
        batch["energy_target"] = torch.stack([torch.nn.functional.pad(s["prosody"]["energy"], [0, 0, 0, max_energy_len - len(s["prosody"]["energy"])]) for s in samples])
        batch["energy_mask"] = self.create_padded_tensor(samples, max_energy_len)
        
        
        batch["net_input"]["source"]["f0_target"] =  torch.stack([torch.nn.functional.pad(s["prosody"]["f0"], [0, 0, 0, max_f0_len - len(s["prosody"]["f0"])]) for s in samples])
        batch["net_input"]["source"]["f0_mask"] =  self.create_padded_tensor(samples, max_f0_len)
        
        
        batch["net_input"]["source"]["energy_target"] =  torch.stack([torch.nn.functional.pad(s["prosody"]["energy"], [0, 0, 0, max_energy_len - len(s["prosody"]["energy"])]) for s in samples])
        batch["net_input"]["source"]["energy_mask"] =  self.create_padded_tensor(samples, max_energy_len)
        return batch
    
 
    
    """ Additional functions for loading and processing
    """
    def get_raw_item(self, dur, f0, energy):
        dur = torch.from_numpy(dur.astype(np.float32))
        f0 = torch.from_numpy(f0.astype(np.float32))
        energy = torch.from_numpy(energy.astype(np.float32))

        return dur, f0, energy
    def prepare_for_one_stream(self, stream, dictionary, is_dur=False, append_BOS_EOS=True):
        """Prepare for one stream

        Args:
            stream: stream to prepare
            dictionary: dictionary to use
            is_dur: whether the stream is duration
            
        Returns:
            stream: prepared stream
        """
        stream = dictionary.encode_line(
            " ".join(map(str, stream.tolist())), append_eos=False, prepend_bos=False
        ).long()
        if append_BOS_EOS == False:
            return stream
        if is_dur:
            stream = torch.cat([stream.new([0]), stream])
            stream = torch.cat([stream, stream.new([0])])
        else:
            stream = torch.cat([stream.new([dictionary.bos()]), stream])
            stream = torch.cat([stream, stream.new([dictionary.eos()])])

        return stream
    def prepare_initial_f0_energy(self, f0_dictionary, energy_dictionary):
        init_f0 = torch.tensor([f0_dictionary.bos()]).long()
        init_energy = torch.tensor([energy_dictionary.bos()]).long()
        return init_f0, init_energy
    def prepare_initial_duration(self, duration_dictionary):
        init_dur = 0
        return init_dur
    def prepare_mel_input(self, f0, energy, dur):
        """Prepare for mel generator

        Args:
            f0: f0
            energy: energy
            dur: duration

        Returns:
            f0_mel_input: f0 input for mel_generator
            energy_mel_input: energy input for mel_generator
        """
        # Convert to numpy arrays if they aren't already
        f0 = f0.numpy() if isinstance(f0, torch.Tensor) else f0
        energy = energy.numpy() if isinstance(energy, torch.Tensor) else energy
        dur = dur.numpy() if isinstance(dur, torch.Tensor) else dur
        
        # Use np.repeat to efficiently repeat each value according to duration
        f0_mel_input = np.repeat(f0, dur)
        energy_mel_input = np.repeat(energy, dur)
        
        # Convert back to torch tensors if needed
        return torch.from_numpy(f0_mel_input), torch.from_numpy(energy_mel_input)
    

    def load_duration_f0_energy(self, video_fn):
        """Load duration, f0, energy from the given video file path

        Args:
            video_fn: path to the video file
            
        Returns:
            dur: duration
            f0: f0
            energy: energy
        """
        f0_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/pitch_energy_aT/frame/pitch/')[:-4]+'.npy'
        
        if os.path.exists(f0_fn):
            f0 = np.load(f0_fn, mmap_mode="r")
        else:   
            raise FileNotFoundError(f"{f0_fn} does not exist")

       
        if os.path.exists(dur_fn):
            dur = np.load(dur_fn, mmap_mode="r")
        else:
            raise FileNotFoundError(f"{dur_fn} does not exist")
        
        energy_fn = os.path.join(self.audio_root, video_fn).replace('/video/', '/pitch_energy_aT/frame/energy/')[:-4]+'.npy'
        
        if os.path.exists(energy_fn):
            energy = np.load(energy_fn, mmap_mode="r")
        else:
            raise FileNotFoundError(f"{energy_fn} does not exist")
        if self.use_duration:
            assert dur.shape == f0.shape == energy.shape, f"Duration, F0, and Energy must have the same shape. Got: {dur.shape}, {f0.shape}, {energy.shape} \n {dur} \n {f0} \n {energy} \n {energy_fn}"
        else:
            assert f0.shape == energy.shape, f"Duration, F0, and Energy must have the same shape. Got: {dur.shape}, {f0.shape}, {energy.shape} \n {dur} \n {f0} \n {energy} \n {energy_fn}"
        
        return dur, f0, energy
    def preprocess_duration(self, duration):
 
        durs, dur_mask = pad_1D(duration)
        durs = torch.from_numpy(durs).long()
        durs_lens = np.array([text.shape[0] for text in durs])
        durs_lens = torch.from_numpy(durs_lens).long()
        dur_mask = get_mask_from_lengths(durs_lens)
        return durs, dur_mask
        #duration
        # dur_list = [s["duration"] for s in samples]
 
        # dur, dur_mask = self.preprocess_duration(dur_list)
 
        # batch["net_input"]["source"]["duration"] = torch.stack([s for s in dur])
        # batch["net_input"]["source"]["duration_mask"] = torch.stack([s for s in dur])
   
 
    def _preprocess_texts(self, texts, processing_function):
        texts = [processing_function(text) for text in texts]
        texts = [np.array(text) for text in texts]
        text_lens = np.array([text.shape[0] for text in texts])
        text_lens = torch.from_numpy(text_lens).long()
        texts, padding_mask = pad_1D(texts)
        texts = torch.from_numpy(texts).long()
        text_mask = get_mask_from_lengths(text_lens)
        return texts, text_mask
 
    def preprocess_textgrid(self, texts):
        return self._preprocess_texts(texts, process_text_texgrid)
 
    def preprocess_text(self, texts):
        return self._preprocess_texts(texts, process_text)
        # phone_list = [s["lip_text"] for s in samples]
        # #phones, phone_mask = self.preprocess_text(phone_list)
        # batch["net_input"]["source"]["phone_emb"] = torch.stack([s for s in phones])
        # batch["net_input"]["source"]["phone_mask"] = torch.stack([s for s in phone_mask])
   
    def preprocess_text_BPE(self, texts):
        return self._preprocess_texts(texts, process_text_BPE)
 
 
    def process_phone_embedding(self, texts):
        texts = [np.array(text) for text in texts]
        text_lens = np.array([text.shape[0] for text in texts])
        text_lens = torch.from_numpy(text_lens).long()
        texts, padding_mask = pad_1D(texts)
        texts = torch.from_numpy(texts).long()
        text_mask = get_mask_from_lengths(text_lens)
        return texts, text_mask
    def get_mask_from_lengths(lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
 
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).cuda()
        mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
        return mask
    def create_padded_tensor(self, samples, max_mel_len):
        padded_tensors = []
        
        for sample in samples:
            mel_length = len(sample["mel"])
            # Create tensor of ones with length equal to mel sequence
            ones = torch.ones(mel_length)
            # Create tensor of zeros for padding
            padding = torch.zeros(max_mel_len - mel_length)
            # Concatenate ones and padding
            padded_sequence = torch.cat([ones, padding])
            padded_tensors.append(padded_sequence)
        
        # Stack all padded sequences into a single tensor
        result = torch.stack(padded_tensors)
        
        return result