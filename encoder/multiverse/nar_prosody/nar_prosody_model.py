import torch
import torch.nn as nn
from collections import OrderedDict
from .Modules import *
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
class VariancePredictor(nn.Module):
    """ Variance Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config.input_dim
        self.filter_size = model_config.conv_channels
        self.kernel = model_config.conv_kernel
        self.conv_output_size = model_config.conv_channels
        self.dropout = model_config.dropout
        self.output_size = model_config.output_dim

        """FFT Blocks params
        """
        self.nb_block = model_config.nb_block
        self.max_seq_len = model_config.max_seq_len
        self.hidden_embed_dim = model_config.hidden_embed_dim

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(self.max_seq_len, self.hidden_embed_dim).unsqueeze(0),
            requires_grad=False,
        )
        self.fft_blocks = []
        for  _ in range(self.nb_block):
            self.fft_blocks.append(FFTBlock(model_config))

        self.linear_layer = nn.Linear(self.conv_output_size, self.output_size)
        
    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        
        
        batch_size, seq_len = encoder_output.shape[0], encoder_output.shape[1]
        
        if self.training and seq_len > self.max_seq_len:
            pe = get_sinusoid_encoding_table(seq_len, self.hidden_embed_dim
                                             )[:out.size(1),:].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            pe = self.position_enc[:, :seq_len, :].expand(batch_size, -1, -1)
        assert pe.shape == out.shape,  f"shape of pe {pe.shape}, shape of mel_specs {out.shape}"
       
        pe = pe.to(out.device)
        out = out + pe
        #out = out.to(mask.device)
        
        out = out.masked_fill(mask.unsqueeze(2), 0)
        
        for fft_block in self.fft_blocks:
            out = fft_block(x=out,film_params=None,mask=mask)


        out = self.linear_layer(out)
        return out, mask
class NARProsodyPredictor(nn.Module):
    """ NAR Prosody Predictor """
    def __init__(self, config):
        super(NARProsodyPredictor, self).__init__()
        self.pitch_predictor = VariancePredictor(config)
        self.energy_predictor = VariancePredictor(config)

    def forward(self, mel_specs, mask):
        pitch_out, pitch_mask = self.pitch_predictor(mel_specs, mask)
        energy_out, energy_mask = self.energy_predictor(mel_specs, mask)
        return {
            "f0": pitch_out, # B x T x output_size
            "energy": energy_out, # B x T x output_size
            "mask": energy_mask, # B x T
        }
def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
class Config:
    def __init__(self):
        # Mel-spectrogram parameters
        self.input_dim = 512
        
        # NAR Prosody Predictor parameters
        self.conv_channels = 256
        self.conv_kernel = 3
        self.dropout = 0.2
        
        self.output_dim = 65 # 65 prosody features

        """FFT Blocks params
        """
        self.nb_block = 4
        self.max_seq_len = 2000
        self.hidden_embed_dim = 256
# @dataclass
# class NARProsodyConfig:
#     # Input/Output dimensions
#     input_dim: int = field(default=512)      # Input embedding dimension
#     output_dim: int = field(default=65)      # Number of prosody features
    
#     # Convolutional layer parameters
#     conv_channels: int = field(default=256)
#     conv_kernel: int = field(default=3)
#     dropout: float = field(default=0.2)
    
#     # FFT Block parameters
#     nb_block: int = field(default=4)         # Number of FFT blocks
#     max_seq_len: int = field(default=2000)   # Maximum sequence length
#     hidden_embed_dim: int = field(default=256)  # Hidden embedding dimension    
def get_config():
    return Config()

def main():
    # Initialize configuration
    config = get_config()
    
    # Create sample batch
    batch_size = 16
    seq_len = 100  # Example sequence length
    
    # Generate random mel-spectrograms
    mel_specs = torch.randn(batch_size, seq_len, config.input_dim)
    
    # Generate random lengths for masking
    lengths = torch.randint(50, seq_len, (batch_size,))
    mask = get_mask_from_lengths(lengths, seq_len)
    
    # Initialize model
    prosody_predictor = NAR_Prosody_Predictor(config)
    
    
    # Forward pass
    print("\nInput/Output Shapes:")
    print(f"Input mel-specs: {mel_specs.shape}")
    print(f"Mask shape: {mask.shape}")
    
    output = prosody_predictor(mel_specs, mask)
    print(f"Output shape: {output['pitch_out'].shape}")
    print(f"Output shape: {output['energy_out'].shape}")
    print(f"Mask shape: {output['mask'].shape}")


if __name__ == "__main__":
    main()