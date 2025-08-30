import torch
import torch.nn as nn
#from Modules import *
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
class SpeechPromptEncoder(nn.Module):
    """
    Speech Prompt Encoder
    - Mel-Spec Pre-net:
        3x
            -  Conv1D
            -  ReLU
            -  LayerNorm
            -  Dropout
    - Positional Encoding
    - 4x FFT Blocks


    """
    def __init__(self, config):
        
        super(SpeechPromptEncoder, self).__init__()
        self.config = config

        """Conv1D params
        """
        self.mel_channels = config.mel_channels
        self.conv_channels = config.conv_channels
        self.conv_kernel = config.conv_kernel
        self.dropout = config.dropout
        self.hidden_embed_dim = config.hidden_embed_dim

        """FFT Blocks params
        """
        self.nb_block = config.nb_block
        self.max_seq_len = config.max_seq_len
        """"""

        self.pre_convs = nn.Sequential(
            ConvNorm1D(in_channels=self.mel_channels, out_channels=self.conv_channels, 
                       kernel_size=self.conv_kernel, padding=int((self.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.GELU(),
            nn.LayerNorm(self.conv_channels),
            nn.Dropout(self.dropout),

            ConvNorm1D(in_channels=self.conv_channels, out_channels=self.conv_channels, 
                       kernel_size=self.conv_kernel, padding=int((self.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.GELU(),
            nn.LayerNorm(self.conv_channels),
            nn.Dropout(self.dropout),
            
            ConvNorm1D(in_channels=self.conv_channels, out_channels=self.conv_channels, 
                       kernel_size=self.conv_kernel, padding=int((self.conv_kernel - 1) / 2),
                       dilation=1, w_init_gain='relu'),
            nn.GELU(),
            nn.LayerNorm(self.conv_channels),
            nn.Dropout(self.dropout)
        )

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(self.max_seq_len, self.hidden_embed_dim).unsqueeze(0),
            requires_grad=False,
        )
        self.fft_blocks = []
        for  _ in range(self.nb_block):
            self.fft_blocks.append(FFTBlock(config))
    def forward(self, mel_specs: torch.Tensor, mel_mask: torch.Tensor):
        
        ''' Forward function of Prompt Speech Encoder:
            
            mel_specs = (B, nb_mels, T_max)
            speaker_ids = (B, )
            output_lengths = (B, )
        '''

        mel_specs = self.pre_convs(mel_specs)

        batch_size, seq_len = mel_specs.shape[0], mel_specs.shape[1]
        
        if self.training and seq_len > self.max_seq_len:
            pe = get_sinusoid_encoding_table(seq_len, self.hidden_embed_dim
                                             )[:mel_specs.size(1),:].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            pe = self.position_enc[:, :seq_len, :].expand(batch_size, -1, -1)
        assert pe.shape == mel_specs.shape,  f"shape of pe {pe.shape}, shape of mel_specs {mel_specs.shape}"

        mel_specs = mel_specs + pe


        mel_specs = mel_specs.masked_fill(mel_mask.unsqueeze(2), 0)

        for fft_block in self.fft_blocks:
            mel_specs = fft_block(x=mel_specs,film_params=None,mask=mel_mask)
        return mel_specs
class Config:
    def __init__(self):
        # Mel-spectrogram parameters
        self.mel_channels = 80  # Number of mel channels (input dimension)
        
        # Conv1D parameters
        self.conv_channels = 256  # Number of channels in conv layers
        self.conv_kernel = 3  # Kernel size for conv layers
        self.dropout = 0.1  # Dropout rate
        self.hidden_embed_dim = 256  # Hidden embedding dimension
        
        # FFT Block parameters
        self.nb_block = 4  # Number of FFT blocks
        self.max_seq_len = 1000  # Maximum sequence length
        
        # FFT Block specific parameters (needed for FFTBlock class)
        self.encoder_dim = 256  # Dimension of encoder
        self.encoder_n_head = 8  # Number of attention heads
        self.encoder_conv1d_filter_size = 1024  # Conv1D filter size in FFT block
        self.fft_conv1d_kernel = 3  # Kernel size for FFT block conv1d
        self.fft_conv1d_padding = 1  # Padding for FFT block conv1d
        self.encoder_dropout = 0.1  # Dropout rate in encoder

        self.attn_dropout = 0.1
        self.attn_nb_heads = 8  
        self.attn_conv1d_filter_size = 1024
        self.attn_conv1d_kernel = 3
        self.attn_conv1d_padding = 1
        self.conv_dropout = 0.1


def get_config():
    return Config()

# Usage example:
def main():
    config = get_config()
    
    # Create sample inputs
    mel_specs = torch.rand(12, 20, config.mel_channels)  # (batch_size, seq_len, mel_channels)
    lengths = torch.randint(10, 20, (12,))  # Random lengths for each sequence
    mel_mask = get_mask_from_lengths(lengths, max_len=20)
    
    # Initialize model
    speech_encoder = SpeechPromptEncoder(config)
    
    # Forward pass\
    print(mel_specs.shape)
    print(mel_mask.shape)
    y = speech_encoder(mel_specs, mel_mask)
    print(y.shape)
if __name__ == "__main__":
    main()