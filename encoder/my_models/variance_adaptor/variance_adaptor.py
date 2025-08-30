import torch
import torch.nn as nn
import sys
import os
from omegaconf import OmegaConf
from collections import OrderedDict
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..'))  )
from utils.utils import pad, get_mask_from_lengths

class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor for multi-target lip2speech
    """
    def __init__(self, cfg):
        super(VarianceAdaptor, self).__init__()
        self.cfg = cfg

        self.duration_predictor = DurationPredictor(cfg)
        self.length_regulator = LengthRegulator()
        # self.pitch_predictor = PitchPredictor(cfg)
        # self.energy_predictor = EnergyPredictor(cfg)

    def forward(self, x, src_mask, duration_target=None, mel_mask=None, max_len=None, d_control=1.0):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
        mel_len = torch.tensor(mel_len, dtype=torch.long) 
        mel_mask = get_mask_from_lengths(mel_len)


        return (
                x,
                log_duration_prediction,        
                mel_mask,
            )        

class DurationPredictor(nn.Module):
    """
    Duration Predictor for multi-target lip2speech
    """
    def __init__(self, variance_config):
            super(DurationPredictor, self).__init__()

            self.input_size = variance_config.transformer_encoder_hidden
            self.filter_size = variance_config.variance_predictor_filter_size
            self.kernel = variance_config.variance_predictor_kernel_size
            self.conv_output_size = variance_config.variance_predictor_filter_size
            self.dropout = variance_config.variance_predictor_dropout
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

            # self.conv_layer2 = nn.Sequential(
            #     OrderedDict(
            #         [
            #             (
            #                 "conv1d_1",
            #                 Conv(
            #                     self.input_size,
            #                     self.filter_size,
            #                     kernel_size=1,
            #                 ),
            #             ),
            #             ("relu_1", nn.ReLU()),
            #             ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            #             ("dropout_1", nn.Dropout(self.dropout)),
            #             (
            #                 "conv1d_2",
            #                 Conv(
            #                     self.filter_size,
            #                     self.filter_size,
            #                     kernel_size=1,
            #                 ),
            #             ),
            #             ("relu_2", nn.ReLU()),
            #             ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            #             ("dropout_2", nn.Dropout(self.dropout)),
            #         ]
            #     )
            # )

        
            # self.conv = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0, dilation=5)
            # self.softplus = nn.Softplus()
            self.linear_layer = nn.Linear(self.conv_output_size, 1)
        
    # def forward(self, encoder_output, mask):
    #     out = self.conv_layer(encoder_output)
    #     out = self.conv(out.transpose(1, 2)).transpose(1, 2)
    #     out = self.softplus(out)
    #     out = out.squeeze(-1)
    #     if mask is not None:
    #         out = out.masked_fill(mask, 0.0)
    #     return out
    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

class LengthRegulator(nn.Module):
    """
    Length Regulator for multi-target lip2speech
    """
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = []
        mel_len = []
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)  
        else:
            output = pad(output)

        return output, mel_len

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
        
        


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

def main():
    # Define configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the config file (two directories up)
    config_path = os.path.join(current_dir, '..', 'conf.yaml')
    print(config_path)
    cfg = OmegaConf.load(config_path)
    # Create VarianceAdaptor instance
    variance_adaptor = VarianceAdaptor(cfg)

    # Create input tensor with shape (batch, len, dim)
    batch_size = 2
    seq_len = 10
    hidden_dim = 256
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Create source mask
        # Create a random source mask with at least one True value per sequence
    src_mask = torch.rand(batch_size, seq_len) > 0.5
    src_mask[:, 0] = True  # Ensure the first element of each sequence is True
    # Forward pass
    output = variance_adaptor(x, src_mask)

    # Print output shapes
    print("Input shape:", x.shape)
    print("Output shapes:")
    print("- x:", output[0].shape)
    print("- log_duration_prediction:", output[1].shape)
    print("- duration_rounded:", output[2].shape)
    print("- mel_len:", output[3])
    print("- mel_mask:", output[4].shape)

if __name__ == "__main__":
    main()