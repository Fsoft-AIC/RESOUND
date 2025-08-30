import torch
import torch.nn as nn

class DualScaleProsodyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, local_dim, global_dim):
        """
        Args:
            input_dim (int): Dimension of input features (dm)
            hidden_dim (int): Dimension of hidden state in GRU
            local_dim (int): Dimension of local-scale output (dL)
            global_dim (int): Dimension of global-scale output (dG)
        """
        super(DualScaleProsodyNetwork, self).__init__()
        
        # Shared GRU layer for both scales
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Local-scale output layers
        self.local_fc = nn.Linear(hidden_dim, local_dim)
        self.local_activation = nn.Tanh()
        
        # Global-scale output layers
        self.global_fc = nn.Linear(hidden_dim, global_dim)
        self.global_activation = nn.Tanh()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, TL, dm)
                where TL is sequence length and dm is feature dimension
                
        Returns:
            tuple: (local_output, global_output)
                - local_output: tensor of shape (batch_size, TL, dL)
                - global_output: tensor of shape (batch_size, dG)
        """
        # Pass input through GRU
        gru_output, gru_hidden = self.gru(x)
        # gru_output shape: (batch_size, TL, hidden_dim)
        # gru_hidden shape: (1, batch_size, hidden_dim)
        
        # Local-scale processing: use full sequence
        local_output = self.local_fc(gru_output)
        local_output = self.local_activation(local_output)
        
        # Global-scale processing: use only final hidden state
        global_output = self.global_fc(gru_hidden[-1])
        global_output = self.global_activation(global_output)
        
        return local_output, global_output
def test_network():
    # Define parameters
    batch_size = 16
    seq_length = 50  # TL
    input_dim = 256  # dm
    hidden_dim = 512
    local_dim = 128  # dL
    global_dim = 64  # dG
    
    # Create model
    model = DualScaleProsodyNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        local_dim=local_dim,
        global_dim=global_dim
    )
    
    # Create sample input
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    local_output, global_output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Local output shape: {local_output.shape}")
    print(f"Global output shape: {global_output.shape}")
    
    return model, local_output, global_output

if __name__ == "__main__":
    test_network()