import torch 
from torch import nn

class ProteinEncoder(nn.Module):
    
    def __init__(self, conv_channels: int = 64, sequence_length: int = 1000, batch_size: int = 32, output_dim: int = 128):
        super().__init__() # type: ignore

        self.conv1 = nn.Conv1d(in_channels=22, out_channels=conv_channels, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=22, out_channels=conv_channels, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=22, out_channels=conv_channels, kernel_size=7)

        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(conv_channels * 3, output_dim)

    
    # Forward pass through the encoder
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # x shape: [batch_size, sequence_length, vocab_size] -> [batch_size, vocab_size, sequence_length]
        x = x.transpose(1, 2) 

        c1 = self.maxpool(self.relu(self.conv1(x))).squeeze(-1)   # Squeeze to remove the last dimension since its size is 1
        c2 = self.maxpool(self.relu(self.conv2(x))).squeeze(-1)  
        c3 = self.maxpool(self.relu(self.conv3(x))).squeeze(-1)

        convolutional_concat = torch.cat([c1, c2, c3], dim=1)  # Concatenate along the channel dimension

        output = self.fc(convolutional_concat)  # Fully connected layer
        return output  # Output shape: [batch_size, output_dim]
