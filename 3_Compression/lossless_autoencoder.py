import torch.nn as nn


class LosslessEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 784),
            nn.PReLU(num_parameters=784),
            
            nn.Linear(784, 784),
            nn.PReLU(num_parameters=784),
        )
        
    def forward(self, x):
        y = self.encoder(x)
        return y
    
    
    
class LosslessDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(784, 784),
            nn.PReLU(num_parameters=784),
            
            nn.Linear(784, 784),
            nn.PReLU(num_parameters=784),
            nn.Unflatten(dim=-1, unflattened_size=(1, 28, 28)),
            nn.Sigmoid()  # to fit into [0; 1]
        )
        
    def forward(self, x):
        y = self.decoder(x)
        return y
    
    
class LosslessAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = LosslessEncoder()
        self.decoder = LosslessDecoder()
        
    def encode(self, x, flatten=True):
        enc = self.encoder(x)
        if flatten:
            enc = enc.flatten(start_dim=-2)
        return enc
    
    def decode(self, latent):
        return self.decoder(latent)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        return reconstructed