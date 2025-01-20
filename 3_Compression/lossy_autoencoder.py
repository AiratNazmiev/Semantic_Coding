import torch.nn as nn


class LossyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.PReLU(num_parameters=64),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.PReLU(num_parameters=128),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1),
            nn.PReLU(num_parameters=64),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
            nn.PReLU(num_parameters=32),
        )
        
    def forward(self, x):
        y = self.encoder(x)
        return y
    
    
    
class LossyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.PReLU(num_parameters=32),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.PReLU(num_parameters=32),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=7),
            nn.PReLU(num_parameters=16),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=10),
            nn.PReLU(num_parameters=16),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5),
            nn.PReLU(num_parameters=8),
            
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3),
            
            nn.Sigmoid()  # to fit into [0; 1]
        )
        
    def forward(self, x):
        y = self.decoder(x)
        return y
    
    
class LossyAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = LossyEncoder()
        self.decoder = LossyDecoder()
        
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