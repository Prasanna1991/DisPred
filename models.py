import torch
import torch.nn.functional as F
from torch import nn

class GenomeAE_disentgl(torch.nn.Module):
  
    def __init__(self, dim=4973, zd = 100, ze = 20, acti=1):
        super(GenomeAE_disentgl, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(dim, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc_d = nn.Linear(200, zd) # disease component
        self.fc_e = nn.Linear(200, ze) # ethnicity component

        # Decoder
        self.fc1_decode = nn.Linear(zd + ze, 200)
        self.fc2_decode = nn.Linear(200, 400)
        self.fc3_decode = nn.Linear(400, dim)
  
        self.outLayer = torch.nn.Hardtanh(0, 2)
        self.distance = nn.PairwiseDistance(2)
        self.acti = acti 

    def getDist(self, s1, s2):
        return self.distance(s1, s2).view(-1, s1.size(0))

    def encode(self, x):
        if self.acti == 1: #Non-linear
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else: #Linear
            x = self.fc1(x)
            x = self.fc2(x)
        z_d = self.fc_d(x)
        z_e = self.fc_e(x)
        return z_d, z_e


    def decode(self, z_d, z_e):
        z = torch.cat([z_d, z_e], 1)
        if self.acti == 1: #Non-linear
            x_hat = F.relu(self.fc1_decode(z))
            x_hat = F.relu(self.fc2_decode(x_hat))
            x_hat = self.outLayer(self.fc3_decode(x_hat))
        else: #Linear
            x_hat = self.fc1_decode(z)
            x_hat = self.fc2_decode(x_hat)
            x_hat = self.fc3_decode(x_hat)          
        return x_hat

    def forward(self, x):
        z_d, z_e = self.encode(x)
        x_hat = self.decode(z_d, z_e)
        return z_d, z_e, x_hat  

