import torch.nn as nn
import torch
import torch.nn.functional as F
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,groups),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU()
        )

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder=nn.Sequential(
            ConvBNReLU(3,32,4,2),
            ConvBNReLU(32,64,4,2),
            ConvBNReLU(64,128,4,2),
            ConvBNReLU(128,256,4,2),
            ConvBNReLU(256,512,4,2)
        )
        self.decoder=nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBNReLU(512,256, 3),
            nn.Upsample(scale_factor=2),
            ConvBNReLU(256, 128, 3),
            nn.Upsample(scale_factor=2),
            ConvBNReLU(128, 64, 3),
            nn.Upsample(scale_factor=2),
            ConvBNReLU(64, 32, 3),
            nn.Upsample(scale_factor=2),
            ConvBNReLU(32, 3, 3),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z= self.encoder(x)
        mu=z.mean()
        var=z.var()
        return self.decoder(z),var,mu

