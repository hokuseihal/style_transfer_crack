from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from conv_vae_model import CVAE
from PIL import Image

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
#device="cpu"


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
class crackDataset(torch.utils.data.Dataset):
    def __init__(self,rawroot,maskroot):
        self.rawroot=rawroot
        self.maskroot=maskroot
        assert len(os.listdir(rawroot))==len(os.listdir(maskroot))
        self.imglist=os.listdir(rawroot)
        self.transformin=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize((0,0,0),(1,1,1))])
        self.transformout=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((256,256)),transforms.ToTensor()])

    def __len__(self):
        return len(os.listdir(self.rawroot))
    def __getitem__(self, item):
        return self.transformin(Image.open(os.path.join(self.rawroot,self.imglist[item]))), self.transformout(Image.open(os.path.join(self.maskroot,self.imglist[item])))
train_loader = torch.utils.data.DataLoader(
    crackDataset("/home/hokusei/Downloads/crack_segmentation_dataset/train/images","/home/hokusei/Downloads/crack_segmentation_dataset/train/masks"),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    crackDataset("/home/hokusei/Downloads/crack_segmentation_dataset/test/images","/home/hokusei/Downloads/crack_segmentation_dataset/test/masks"),
    batch_size=args.batch_size, shuffle=True, **kwargs)
#train_loader=torch.utils.data.DataLoader(datasets.STL10('data',download=True,transform=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),batch_size=256,shuffle=True)
#test_loader=torch.utils.data.DataLoader(datasets.STL10('data',download=True,transform=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),batch_size=256,shuffle=True)

from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter()



model = CVAE().to(device)
optimizer = optim.Adam(model.parameters())


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x.view(-1), x.view(-1), reduction='mean')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE
loss_function= lambda x,y:F.binary_cross_entropy(x.view(-1),y.view(-1))
loss_function=F.mse_loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target=data.to(device)
        optimizer.zero_grad()
        recon_batch= model(data)
        loss = loss_function(recon_batch, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)))
    writer.add_scalar('Loss:Train',train_loss / len(train_loader),epoch)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = data.to(device)
            recon_batch= model(data)
            test_loss += loss_function(recon_batch, target).item()
            if i == 0:
                n = min(data.size(0), 8)
                #sample = torch.randn(8, 512, 2, 2).to(device)
                #sample = model.decoder(sample)
                img=torch.cat([data[:n],recon_batch[:n]]).cpu()
                save_image(img,
                           'results/result' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    writer.add_scalar('Loss:Test', test_loss, epoch)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
