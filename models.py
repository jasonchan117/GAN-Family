import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torch


class Discriminator(nn.Module):

    def __init__(self,in_c=3,out=3,dropout=0):
        super(Discriminator, self).__init__()
        self.resnet=models.resnet34(pretrained=True)
        self.seq=nn.Sequential(
            nn.Conv2d(in_c, out*3, 5, 1, bias=False),
            nn.BatchNorm2d(out*3),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(out*3, out*3, 5, 1, bias=False),
            nn.BatchNorm2d(out*3),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(out*3, out, 5, 1, bias=False),
            nn.BatchNorm2d(out),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.AvgPool2d(2, stride=2)
        )
        self.fc = nn.Linear(300, 1)
        self.fc_label=nn.Linear(300,10)
        self.sm=nn.Softmax()
    def forward(self, x):

        x = self.seq(x)
        x=x.view(x.size(0), -1)
        #print(x.size())
        out= self.fc(x)
        out_label= self.fc_label(x)
        out_label=self.sm(out_label)
        return torch.sigmoid(out) , out_label


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 100)

        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



LATENT_CODE_NUM = 32
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

        self.fc11 = nn.Linear(128 * 7 * 7, LATENT_CODE_NUM)
        self.fc12 = nn.Linear(128 * 7 * 7, LATENT_CODE_NUM)

        self.fc2 = nn.Linear(LATENT_CODE_NUM, 128 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())



    def reparameterize(self, mu, logvar):

        eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        z = mu + eps * torch.exp(logvar / 2)
        return z


    def forward(self, x):

        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        z = self.reparameterize(mu, logvar)  # batch_s, latent
        out3 = self.fc2(z).view(z.size(0), 128, 7, 7)  # batch_s, 8, 7, 7
        return self.decoder(out3), mu, logvar