import torch
import torch.nn as nn
import torch.optim as optim
from skimage.color import rgb2gray
import warnings
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision
import torchvision.models as models

import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
from models import Discriminator
from models import Generator
import scipy.stats
from utils import *
from matplotlib import pyplot as plt
from torch.autograd import Variable
batch_size=64
generate_size=(48,48)
ckpt='./ckpt/'

#Cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_para(net, bool):
    for para in net.parameters():
        para.requires_grad=bool

if __name__ == '__main__':
    #Pre-process of training data and test data
    transforms_cifa_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))

    ])
    transforms_cifa_test = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    transforms_minist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    #Download the dataset from the internet and get the loader object
    trainset_minist=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms_minist)
    trainloader_minist = torch.utils.data.DataLoader(trainset_minist, batch_size=batch_size, shuffle=True,num_workers=2)
    testset_minist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms_minist)
    testloader_minist = torch.utils.data.DataLoader(testset_minist, batch_size=batch_size, shuffle=True,num_workers=2)

    trainset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifa_train)
    trainloader_cifar = torch.utils.data.DataLoader(trainset_cifar, batch_size=batch_size, shuffle=True, num_workers=2)
    testset_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifa_test)
    testloader_cifar= torch.utils.data.DataLoader(testset_cifar, batch_size=batch_size, shuffle=True, num_workers=2)



    # GAN
    disc_cifar=Discriminator().to(device)
    gen_cifar=Generator().to(device)
    criterion = nn.BCELoss().to(device)
    class_criterion= nn.CrossEntropyLoss()

    epochs=200

    disc_cifar_optimizer = torch.optim.Adam(disc_cifar.parameters(), lr=0.0003)
    gen_cifar_optimizer = torch.optim.Adam(gen_cifar.parameters(), lr=0.0003)


    ep=1
    jsd_sum=[]
    ind=0
    k=5
    for epoch in range(epochs):
        for i, (img, labels) in enumerate(trainloader_cifar):
            disc_cifar.train()
            gen_cifar.train()
            # Train discriminator
            #set_para(gen_cifar,False)
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.size(0), 100)))
            gen_labels=torch.cuda.LongTensor(np.random.randint(0, 10, img.size(0)))


            img=img.to(device)
            labels=labels.to(device)
            real_label = torch.ones(img.size(0)).to(device)
            fake_label = torch.zeros(img.size(0)).to(device)

            y_real,label_real=disc_cifar(img)
            loss_real=(criterion(y_real, real_label)+class_criterion(label_real,labels))/2

            noise=torch.randn((img.size(0), 3, generate_size[0], generate_size[1])).to(device)
            #gen_res= gen_cifar(noise)
            gen_res= gen_cifar(z,gen_labels)
            y_fake,label_fake= disc_cifar(gen_res)

            loss_fake=(criterion(y_fake, fake_label)+class_criterion(label_fake,labels))/2

            loss= (loss_real + loss_fake)/2
            disc_cifar_optimizer.zero_grad()
            loss.backward()
            disc_cifar_optimizer.step()
            #set_para(gen_cifar, True)



            # Train generator
            #set_para(disc_cifar,False)
            #noise = torch.randn((img.size(0), 3, generate_size[0], generate_size[1])).to(device)

            gen_res= gen_cifar(z,gen_labels)
            d_res,label=disc_cifar(gen_res)
            print(d_res[0].item())
            gen_loss=(criterion(d_res,real_label)+class_criterion(label,labels))/2

            gen_cifar_optimizer.zero_grad()
            gen_loss.backward()
            gen_cifar_optimizer.step()

            #set_para(disc_cifar,True)
            with torch.no_grad():
                gen_cifar.eval()
                jsd=JSD(img,gen_res)

                print('Epoch-->{} || Loss-->d:{} g:{} || JSD-->{}'.format(ep,loss,gen_loss,jsd))
                jsd_sum.append(jsd)
                ind+=1
        if ep % 10 == 0:
            torch.save(disc_cifar,ckpt+'disc_cifar'+str(epoch)+'.pth')
            torch.save(gen_cifar,ckpt+'gen_cifar'+str(epoch)+'.pth')
        ep+=1

    plt.subplot(1, 1, 1)
    plt.plot(range(0, ind), jsd_sum,'o-')
    plt.title('JSD')
    plt.ylabel('Jsd')
    plt.xlabel('Epochs')
