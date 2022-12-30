import sys
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import math
import gc
import h5py

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument("-trn", "--trainsize", help="Training data set size", nargs='?', const=876, default=876, type=int)
parser.add_argument("-tst", "--testsize", help="Test data set size", nargs='?', const=103, default=103, type=int)
parser.add_argument("-err", "--errortype", help="L1 or L2 error [l1/l2]", nargs='?', const="l2", default="l2", type=str)
parser.add_argument("-zerr", "--zerror", help="Incorporate z error [0/1]", nargs='?', const=0, default=0, type=int)
parser.add_argument("-lr", "--learnrate", help="Learning rate", nargs='?', const=0.0005, default=0.0005, type=float)
parser.add_argument("-ep", "--epochs", help="Number of epochs", nargs='?', const=100, default=100, type=int)
parser.add_argument("-norm", "--normmethod", help="normalization method [minmax/zscore]", nargs='?', const="zscore", default="zscore", type=str)
parser.add_argument("-m1", "--mode1", help="Mode 1", nargs='?', const=12, default=12, type=int)
parser.add_argument("-m2", "--mode2", help="Mode 2", nargs='?', const=12, default=12, type=int)
parser.add_argument("-m3", "--mode3", help="Mode 3", nargs='?', const=12, default=12, type=int)
parser.add_argument("-w", "--width", help="Width", nargs='?', const=32, default=32, type=int)

args = parser.parse_args()

#########################################################################################
# load model architecture
#########################################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.resconv_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.resconv_2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.resconv_3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.resconv_4 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.resconv_5 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer0 = self.output_layer(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        
        # resconv 1
        identity0 = out_conv3
        resconv1 = self.resconv_1(out_conv3)
        resconv1 = resconv1 + identity0
        resconv1 = F.relu(resconv1)
        
        # resconv 2
        identity1 = resconv1
        resconv2 = self.resconv_2(resconv1)
        resconv2 = resconv2 + identity1
        resconv2 = F.relu(resconv2)
        
        # resconv 3
        identity2 = resconv2
        resconv3 = self.resconv_3(resconv2)
        resconv3 = resconv3 + identity2
        resconv3 = F.relu(resconv3)
        
        # resconv 4
        identity3 = resconv3
        resconv4 = self.resconv_4(resconv3)
        resconv4 = resconv4 + identity3
        resconv4 = F.relu(resconv4)
        
        # resconv 5
        identity4 = resconv4
        resconv5 = self.resconv_5(resconv4)
        resconv5 = resconv5 + identity4
        resconv5 = F.relu(resconv5)

        out_deconv2 = self.deconv2(resconv5)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer0(concat0)

        return out
    
    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output_layer(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)


class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
#         print(x.shape)
        x3 = self.unet3(x) 
#         print(x3.shape)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

class Net3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(Net3d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width)


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        x = F.pad(F.pad(x, (0,0,0,8,0,8), "replicate"), (0,0,0,0,0,0,0,8), 'constant', 0)
        x = self.conv1(x)
#         print(x.shape)
        x = x.view(batchsize, size_x+8, size_y+8, size_z+8, 1)[..., :-8,:-8,:-8, :]
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

#########################################################################################
# create model
#########################################################################################
mode1 = args.mode1 # [max: 12, i.e. half of 24, the smallest dimension of data set]
mode2 = args.mode2
mode3 = args.mode3
width = args.width # [max: only constrained by memory]
device = torch.device('cuda:0')
model = Net3d(mode1, mode2, mode3, width)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)

#########################################################################################
# load and process data
#########################################################################################
# Load SG Data
hf_r = h5py.File(f'/scratch/groups/smbenson/gegewen/shale_clean.hdf5', 'r')
data_x_shale = np.array(hf_r.get('x'))
data_SG_shale = np.array(hf_r.get('SG'))
hf_r.close()

# Normalization
normmethod = args.normmethod # "minmax" or "zscore"

# MIN MAX NORMALIZATION
if normmethod == "minmax":
    SG_min = np.min(data_SG_shale)
    SG_max = np.max(data_SG_shale)
    
    data_SG_shale = (data_SG_shale - SG_min)/(SG_max - SG_min)
    
# Z SCORE NORMALIZATION
elif normmethod == "zscore":
    SG_mean = np.mean(data_SG_shale)
    SG_std = np.std(data_SG_shale)
    
    data_SG_shale = (data_SG_shale - SG_mean)/(SG_std)
    
data_x = np.concatenate([data_x_shale], axis=0)
data_SG = np.concatenate([data_SG_shale], axis=0)

data_nr = data_x.shape[0]
test_nr = args.testsize
train_nr = args.trainsize
train_nr = train_nr // 4 * 4

np.random.seed(0)
shuffle_index = np.random.choice(data_nr, data_nr, replace=False)

data_x = data_x[shuffle_index, ...]
data_SG = data_SG[shuffle_index, ...]

idx = [0,6,12,18,19,20,21,22,23]
data_x_fit = np.zeros((data_x.shape[0], len(idx)+3, 96, 200))
for j, index in enumerate(idx):
    data_x_fit[:,j,:,:] = data_x[:,index,:,:]
    
dz = 2.083330
dx = [0.1]

with open('DRV.txt') as f:
    for line in f:
        line = line.strip().split('*')
        dx.append(float(line[-1]))
dx = np.cumsum(dx)
grid_x = dx/np.max(dx)
grid_x = grid_x[1:]
grid_y = np.linspace(0, 200, 96)/np.max(dx)

data_x_fit[:,-3,:,:] = grid_x[np.newaxis, np.newaxis, :]
data_x_fit[:,-2,:,:] = grid_y[np.newaxis, :, np.newaxis]
data_x_fit[:,-1,:,:] = np.ones(data_x_fit[:,-1,:,:].shape)

data_x_fit[:,-3,:,:] = data_x_fit[:,-3,:,:]/np.max(data_x_fit[:,-3,:,:])
data_x_fit[:,-2,:,:] = data_x_fit[:,-2,:,:]/np.max(data_x_fit[:,-2,:,:])

x_in = data_x_fit.transpose((0,2,3,1))
SG = data_SG.transpose((0,2,3,1))

x_in = x_in.astype(np.float32)
SG = SG.astype(np.float32)

x_in = torch.from_numpy(x_in)
SG = torch.from_numpy(SG)

# a input u output
train_a = x_in[:train_nr,:,:,:]
train_u = SG[:train_nr,:,:,:]

test_a = x_in[train_nr:train_nr+ test_nr,:,:,:]
test_u = SG[train_nr:train_nr+ test_nr,:,:,:]

T = 24

train_a = train_a[:,:,:,np.newaxis,:]
test_a = test_a[:,:,:,np.newaxis,:]

train_a = train_a.repeat([1,1,1,T,1])
test_a = test_a.repeat([1,1,1,T,1])

print(train_a.shape)
print(test_a.shape)

t = np.cumsum(np.power(1.421245, range(24)))
t /= np.max(t)
for i in range(24):
    train_a[:,:,:,i,-1] = t[i]
    test_a[:,:,:,i,-1] = t[i]
    
del data_x_shale
del data_SG_shale

batch_size = 2

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)

dz = 2.083330
dx = [0.1]

with open('DRV.txt') as f:
    for line in f:
        line = line.strip().split('*')
        dx.append(float(line[-1]))
dx = np.cumsum(dx)
grid_x = dx/np.max(dx)
grid_x = grid_x[1:]
grid_y = np.linspace(0, 200, 96)/np.max(dx)

dx = []
for j in range(1,199):
    dx.append(grid_x[j] + grid_x[j-1]/2 + grid_x[j+1]/2)
dx = np.array(dx)
dx = dx[np.newaxis, np.newaxis, :, np.newaxis]
dx = dx.astype(np.float32)
dx = torch.from_numpy(dx)
dx = dx.to(device)

epochs = args.epochs
learning_rate = args.learnrate
scheduler_step = 20
scheduler_gamma = 0.8

#########################################################################################
# train model
#########################################################################################

class LpLoss(object):
    def __init__(self, d=2, p=1, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

l1orl2 = args.errortype # "l1" or "l2"

if l1orl2 == "l1":
    myloss1 = LpLoss(size_average=False, p=1)
elif l1orl2 == "l2":
    myloss2 = LpLoss(size_average=False, p=2)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/SG3d_FNORUNet_{args.width}width_{args.mode1}m1_{args.mode2}m2_{args.mode3}m3_{train_nr}train_{args.epochs}eps_{args.errortype}err_{args.learnrate}lr_{args.zerror}zerr_{args.normmethod}norm_andrew_FNORUNet3_5layer')
savefn = f'outputs/SG3d_FNORUNet_{args.width}width_{args.mode1}m1_{args.mode2}m2_{args.mode3}m3_{train_nr}train_{args.epochs}eps_{args.errortype}err_{args.learnrate}lr_{args.zerror}zerr_{args.normmethod}norm_andrew_FNORUNet3_5layer.txt'

if l1orl2 == "l1":
    train_l1 = 0.0
elif l1orl2 == "l2":
    train_l2 = 0.0
    
last5epocherr = []

t0 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    if l1orl2 == "l1":
        train_l1 = 0.0
    elif l1orl2 == "l2":
        train_l2 = 0.0
    
    counter = 1
    for x, y in train_loader: # iterating through batches, not single data points
    
        x, y = x.to(device), y.to(device)
        dy_r = (y[:,:,2:,:] - y[:,:,:-2,:])/dx
        if args.zerror == 1:
            dy_z = (y[:,2:,:,:] - y[:,:-2,:,:])/2.083
        
        optimizer.zero_grad()
        out = model(x).view(-1,96,200,24)
        dy_r_pred = (out[:,:,2:,:] - out[:,:,:-2,:])/dx
        if args.zerror == 1:
            dy_z_pred = (out[:,2:,:,:] - out[:,:-2,:,:])/2.083
        
        if l1orl2 == "l1":
            l1 = myloss1(out.reshape(y.shape[0], -1), y.view(y.shape[0], -1)) 
            l1 += myloss1(dy_r_pred.reshape(dy_r_pred.shape[0], -1), dy_r.view(dy_r.shape[0], -1))
            if args.zerror == 1:
                l1 += myloss1(dy_z_pred.reshape(dy_z_pred.shape[0], -1), dy_z.view(dy_z.shape[0], -1))
            l1.backward()
        elif l1orl2 == "l2":
            l2 = myloss2(out.reshape(y.shape[0], -1), y.view(y.shape[0], -1)) 
            l2 += myloss2(dy_r_pred.reshape(dy_r_pred.shape[0], -1), dy_r.view(dy_r.shape[0], -1))
            if args.zerror == 1:
                l2 += myloss2(dy_z_pred.reshape(dy_z_pred.shape[0], -1), dy_z.view(dy_z.shape[0], -1))
            l2.backward()
        
        optimizer.step()
        if l1orl2 == "l1":
            train_l1 += l1.item()
        elif l1orl2 == "l2":
            train_l2 += l2.item()
            
        # writing the epoch loss of each batch of points in the training data set (e.g. 4 reservoirs) to tensorboard
        if l1orl2 == "l1":
            writer.add_scalar('epoch l1 loss', l1.item()/batch_size, ep*len(train_loader) + counter)
        elif l1orl2 == "l2":
            writer.add_scalar('epoch l2 loss', l2.item()/batch_size, ep*len(train_loader) + counter)
        
        if counter % 10 == 0:
            if l1orl2 == "l1":
                print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train l1 loss: {l1.item()/batch_size:.4f}')
            elif l1orl2 == "l2":
                print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train l2 loss: {l2.item()/batch_size:.4f}')
            
        counter += 1
        
    scheduler.step()

    if l1orl2 == "l1":
        train_l1 /= train_a.shape[0]
    elif l1orl2 == "l2":
        train_l2 /= train_a.shape[0]
    
    if epochs - ep <= 5:
        if l1orl2 == "l1":
            last5epocherr.append(train_l1)
        elif l1orl2 == "l2":
            last5epocherr.append(train_l2)

    t2 = default_timer()
    if l1orl2 == "l1":
        print(f'epoch: {ep}, {int(t2-t1)}s, train l1 loss: {train_l1:.4f}')
    elif l1orl2 == "l2":
        print(f'epoch: {ep}, {int(t2-t1)}s, train l2 loss: {train_l2:.4f}')
    
    if ep % 2 == 0 or ep == epochs-1:
        PATH = f'saved_models/SG3d_FNORUNet_{ep}ep_{args.width}width_{args.mode1}m1_{args.mode2}m2_{args.mode3}m3_{train_nr}train_{args.epochs}eps_{args.errortype}err_{args.learnrate}lr_{args.zerror}zerr_{args.normmethod}norm_andrew_FNORUNet3_5layer'
        torch.save(model, PATH)
        
print(f'Finished training a total of {epochs} epochs in {int(t2-t0)} seconds')
print(f'Training {l1orl2} losses for last 5 epochs are {last5epocherr}')

with open(savefn, 'w') as f: # write (since first one)
    f.write(f'Finished training a total of {epochs} epochs in {int(t2-t0)} seconds')
    f.write(f'\nTraining {l1orl2} losses for last 5 epochs are {last5epocherr}')

#########################################################################################
# evaluate model
#########################################################################################

def rsquared(x,y):
    correlation_matrix = np.corrcoef(x.flatten(), y.flatten())
    correlation_xy = correlation_matrix[0,1]
    return correlation_xy**2

def rel_err(yhat, y):
    abs_err = np.abs(y - yhat)
    rel_err = np.zeros(abs_err.shape)
    rel_err = abs_err/(torch.max(y))
    return rel_err

# Training R2
R2_train = []

print("Beginning error analysis for training data set")
t1 = default_timer()
eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).view(-1,96,200,24)

        y_plot = y.cpu().detach().numpy()
        x_plot = x.cpu().detach().numpy()
        pred_plot = pred.cpu().detach().numpy()

        mask = x_plot[0,:,:,0,0] != 0
        fullmask = x_plot[0, :, :, :, 0] != 0
        thickness = sum(mask[:,0])

        R2 = rsquared(pred_plot[0,:,:,:][fullmask], y_plot[0,:,:,:][fullmask])
        #R2s_train[i] = R2
        R2_train.append(R2)
t2 = default_timer()

t3 = default_timer()       
print("Mean R^2 for a random sample of ", train_a.shape[0], " reservoirs from the training data set of size ", np.size(train_u, 0), " is: ", np.mean(R2_train), ". Calculated in ", int(t2 - t1), " seconds.")

with open(savefn, 'a') as f:
    f.write(f'\nMean R^2 for a random sample of {train_a.shape[0]} reservoirs from the training data set of size {np.size(train_u, 0)} is: {np.mean(R2_train)}. Calculated in {int(t2 - t1)} seconds.')
    
    
# Test R2
R2_test = []

print("Beginning error analysis for test data set")
t1 = default_timer()
eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in eval_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).view(-1,96,200,24)

        y_plot = y.cpu().detach().numpy()
        x_plot = x.cpu().detach().numpy()
        pred_plot = pred.cpu().detach().numpy()

        mask = x_plot[0,:,:,0,0] != 0
        fullmask = x_plot[0, :, :, :, 0] != 0
        thickness = sum(mask[:,0])

        R2 = rsquared(pred_plot[0,:,:,:][fullmask], y_plot[0,:,:,:][fullmask])
        #R2s_train[i] = R2
        R2_test.append(R2)
t2 = default_timer()

t3 = default_timer()       
print("Mean R^2 for a random sample of ", test_a.shape[0], " reservoirs from the test data set of size ", np.size(test_u, 0), " is: ", np.mean(R2_test), ". Calculated in ", int(t2 - t1), " seconds.")

with open(savefn, 'a') as f:
    f.write(f'\nMean R^2 for a random sample of {test_a.shape[0]} reservoirs from the test data set of size {np.size(test_u, 0)} is: {np.mean(R2_test)}. Calculated in {int(t2 - t1)} seconds.')
