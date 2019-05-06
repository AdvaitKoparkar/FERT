import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.classifier import Classifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import math
import pdb

class AlexNet(Classifier):
    def __init__(self, dset, **kwargs):
        self.model_name = 'AlexNet'
        self.dset = dset
        self.val_dset = kwargs.get('val_dset', None)
        self.test_dset = kwargs.get('test_dset', None)
        self.in_channels = kwargs.get('in_channels', 1)
        self.num_classes = len(self.dset.classes)
        self.size = kwargs.get('size', (48,48))
        self.num_epochs = kwargs.get('epochs', 3)
        self.on_cuda = kwargs.get('cuda', False)
        self.criterion = kwargs.get('loss_fn', nn.CrossEntropyLoss)
        self.optimizer = kwargs.get('optimizer', optim.SGD)
        self.initial_lr = kwargs.get('lr', 1e-4)
        self.lr_scheduler = kwargs.get('lr_scheduler', optim.lr_scheduler.StepLR)
        self.batch_size = kwargs.get('batch_size', 4)
        self.train_writer = kwargs.get('train_writer', None)
        self.log_every = kwargs.get('log_every', 500)
        self.step = kwargs.get('step', 0)
        self.clf = _AlexNet(num_classes=self.num_classes)
        if self.on_cuda:
            self.clf = self.clf.cuda()
            
    def train(self):
        dataloader = DataLoader(self.dset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        opt = self.optimizer(self.clf.parameters(), lr=self.initial_lr, momentum=0.9)
        scheduler = self.lr_scheduler(opt, step_size=30, gamma=0.99)
        global_step = self.step
        av_loss = 0.0
        for epoch_idx in range(self.num_epochs):
            self.clf.train()
            print("Epoch %d" %epoch_idx)
            tbar = tqdm(dataloader)
            scheduler.step()
            for step_idx, data in enumerate(tbar):
                img, label = data
                opt.zero_grad()
                if self.on_cuda:
                    img = img.cuda()
                    label = label.cuda()
                op = self.clf(img)
                crt = self.criterion()
                loss = crt(op, label)
                av_loss += loss.item()
                if (global_step+1) % self.log_every == 0:
                    av_loss /= self.log_every
                    self.train_writer.add_scalar('Loss', av_loss, global_step)
                    # Validatation
                    val_dataloader = DataLoader(self.val_dset, batch_size=self.batch_size, shuffle=False, num_workers=1)
                    acc = self.score(dloader=val_dataloader)
                    self.train_writer.add_scalar('Validation Accuracy', acc, global_step)
                    av_loss = 0.0
                loss.backward()
                opt.step()
                global_step += 1
        self.step = global_step
        
        
    def test(self, X=None, y=None, dloader=None):
        self.clf.eval()
        y_pred = None
        y_true = None
        if X is not None:
            # dummy_labels = -1*np.ones((X,shape[0], 1))
            if X.ndim == 2:
                X = np.reshape(X, (1, X.shape[0], X.shape[1], 1))
            if X.ndim == 3:
                X = np.reshape(X[:, :, 0], (1, X.shape[0], X.shape[1], 1))
            if X.ndim == 4:
                X = np.transpose(X[:, :, :, 0:1], axes=[0, 3, 1, 2])
            if y is None:
                y = -1*np.ones((X.shape[0], 1))

            temp_dset = torch.utils.data.TensorDataset(torch.tensor(X).type(torch.FloatTensor), torch.tensor(y).type(torch.FloatTensor))
            dloader = DataLoader(temp_dset, batch_size=1, shuffle=False, num_workers=1)
        for img, label in tqdm(dloader):
            if self.on_cuda:
                img = img.cuda()
                label = label.cuda()
            op = self.clf(img)
            if y_pred is None:
                y_pred = torch.argmax(op, dim=1, keepdim=True).cpu().detach().numpy()
                y_true = label.cpu().detach().numpy()[:, None]
            else:
                y_pred = np.vstack((y_pred, torch.argmax(op, dim=1, keepdim=True).cpu().detach().numpy()))
                y_true = np.vstack((y_true, label.cpu().detach().numpy()[:, None]))
        self.clf.train()
        acc = np.sum(y_pred == y_true) / np.size(y_pred)
        return y_pred, acc
    
    def get_state(self):
        return self.clf.state_dict()

    def set_state(self, state_dict):
        self.clf.load_state_dict(state_dict)
        
class ConvBlock(nn.Module):
    def __init__(self, in_plane, out_plane, stride=1, k_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1=nn.Conv2d(in_plane, out_plane, kernel_size=k_size, stride=stride, padding=padding)
        
    def forward(self,x):
        out=F.relu(self.conv1(x), inplace=True)
        return out
    
class MaxPoolBlock(nn.Module):
    def __init__(self,k_size=3,stride=1):
        super(MaxPoolBlock, self).__init__()
        self.mp1=nn.MaxPool2d( kernel_size=k_size, stride=stride)
        
    def forward(self,x):
        out=self.mp1(x)
        return out
        
class _AlexNet(nn.Module):
    def __init__(self, out_planes=[64,192,384,256,256],k_size=[11,2,5,2,3,3,3,2],padding=[5,2,1,1,1], num_classes=7, size=48 ):
        super(_AlexNet, self).__init__()
#        self.name = self.__class__.__name_
        
        self.in_planes=1
        self.layer1=self._make_layer(1,self.in_planes,out_planes[0], 4,k_size[0],padding[0])
        self.layer2=self._make_layer(0,None, None,k_size[1],2, None)
        self.layer3=self._make_layer(1,out_planes[0],out_planes[1], 1,k_size[2],padding[1])
        self.layer4=self._make_layer(0,None, None,k_size[3],2, None)
        self.layer5=self._make_layer(1,out_planes[1],out_planes[2], 1,k_size[4],padding[2])
        self.layer6=self._make_layer(1,out_planes[2],out_planes[3], 1,k_size[5],padding[3])
        self.layer7=self._make_layer(1,out_planes[3],out_planes[4], 1,k_size[6],padding[4])
        self.layer8=self._make_layer(0,None, None,k_size[7],2,None)
        
        self.classifier=nn.Linear(256, num_classes)
        
    def _make_layer(self, block_type, in_planes, out_planes, stride, k_size, padding):
        
        if block_type==1:
            return ConvBlock(in_planes, out_planes, stride, k_size, padding)
        else:
            return MaxPoolBlock(k_size, stride)
        
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)     
        out=self.layer3(out) 
        out=self.layer4(out)
        out=self.layer5(out)
        out=self.layer6(out)
        out=self.layer7(out)
        out=self.layer8(out) 
        out=out.view(out.size(0), -1)
        out=self.classifier(out)
        return out 
        