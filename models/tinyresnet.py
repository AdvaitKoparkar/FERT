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

class TinyResNet(Classifier):
    def __init__(self, dset, **kwargs):
        self.model_name = 'TinyResNet'
        self.dset = dset
        self.val_dset = kwargs.get('val_dset', None)
        self.test_dset = kwargs.get('test_dset', None)
        self.in_channels = kwargs.get('in_channels', 1)
        self.num_classes = len(self.dset.classes)
        self.size = kwargs.get('size', (48,48))
        self.num_epochs = kwargs.get('epochs', 30)
        self.on_cuda = kwargs.get('cuda', True)
        self.criterion = kwargs.get('loss_fn', nn.CrossEntropyLoss)
        self.optimizer = kwargs.get('optimizer', optim.SGD)
        self.initial_lr = kwargs.get('lr', 1e-3)
        self.lr_scheduler = kwargs.get('lr_scheduler', optim.lr_scheduler.StepLR)
        self.batch_size = kwargs.get('batch_size', 4)
        self.train_writer = kwargs.get('train_writer', None)
        self.log_every = kwargs.get('log_every', 500)
        self.step = kwargs.get('step', 0)
        self.clf = _TinyResNet(num_classes=self.num_classes)
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


    def test(self, X=None, y=None, dloader=None, get_embeddings=False):
        self.clf.eval()
        y_pred = None
        y_true = None
        embeddings = None
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
            if embeddings is None and get_embeddings is True:
                embeddings = op.cpu().detach().numpy()
            elif get_embeddings is True:
                embeddings = np.vstack((embeddings, op.cpu().detach().numpy()))
        self.clf.train()
        # pdb.set_trace()
        acc = np.sum(y_pred == y_true) / np.size(y_pred)
        if get_embeddings is False:
            return y_pred, acc
        else:
            return embeddings

    def get_state(self):
        return self.clf.state_dict()

    def set_state(self, state_dict):
        self.clf.load_state_dict(state_dict)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class _TinyResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[3,4,6,3], num_classes=7, size=48):
        super(_TinyResNet, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*int(size/8)*int(size/8), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out
