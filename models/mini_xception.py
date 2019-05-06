from models.classifier import Classifier
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.classifier import Classifier
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class MiniXception(Classifier):
    def __init__(self, dset, **kwargs):
        self.model_name = 'MiniXception'
        self.dset = dset
        self.val_dset = kwargs.get('val_dset', None)
        self.test_dset = kwargs.get('test_dset', None)
        self.in_channels = kwargs.get('in_channels', 1)
        self.num_classes = len(self.dset.classes)
        self.size = kwargs.get('size', (48,48))
        self.num_epochs = kwargs.get('epochs', 1)
        self.on_cuda = kwargs.get('cuda', True)
        self.criterion = kwargs.get('loss_fn', nn.CrossEntropyLoss)
        self.optimizer = kwargs.get('optimizer', optim.SGD)
        self.initial_lr = kwargs.get('lr', 1e-2)
        self.lr_scheduler = kwargs.get('lr_scheduler', optim.lr_scheduler.StepLR)
        self.batch_size = kwargs.get('batch_size', 4)
        self.train_writer = kwargs.get('train_writer', None)
        self.log_every = kwargs.get('log_every', 500)
        self.step = kwargs.get('step', 0)
        self.clf = _MiniXception(num_classes=self.num_classes)
        if self.on_cuda:
            self.clf = self.clf.cuda()

    def train(self):
        dataloader = DataLoader(self.dset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        opt = self.optimizer(self.clf.parameters(), lr=self.initial_lr, momentum=0.9)
        scheduler = self.lr_scheduler(opt, step_size=30, gamma=0.1)
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

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class StartBlock(nn.Module):
    def __init__(self, in_planes = 1, out_planes = 8, kernel_size = (3,3), stride=(1,1), bias = False):
        super(StartBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size = (3,3), stride=stride, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        in_planes = out_planes
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size = (3,3), stride=stride, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MiddleBlock(nn.Module):
    def __init__(self, in_planes = 16, out_planes = 16, kernel_size = (3,3), stride=(2,2), bias = False):
        super(MiddleBlock, self).__init__()
        self.residual = nn.Conv2d(in_planes, out_planes, kernel_size = (1,1), stride=stride, padding = 1, bias=bias)
        in_planes = out_planes
        self.conv1 = SeparableConv2d(in_planes, out_planes, kernel_size = kernel_size, padding = 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = SeparableConv2d(in_planes, out_planes, kernel_size = kernel_size, padding = 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.maxPool = nn.MaxPool2d(kernel_size = (3, 3), stride=stride, padding=1)

    def forward(self, x):
        out = self.bn1(self.residual(x))
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.maxPool(self.bn2(self.conv2(out)))
        return out

class EndBlock(nn.Module):
    def __init__(self, in_planes = 128, out_planes = 128, kernel_size = (3,3), stride=(2,2), bias = False, num_classes = 7):
        super(EndBlock, self).__init__()
        self.residual = nn.Conv2d(in_planes, out_planes, kernel_size = (1,1), stride=stride, padding = 1, bias=bias)
        in_planes = out_planes
        self.conv1 = SeparableConv2d(in_planes, out_planes, kernel_size = (3,3), padding = 1, bias = bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = SeparableConv2d(in_planes, out_planes, kernel_size = kernel_size, padding = 1, bias = bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.maxPool = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
        self.conv3 = nn.Conv2d(out_planes, num_classes, kernel_size = kernel_size, padding = 1)
        # self.globalLayer = nn.Flatten() #GlobalAveragePooling2D
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.bn1(self.residual(x))
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.maxPool(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        # out = self.globalLayer(out)
        out = out.view(out.size()[0], -1) #Flattened tensor
        out = self.softmax(out)
        return out

class _MiniXception(nn.Module):
    def __init__(self, in_planes = 1, out_planes = [8,16,32,64,128], num_classes=7, size=48):
        super(_MiniXception, self).__init__()

        self.in_planes = 1
        self.layer1 = self._make_layer(-1, self.in_planes, out_planes[0])
        self.layer2 = self._make_layer(0, out_planes[0], out_planes[1])
        self.layer3 = self._make_layer(0, out_planes[1], out_planes[2])
        self.layer4 = self._make_layer(0, out_planes[2], out_planes[3])
        self.layer5 = self._make_layer(1, out_planes[3], out_planes[4])

    def _make_layer(self, block_type, in_channels, out_channels):

        if block_type == -1:
            return StartBlock(in_planes = in_channels, out_planes = out_channels)
        elif block_type == 1:
            return EndBlock(in_planes = in_channels, out_planes = out_channels)
        elif block_type ==0:
            return MiddleBlock(in_planes = in_channels, out_planes = out_channels)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
