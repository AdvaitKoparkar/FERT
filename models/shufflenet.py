import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.classifier import Classifier
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class ShuffleNet(Classifier):
    def __init__(self, dset, **kwargs):
        self.model_name = 'ShuffleNet'
        self.dset = dset
        self.val_dset = kwargs.get('val_dset', None)
        self.test_dset = kwargs.get('test_dset', None)
        self.in_channels = kwargs.get('in_channels', 1)
        self.num_classes = len(self.dset.classes)
        self.size = kwargs.get('size', (48,48))
        self.num_epochs = kwargs.get('epochs', 3)
        self.on_cuda = kwargs.get('cuda', True)
        self.criterion = kwargs.get('loss_fn', nn.CrossEntropyLoss)
        self.optimizer = kwargs.get('optimizer', optim.SGD)
        self.initial_lr = kwargs.get('lr', 1e-2)
        self.lr_scheduler = kwargs.get('lr_scheduler', optim.lr_scheduler.StepLR)
        self.batch_size = kwargs.get('batch_size', 4)
        self.train_writer = kwargs.get('train_writer', None)
        self.log_every = kwargs.get('log_every', 500)
        self.step = kwargs.get('step', 0)
        self.clf = _ShuffleNet(batch_size=self.batch_size, num_classes=self.num_classes)
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


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = int(out_planes/4)
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out


class _ShuffleNet(nn.Module):
    def __init__(self, out_planes=[200,400,800], num_blocks=[4,8,4], groups=2, num_classes=7, batch_size=4, size=48):
        super(_ShuffleNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
