import os
import pandas as pd
import torch
import sklearn
import numpy as np
import torchvision.transforms as tvtfs
from torch.utils.data import Dataset


class FERDataset(Dataset):
    ''' FER Dataset Class '''
    def __init__(self, root="./data/fer2013", filename="fer2013.csv", mode='train', shuffle=False, size=48, classes={0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}, transforms=None, tensor=False):
        self.root = root
        self.filename = filename
        self.classes = classes
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        # Loading Dataset from csv
        self.dataframe = pd.read_csv(os.path.join(self.root, self.filename))
        if mode == 'train':
            self.dataframe = self.dataframe[self.dataframe['Usage'] == 'Training']
        elif mode == 'val':
            self.dataframe = self.dataframe[self.dataframe['Usage'] == 'PublicTest']
        elif mode == 'test':
            self.dataframe = self.dataframe[self.dataframe['Usage'] == 'PrivateTest']
        else:
            raise Exception('Invalid mode')
        # if shuffle:
        #     self.dataframe = sklearn.utils.shuffle(self.dataframe)

        if transforms is None and tensor is True:
            self.transforms = tvtfs.Compose([
                tvtfs.ToPILImage(),
                tvtfs.ToTensor(),
                tvtfs.Normalize((129.47,), (65.02,)),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        pixel_string = self.dataframe.iloc[idx]['pixels']
        img_np = np.fromstring(pixel_string, dtype=np.float32, sep=' ')
        img = np.reshape(img_np, self.size)[:,:,None]
        label = self.dataframe.iloc[idx]['emotion']
        if self.transforms is not None:
            img_transformed = self.transforms(img)
        else:
            img_transformed = img
        return (img_transformed, label)
