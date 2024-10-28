import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class IrisDataset(Dataset):
    def __init__(self, file_path='../data/iris.data', header=None, transform=None, 
                 target_transform=None ):

        df=pd.read_csv(file_path, header = header)
       
        self.X=torch.tensor(df.iloc[:,:-1].values)
        label_ids = [['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'].index(x) for x in df.iloc[:,-1].values] # map string label to int
        self.y=torch.tensor(label_ids)

        self.transform = transform
        self.target_transform = target_transform

        # print(self.X)
        # print(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        """
        x = self.X[idx]
        label = self.y[idx]
        # print(idx, x, label)
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        return x, label

    def __desc__(self):
        '''
        额外增加一个方法，返回数据集的说明
        '''
        with open('../data/iris.names') as f:
            lines = f.readlines()
            desc = ''
            for l in lines:
                desc+=l
            return desc