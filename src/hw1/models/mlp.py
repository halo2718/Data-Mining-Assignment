import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class MTDataset(Dataset):
    def __init__(self, data, label):  
        self.x = torch.Tensor(data)
        self.y = torch.Tensor(label)
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y
    def __len__(self):
        return self.x.shape[0]

class myMLP(nn.Module):
    def __init__(self, input_dim=59):
        super(myMLP, self).__init__()
        self.l1 = torch.nn.Linear(59, 256)
        self.l2 = torch.nn.Linear(256, 512)
        self.l3 = torch.nn.Linear(512, 256)
        self.l4 = torch.nn.Linear(256, 64)
        self.l5 = torch.nn.Linear(64, 16)
        self.l6 = torch.nn.Linear(16, 2)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)

