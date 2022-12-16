import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import json 
import re

class CnnDataset(Dataset):
    def __init__(self, mode, proportion, origin_data, origin_label, csv_data=r"C:\Users\lt\Desktop\Data-Mining-Assignment\data\OnlineNewsPopularity\OnlineNewsPopularity.csv"):
        super(CnnDataset,self).__init__()

        label_pool = []
        with open(csv_data, "r") as F:
            line = F.readline()
            line = F.readline()
            while line:
                shares = int(line.split(",")[-1].strip())
                label_pool.append(1 if shares > 1400 else 0)
                line = F.readline()

        with open("./corpus.json", "r") as F:
                json_data = json.load(F)
        if mode == 'train':
            self.file_path_list  = json_data[0: int(len(label_pool) * proportion)]
            self.label_path_list = label_pool[0: int(len(label_pool) * proportion)]
        elif mode == 'test':
            self.file_path_list  = json_data[int(len(label_pool) * proportion): len(label_pool)]
            self.label_path_list = label_pool[int(len(label_pool) * proportion): len(label_pool)]
        
        self.origin_data = torch.Tensor(origin_data)
        self.origin_label = torch.Tensor(origin_label)

        print(self.origin_label.shape[0])
        print(len(self.file_path_list))
        print(len(self.label_path_list))

        with open("embed.json", "r", encoding='utf8') as F:
            self.glove_data = json.load(F)

    def __getitem__(self, idx):
        info = self.file_path_list[idx]
        content = []
        for word in info:
            if word in self.glove_data:
                content.append(self.glove_data[word])
            else:
                content.append([0 for p in range(300)])
        origin  = self.origin_data[idx]
        label   = self.label_path_list[idx]
        return torch.Tensor(content), origin, label
        
    def __len__(self):
        return len(self.file_path_list)

class CNNMLP(nn.Module):
    def __init__(self, input_dim=59):
        super(CNNMLP, self).__init__()
        self.l1 = torch.nn.Linear(128, 256)
        self.l2 = torch.nn.Linear(256, 512)
        self.l3 = torch.nn.Linear(512, 256)
        self.l4 = torch.nn.Linear(256, 64)
        self.l5 = torch.nn.Linear(64, 16)
        self.l6 = torch.nn.Linear(16, 2)

        self.convs = nn.ModuleList(nn.Conv2d(1, 256, (k, 300)) for k in (2,3,4))
        self.dropout = nn.Dropout(0.2)
        self.proj = nn.Linear(256*3, 69)

    def conv_and_pool(self, x, conv):
        x = nn.functional.relu(conv(x)).squeeze(3)
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x    

    def forward(self, x, emb):
        emb = emb.unsqueeze(1)
        result = torch.cat([self.conv_and_pool(emb, conv) for conv in self.convs],1)
        proj = self.proj(result)

        x = torch.concat((x, proj), dim = 1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)

