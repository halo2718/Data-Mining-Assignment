import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import json 
import re
from transformers import BertModel, BertTokenizer

class AllDataset(Dataset):
    def __init__(self, mode, proportion, origin_data, origin_label, csv_data=r"C:\Users\lt\Desktop\Data-Mining-Assignment\data\OnlineNewsPopularity\OnlineNewsPopularity.csv"):
        super(AllDataset,self).__init__()

        label_pool = []
        with open(csv_data, "r") as F:
            line = F.readline()
            line = F.readline()
            while line:
                shares = int(line.split(",")[-1].strip())
                label_pool.append(1 if shares > 1400 else 0)
                line = F.readline()

        with open("./text_data.json", 'r') as F:
            json_data = json.load(F)
        if mode == 'train':
            ridx = [0, int(len(label_pool) * proportion)]
        elif mode == 'test':
            ridx = [int(len(label_pool) * proportion), len(label_pool)]

        self.file_path_list  = []
        self.label_path_list = []
        for i in range(ridx[0], ridx[1]):
            if str(i) in json_data:
                self.file_path_list.append(json_data[str(i)])
                self.label_path_list.append(label_pool[i])
            else:
                self.file_path_list.append({'title':"Null", 'author':"Null", 'content':"Null"})
                self.label_path_list.append(label_pool[i])
        
        self.origin_data = torch.Tensor(origin_data)
        self.origin_label = torch.Tensor(origin_label)
        print("["*80)
        print(len(self.origin_data))

    def tokenize(self,text):
        fileters = ['!','"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','\?','@'
            ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
        text = re.sub("<.*?>"," ",text,flags=re.S)	
        text = re.sub("|".join(fileters)," ",text,flags=re.S)	
        return text	

    def __getitem__(self, idx):
        info = self.file_path_list[idx]
        content = info['title']
        # content = self.tokenize(content) #处理文本中的奇怪符号
        origin  = self.origin_data[idx]
        label   = self.label_path_list[idx]
        return [content], [origin], [label]
        
    def __len__(self):
        return len(self.file_path_list)

class TransMLP(nn.Module):
    def __init__(self, input_dim=59):
        super(TransMLP, self).__init__()
        self.l1 = torch.nn.Linear(128, 256)
        self.l2 = torch.nn.Linear(256, 512)
        self.l3 = torch.nn.Linear(512, 256)
        self.l4 = torch.nn.Linear(256, 64)
        self.l5 = torch.nn.Linear(64, 16)
        self.l6 = torch.nn.Linear(16, 2)

        model_name = 'bert-base-uncased'
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
        # for p in self.bert.parameters(): 
        #     p.requires_grad = False
        self.proj = torch.nn.Linear(768, 69)
        
    def forward(self, x, input_ids, attention_mask):
        bert_out=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state =bert_out[0]
        bert_cls_hidden_state=last_hidden_state[:,0,:]

        bert_proj = F.relu(self.proj(bert_cls_hidden_state))

        # print()
        x = torch.concat((x, bert_proj), dim = 1)
        # print(x.shape)
        # print("===================================================")

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return self.l6(x)

