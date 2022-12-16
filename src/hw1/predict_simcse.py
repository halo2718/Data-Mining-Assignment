# _*_ coding:utf-8 _*_

import torch
from torch.utils.data import DataLoader,Dataset
import os
import re
from random import sample
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import json
 
class BertClassificationModel(nn.Module):
    def __init__(self,hidden_size=768):
        super(BertClassificationModel, self).__init__()
        model_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        # for p in self.bert.parameters(): 
        #         p.requires_grad = False
        self.fc = nn.Linear(hidden_size,2)

    def forward(self, inputs):   
        # print(inputs)
        embeddings = self.bert(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        fc_out=self.fc(embeddings)
        return fc_out

class FooDataset(Dataset):
    def __init__(self, src, mode, csv_data=r"C:\Users\lt\Desktop\Data-Mining-Assignment\data\OnlineNewsPopularity\OnlineNewsPopularity.csv"):
        super(FooDataset,self).__init__()

        label_pool = []
        with open(csv_data, "r") as F:
            line = F.readline()
            line = F.readline()
            while line:
                shares = int(line.split(",")[-1].strip())
                label_pool.append(1 if shares > 1400 else 0)
                line = F.readline()
                
        sub_dir = os.listdir(src)
        if mode == "train":
            self.file_path_list  = []
            self.label_path_list = []
            for i in range(9):
                # cnt = 0
                with open(os.path.join(src, str(i), "data.json"), 'r') as F:
                    json_data = json.load(F)
                    for j in range(4000):
                        if str(j+i*4000) in json_data:
                            self.file_path_list.append(json_data[str(j+i*4000)])
                            self.label_path_list.append(label_pool[j+i*4000])
                        else:
                            self.file_path_list.append({'title':"Null", 'author':"Null", 'content':"Null"})
                            self.label_path_list.append(label_pool[j+i*4000])
                        # if label_pool[j+i*4000] == 1:
                        #     cnt+=1
                # print(cnt)
        elif mode == "test":
            self.file_path_list  = []
            self.label_path_list = []
            for i in range(9,10):
                cnt = 0
                with open(os.path.join(src, str(i), "data.json"), 'r') as F:
                    json_data = json.load(F)
                    for j in range(4000):
                        if j + i*4000 > 39643:
                            break
                        else:
                            pass
                        if str(j+i*4000) in json_data:
                            cnt += 1
                            self.file_path_list.append(json_data[str(j+i*4000)])
                            self.label_path_list.append(label_pool[j+i*4000])
                        else:
                            self.file_path_list.append({'title':"Null", 'author':"Null", 'content':"Null"})
                            self.label_path_list.append(label_pool[j+i*4000])
                # print("test cnt {}".format(cnt))
        # print(self.file_path_list[-5:])
        # print(self.label_path_list[-5:])

    def tokenize(self,text):
        fileters = ['!','"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','\?','@'
            ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
        text = re.sub("<.*?>"," ",text,flags=re.S)	
        text = re.sub("|".join(fileters)," ",text,flags=re.S)	
        return text	

    def __getitem__(self, idx):
        info = self.file_path_list[idx]
        content = info['content']
        # content = self.tokenize(content) #处理文本中的奇怪符号
        label   = self.label_path_list[idx]
        return [content], [label]
        
    def __len__(self):
        return len(self.file_path_list)

def main():
    device = torch.device("cuda")
    batchsize = 16
    trainDatas = FooDataset(src="./crawl", mode="train") 
    validDatas = FooDataset(src="./crawl", mode="test") 

    train_loader = torch.utils.data.DataLoader(trainDatas, batch_size=batchsize, shuffle=False)#遍历train_dataloader 每次返回batch_size条数据
    val_loader = torch.utils.data.DataLoader(validDatas, batch_size=batchsize, shuffle=False)
    epoch_num = 400  

    print('training...(约1 hour(CPU))')
    
    model=BertClassificationModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # 首先定义优化器，这里用的AdamW，lr是学习率，因为bert用的就是这个
    criterion = nn.CrossEntropyLoss()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='princeton-nlp/sup-simcse-bert-base-uncased')
    
    print("模型数据已经加载完成,现在开始模型训练。")
    for epoch in range(epoch_num):
        model.train()
        for i, (data,labels) in enumerate(train_loader, 0):
            inputs = tokenizer(data[0], padding=True, truncation=True, return_tensors="pt").to(device)
            cur_label = labels[0].to(device)
            output = model(inputs)
            optimizer.zero_grad()  
            loss = criterion(output, cur_label)  
            loss.backward()  
            optimizer.step() 
            if i % 50 == 0:
                print('batch:%d loss:%.5f' % (i, loss.item()))
        print('epoch:%d loss:%.5f' % (epoch, loss.item()))
        model.eval()
        with torch.no_grad():
            num = 0 
            for j, (data,labels) in enumerate(val_loader, 0):
                inputs = tokenizer(data[0], padding=True, truncation=True, return_tensors="pt").to(device)
                cur_label = labels[0].to(device)
                output = model(inputs)
                out = output.argmax(dim=1)
                # print(out)
                num += (out == cur_label).sum().item()
            print('Accuracy:', num / validDatas.__len__())

if __name__ == '__main__':
    main()