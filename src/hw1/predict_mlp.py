from timeit import repeat
import numpy as np
import sklearn
import csv
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import matplotlib
from models.mlp import MTDataset, myMLP


from models.linear_regressor import LinearRegressor

def read_csv(path):
    dataset = []
    with open(path, 'r') as F:
        csv_file = csv.reader(F)
        for row in csv_file:
            dataset.append(row)
    return dataset

def get_title_and_content(dataset):
    return dataset[0], dataset[1:]

# Extend to remove selected columns
def get_raw_label(content):
    return [row[-1] for row in content]

def remove_url(content):
    return [row[1:] for row in content]

def remove_label(content):
    return [row[:-1] for row in content]

def get_label(content, threshold = 1400):
    return [1 if int(row[-1]) >= threshold else 0 for row in content]

def list2np_float(alist):
    float_list = [list(map(float, row)) for row in alist]
    return np.array(float_list)

def min_max_normalizer(data, mode='train', train_max = None, train_min = None):
    '''
        input:      [R * C] numpy array
        output:     [R * C] numpy array
        mode:       set difference between trainset and testset to avoid LOOK AHEAD mistake.
    ''' 
    if mode == 'train':
        col_max = np.max(data, axis = 0).reshape((1, data.shape[1])).repeat(repeats = [data.shape[0]], axis = 0)
        col_min = np.min(data, axis = 0).reshape((1, data.shape[1])).repeat(repeats = [data.shape[0]], axis = 0)
        return col_max, col_min, (data - col_min) / (col_max - col_min)
    elif mode == 'test':
        return (data - train_min[:data.shape[0]]) / (train_max[:data.shape[0]] - train_min[:data.shape[0]])

def get_split(data, label, proportion=0.8):
    train_set = data[:int(data.shape[0] * proportion)]
    train_label = label[:int(data.shape[0] * proportion)]
    test_set = data[int(data.shape[0] * proportion):]
    test_label = label[int(data.shape[0] * proportion):]
    return train_set, train_label, test_set, test_label


if __name__ == "__main__":
    dataset = read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
    title, content = get_title_and_content(dataset)
    new_content = remove_url(content)

    print(new_content[0])
    
    raw_label   = np.array(list(map(float, get_raw_label(content))))
    label = get_label(new_content)
    new_content = remove_label(new_content)
    new_content = list2np_float(new_content)
    train_set, train_label, test_set, test_label = get_split(new_content, raw_label)
    # lr = LinearRegressor(train_set, train_label)
    # lr.solve()
    # pred = lr.predict(test_set)
    # pred = np.where(pred >= 1400, 1, 0)
    # gt   = np.where(test_label >= 1400, 1, 0)

    # print("acc: {}".format(np.sum(np.abs(pred - gt)) / pred.shape[0]))
    col_max, col_min, train_set = min_max_normalizer(train_set, mode='train')
    test_set = min_max_normalizer(test_set, mode='test', train_max=col_max, train_min=col_min)
    train_label = np.where(train_label >= 1400, 1, 0)
    test_label = np.where(test_label >= 1400, 1, 0)


    train_loader=DataLoader(dataset=MTDataset(train_set,train_label), batch_size=32, shuffle=True)
    test_loader =DataLoader(dataset=MTDataset(test_set,test_label),   batch_size=32, shuffle=False)

    ccnt = 0
    # plt.plot([i for i in range(len(train_y))], train_y, "r")
    # plt.plot([i for i in range(len(train_y), len(train_y) + len(val_y))], val_y, "g")
    # plt.plot([i for i in range(len(train_y) + len(val_y), len(train_y) + len(val_y) + len(test_y))], test_y, "b")
    # plt.show()
    # GPU
    gpu = torch.cuda.is_available()


    model = myMLP()

    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)  # ?????????
    model = model.cuda()

    for epoch in range(500):
        total_loss = 0
        tot = 0
        cor = 0
        model.train()
        for idx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.squeeze(1).cuda()
            label = label.cuda()
            pred = model(data)
            pred_label = torch.argmax(pred, dim=1).long()
            loss = criterion(pred,label.long())
            tcor = pred_label.shape[0] - torch.sum(torch.abs(pred_label - label))
            # print(cor)
            # print(pred_label)
            # print(label)
            loss.backward()
            optimizer.step()
            cor += tcor 
            tot += pred_label.shape[0]
            # if idx % 100 == 0:
            #     print('Epoch {} Iter {}: Loss {} Acc {}'.format(epoch, idx, loss, cor / tot))
        model.eval()
        tot_test = 0
        cor_test = 0
        with torch.no_grad():
            for idx, (data, label) in enumerate(test_loader):
                data = data.squeeze(1).cuda()
                label = label.cuda()
                pred = model(data)
                pred_label = torch.argmax(pred, dim=1).long()
                tcor = pred_label.shape[0] - torch.sum(torch.abs(pred_label - label))
                cor_test += tcor 
                tot_test += pred_label.shape[0]
        print("============ Epoch {} + Training Acc {} + Test Acc {} ============".format(epoch, cor / tot, cor_test / tot_test))
        print("============ Epoch {} Acc {} ============".format(epoch, cor / tot))
        # print(labe_np.shape)
            # plt.plot(preds_t,"r")
            # plt.plot(labels_t,"b")
            # plt.show()
            # plt.close('all')
    # ????????????
    # torch.save(model, opt.save_name)

        # ????????????
