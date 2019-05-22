from __future__ import print_function, division
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
from torch.utils.data import DataLoader
import csv

class MyDataset(Dataset):
    def __init__(self, path):
        self.filenames = []

        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                words = line.split(",")
                filename = words[0].strip()
               # filename = filename[1:]
                filename = os.path.join('./data/public_test_data/', filename)

                self.filenames.append(filename)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path = self.filenames[index]
        img = cv2.imread(img_path)  # BGR
        img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
        assert img is not None, 'File Not Found ' + img_path
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return torch.from_numpy(img), img_path


def test(model):
    label = []
    imgpath = []
    tags = {"0": 0.1,
            "1": 0.2,
            "2": 0.5,
            "3": 1,
            "4": 2,
            "5": 5,
            "6": 10,
            "7": 50,
            "8": 100}
    model.eval()
    for i, (inputs, path) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        value, preds = torch.max(outputs, 1)
        preds = tags[str(preds.item())]
        if value <0.5:
            print(path)
            print(value)
            imgpath.append(path)

        label.append(preds)
    return  label


dataset = MyDataset('test1.csv')
dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=0,
                            shuffle=False,
                            pin_memory=True,
                            )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 9)
weights = './wights/'
latest = weights + 'latest.pt'
checkpoint = torch.load(latest)
model_ft.load_state_dict(checkpoint['model_state_dict'])


#model_ft.load_state_dict(latest)
model_ft = model_ft.to(device)
labels = test(model_ft)
with open('test.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['name','label'])
    i = 0
    with open('test1.csv', 'r', newline='') as f1:
        for each_line in f1:
            writer.writerow([each_line[:12], labels[i]])
            i = i + 1
