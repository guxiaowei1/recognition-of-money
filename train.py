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

class MyDataset(Dataset):
    def __init__(self, path):
        self.filenames = []
        self.labels = []
        tags = {"0.1": 0,
                "0.2": 1,
                "0.5": 2,
                "1": 3,
                "2": 4,
                "5": 5,
                "10": 6,
                "50": 7,
                "100": 8}
        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                words = line.split(",")
                filename = words[0].strip()
                filename = os.path.join('./data/train_data/', filename)
                t = words[1].strip()
                label = tags[t]
                self.filenames.append(filename)
                self.labels.append(label)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path = self.filenames[index]
      #  label = np.zeros(9)
      #  y = self.labels[index]
        label=self.labels[index]
        img = cv2.imread(img_path)  # BGR
        img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
        assert img is not None, 'File Not Found ' + img_path
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return torch.from_numpy(img), label, img_path


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
           # for inputs, labels in enumerate(dataloader):
            for i, (inputs, labels, _) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i % 50 == 0:
                    print('epoch: {}, loss: {:.4}'.format(i, loss.data.item()))

            epoch_loss = running_loss / 39620
            epoch_acc = running_corrects.double() / 39620

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if  epoch_acc > best_acc:
                best_acc = epoch_acc


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    torch.save(model.state_dict(), latest)
    return model

dataset = MyDataset('train_face_value_label.csv')
dataloader = DataLoader(dataset,
                            batch_size=16,
                            num_workers=0,
                            shuffle=True,
                            pin_memory=True,
                            )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 9)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()


# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                      num_epochs=25)


