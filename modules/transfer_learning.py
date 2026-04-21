import os
import sys
import random
import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import warnings
warnings.filterwarnings('ignore')

phases = ['training', 'validation']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TL model for feature extraction
class ConvNet(nn.Module):
    def __init__(self, model, num_classes):
        super(ConvNet, self).__init__()
        self.base_model = nn.Sequential(*list(model.children())[:-1])
        self.linear1 = nn.Linear(in_features=2048, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=num_classes)
        self.relu = nn.LeakyReLU()
  
    # For Grad-CAM (GIỮ NGUYÊN GỐC)
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.base_model[:-1](x)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        lin = self.relu(x)
        x = self.linear2(lin)
        return lin, x

# SỬA: Thêm tham số save_path để Thái lưu model
def train_model(model, criterion, optimizer, scheduler, data_loader, batch_size, num_epochs=30, save_path="best_model.pth"):

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0

    for epoch in range(num_epochs):
        for phase in phases:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            epoch_corrects = 0

            for ii, (images, labels) in enumerate(data_loader[phase]):
                images = images.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'training'):
                    _, outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                epoch_corrects += torch.sum(preds == labels.data)
                epoch_loss += loss.item() * images.size(0)

            # Tính toán Accuracy
            epoch_accuracy = epoch_corrects.double() / len(data_loader[phase].dataset)
            epoch_loss /= len(data_loader[phase].dataset)

            if phase == 'training':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_accuracy.item())
                scheduler.step()
            if phase == 'validation':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_accuracy.item())

            print(f'Epoch: [{epoch+1}/{num_epochs}] Phase: {phase} | Loss: {epoch_loss:.6f} Accuracy: {epoch_accuracy:.6f}')

            # LƯU MODEL TỐT NHẤT (Chỗ Thái cần đây)
            if phase == 'validation' and epoch_accuracy >= best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                # Xuất file .pth ngay khi tìm thấy model tốt hơn
                torch.save(model.state_dict(), save_path)
                print(f'====> Best accuracy reached so far. Saved model to: {save_path}')

        print('-------------------------------------------------------------------------')

    print(f'Best Validation Accuracy: {best_accuracy:4f}')
    model.load_state_dict(best_model_wts)

    history = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }

    return model, history

# SỬA: Thêm num_classes để không lỗi khi chạy DermaMNIST (7 lớp)
def eval_model_extract_features(features, true_labels, model, dataloader, phase, num_classes=2):

    conf_mat = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        model.eval()

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            true_labels.append(labels.cpu())
            ftrs, outputs = model(images)
            features.append(ftrs.cpu())

            _, preds = torch.max(outputs, 1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_mat[t.long()][p.long()] += 1

            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()

        accuracy = n_correct/float(n_samples)
        print(f'Accuracy of model on {phase} set = {(100.0 * accuracy):.4f} %')

    print("Confusion Matrix:")
    print(conf_mat.numpy())
    return features, true_labels

def get_features(features, true_labels):
    # Gộp list tensors thành numpy chuẩn
    ftrs = torch.cat(features, dim=0).numpy()
    lbls = torch.cat(true_labels, dim=0).numpy()
    return ftrs, lbls