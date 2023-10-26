import os

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score

# Define hyperparameters
batch_size = 16
num_epochs = 500

# Define data transforms
data_transform = transforms.Compose([
    transforms.Resize(256),
    # rotate between -10 and 10 degrees
    transforms.RandomRotation(10),
    # flip horizontally with probability 0.5
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.2])])

# Dataset class
class BinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [] 
        self.labels = []
        
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            for filename in os.listdir(label_path):
                filepath = os.path.join(label_path, filename)
                self.filenames.append(filepath)
                self.labels.append(int(label))
                
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        img_path = self.filenames[index]
        img = Image.open(img_path)
        # if 3 channel, convert to 1 channel
        if img.mode == 'RGB':
            img = img.convert('L')
        img = self.transform(img)
        label = self.labels[index]
        return img, label

# Instantiate dataset
train_dataset = BinaryDataset('data/train', transform=data_transform)  
test_dataset = BinaryDataset('data/test', transform=data_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size) 

# Define model
model = models.resnet50(pretrained=True)
# input as (1, 256, 256) instead of (3, 256, 256)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) 

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# cyclic learning rate
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=20, mode='triangular2')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# init wandb
import wandb
wandb.init(project="pytorch-resnet50")
# task name
wandb.run.name = "single resnet50"

# Train loop
auc = 0
gts = []
preds = []
for epoch in range(num_epochs):
    for images, labels in train_loader:  
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward and backward passes
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        gts.append(labels.cpu().numpy())
        preds.append(outputs.detach().cpu().numpy())
        # wandb
        wandb.log({"Epoch": epoch, "Loss": loss.item(), "Learning Rate": scheduler.get_last_lr()[0]})
        
    scheduler.step()

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    auc = roc_auc_score(gts, preds[:, 1])
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, AUC: {auc:.4f}')
    # wandb display
    wandb.log({"Epoch": epoch, "Train AUC": auc})

    # Test loop   
    auc = 0
    gts = []
    preds = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device) 
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            preds.append(outputs.detach().cpu().numpy())
            gts.append(labels.cpu().numpy())
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()
            # wandb
            wandb.log({"Epoch": epoch, "Test Accuracy": n_correct / n_samples})

        gts = np.concatenate(gts)
        preds = np.concatenate(preds)
        auc = roc_auc_score(np.concatenate(gts), np.concatenate(preds)[:, 1])
        print(f'AUC: {auc:.4f}')
        # wandb display
        wandb.log({"Epoch": epoch, "Test AUC": auc})