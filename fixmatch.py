### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
from model_pytorch import ORDisModel, SupervisedContrastiveLoss
import time
from sklearn.metrics import f1_score
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
#from torchvision.models import convnext_tiny
from functions import MyRotateTransform, MyDataset_Unl, MyDataset
import torchvision.transforms as T 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from functions import cumulate_EMA, transform, TRAIN_BATCH_SIZE, LEARNING_RATE, MOMENTUM_EMA, EPOCHS, TH_FIXMATCH, WARM_UP_EPOCH_EMA
#import functions
import os


def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

def dataAugRotate(data, labels, axis):
    new_data = []
    new_label = []
    for idx, el in enumerate(data):
        for k in range(4):
            new_data.append( np.rot90(el, k, axis) )
            new_label.append( labels[idx] )
    return np.array(new_data), np.array(new_label)


dir_ = sys.argv[1]
target_prefix = sys.argv[2]
nsamples = sys.argv[3]
nsplit = sys.argv[4]


data = np.load("%s/%s_data_filtered.npy"%(dir_, target_prefix))
label = np.load("%s/%s_label_filtered.npy"%(dir_, target_prefix))

train_idx = np.load("%s/%s_%s_%s_train_idx.npy"%(dir_, target_prefix, nsplit, nsamples))
test_idx = np.setdiff1d(np.arange(data.shape[0]), train_idx)



test_data = data[test_idx]
test_label = label[test_idx]

unl_data = data[test_idx]

train_data = data[train_idx]
train_label = label[train_idx]



n_classes = len(np.unique(train_label))
#train_batch_size = 512#1024#512

train_data, train_label = shuffle(train_data, train_label)

#DATALOADER TRAIN
x_train_source = torch.tensor(train_data, dtype=torch.float32)
y_train_source = torch.tensor(train_label, dtype=torch.int64)

#dataset_source = TensorDataset(x_train_source, y_train_source)
'''
angle = [0, 90, 180, 270]
transform_source = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5)
    ])
'''
dataset_source = MyDataset(x_train_source, y_train_source, transform=transform)
dataloader_train = DataLoader(dataset_source, shuffle=True, batch_size=TRAIN_BATCH_SIZE)


#DATALOADER TARGET UNLABELLED
x_train_target_unl = torch.tensor(unl_data, dtype=torch.float32)

'''
transform_target_unl = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5),
    T.RandomApply([T.ColorJitter()], p=0.5)
    ])
'''
dataset_train_target_unl = MyDataset_Unl(x_train_target_unl, transform)
dataloader_train_unl = DataLoader(dataset_train_target_unl, shuffle=True, batch_size=TRAIN_BATCH_SIZE)


#DATALOADER TEST
x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_label, dtype=torch.int64)
dataset_test = TensorDataset(x_test, y_test)
dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=TRAIN_BATCH_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = resnet18(weights=None)
model.conv1 = nn.Conv2d(train_data.shape[1], 64, kernel_size=7, stride=2, padding=3,bias=False)
model._modules["fc"]  = nn.Linear(in_features=512, out_features=n_classes )

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)


ema_weights = None
# Loop through the data
valid_f1 = 0.0

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    tot_loss = 0.0
    den = 0
    for x_batch, y_batch in dataloader_train:
        optimizer.zero_grad()
        x_batch_unl, x_batch_unl_aug = next(iter(dataloader_train_unl))
        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        x_batch_unl = x_batch_unl.to(device)
        x_batch_unl_aug = x_batch_unl_aug.to(device)

        pred = model(x_batch)        
        loss_pred = loss_fn(pred, y_batch)
        pred_u_weak = model(x_batch_unl)
        prede_u_strong = model(x_batch_unl_aug)

        #l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        with torch.no_grad():
            pseudo_labels = torch.softmax(pred_u_weak, dim=1)
            max_probs, targets_u = torch.max(pseudo_labels, dim=1)
            mask = max_probs.ge(TH_FIXMATCH).float()

        unlabeled_loss = (F.cross_entropy(prede_u_strong, targets_u, reduction="none") * mask).mean()

        loss = loss_pred + unlabeled_loss
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.

    end = time.time()
    pred_valid, labels_valid = evaluation(model, dataloader_test, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")

    ####################### EMA #####################################
    f1_val_ema = 0
    if epoch >= WARM_UP_EPOCH_EMA:
        ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)
        current_state_dict = model.state_dict()
        model.load_state_dict(ema_weights)
        pred_valid, labels_valid = evaluation(model, dataloader_test, device)
        f1_val_ema = f1_score(labels_valid, pred_valid, average="weighted")
        model.load_state_dict(current_state_dict)
    ####################### EMA #####################################

    print("TRAIN LOSS at Epoch %d: %.4f with acc on TEST TARGET SET (ORIG) %.2f (EMA) %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_val,100*f1_val_ema, (end-start)))    
    sys.stdout.flush()

dir_name = dir_+"/FIXMATCH"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

output_file = dir_name+"/%s_%s_%s.pth"%(target_prefix, nsplit, nsamples)
model.load_state_dict(ema_weights)
torch.save(model.state_dict(), output_file)