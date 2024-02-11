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
from torch.optim.swa_utils import AveragedModel#, get_ema_multi_avg_fn
#from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import torchcontrib
from collections import OrderedDict
from functions import MyRotateTransform, MyDataset_Unl, MyDataset, cumulate_EMA, modify_weights, transform, TRAIN_BATCH_SIZE, LEARNING_RATE, MOMENTUM_EMA, EPOCHS, TH_FIXMATCH, WARM_UP_EPOCH_EMA
import functions
import os


def get_kTop(pred_s, pred_w):
    pseudo_labels = F.softmax(pred_w, dim=-1)
    softmax_pred = F.softmax(pred_s, dim=-1)
    _, targets = torch.max(pseudo_labels, dim=1)
    targets = targets.unsqueeze(-1)
    sorted_idx = torch.argsort(softmax_pred, descending=True, dim=1)
    mask = sorted_idx.eq(targets).float()
    mask = mask.sum(dim=0).cpu().detach().numpy()
    idx = np.arange(mask.shape[0])
    idx = idx[::-1]
    for i in idx:
        if mask[i] != 0:
            return i + 1


def to_onehot(labels, n_categories, device, dtype=torch.float32):
    batch_size = labels.shape[0]
    one_hot_labels = torch.ones(size=(batch_size, n_categories), dtype=dtype).to(device)
    for i, label in enumerate(labels):
        one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=0)
    return one_hot_labels

#use pred_w.detach() to compute this loss
def nl_loss(pred_s, pred_w, k, device):
    softmax_pred = F.softmax(pred_s, dim=-1)
    pseudo_label = F.softmax(pred_w, dim=-1)
    topk = torch.topk(pseudo_label, k)[1]
    mask_k_npl = to_onehot(topk, pseudo_label.shape[1], device)
    mask_k_npl = mask_k_npl.to(device)
    loss_npl = (-torch.log(1-softmax_pred+1e-10) * mask_k_npl).sum(dim=1).mean()
    return loss_npl



@torch.no_grad()
def update_bn(dataloader_source, dataloader_train_target, model):
    model.train()
    for x_batch_target, y_batch_target in dataloader_train_target:
        x_batch_source, y_batch_source = next(iter(dataloader_source))

        x_batch_source = x_batch_source.to(device)
        y_batch_source = y_batch_source.to(device)
        
        x_batch_target = x_batch_target.to(device)
        y_batch_target = y_batch_target.to(device)

        model([x_batch_source, x_batch_target])


def sim_dist_specifc_loss_spc(spec_emb, ohe_label, ohe_dom, scl, epoch):
    norm_spec_emb = nn.functional.normalize(spec_emb)
    hash_label = {}
    new_combined_label = []
    for v1, v2 in zip(ohe_label, ohe_dom):
        key = "%d_%d"%(v1,v2)
        if key not in hash_label:
            hash_label[key] = len(hash_label)
        new_combined_label.append( hash_label[key] )
    new_combined_label = torch.tensor(np.array(new_combined_label), dtype=torch.int64)
    #print(len(hash_label))
    return scl(norm_spec_emb, new_combined_label, epoch=epoch)


#evaluation(model, dataloader_test_target, device, source_prefix)
def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = None
        _, _ ,_, pred = model.forward_test_target(x_batch)
        #if source_prefix == "MS":
        #    _, _ ,_, pred = model.forward_test_sar(x_batch)
        #else:
        #    _, _ ,_, pred = model.forward_test_opt(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels


dir_ = sys.argv[1]
source_prefix = sys.argv[2]
target_prefix = sys.argv[3]
nsamples = sys.argv[4]
nsplit = sys.argv[5]

source_data = np.load("%s/%s_data_filtered.npy"%(dir_,source_prefix) )
target_data = np.load("%s/%s_data_filtered.npy"%(dir_,target_prefix) )
source_label = np.load("%s/%s_label_filtered.npy"%(dir_,source_prefix) )
target_label = np.load("%s/%s_label_filtered.npy"%(dir_,target_prefix) )

print("data loaded")
print("source_data ",source_data.shape)
print("target_data ",target_data.shape)
print("source_label ",source_label.shape)
print("target_label ",target_label.shape)
sys.stdout.flush()
train_target_idx = np.load("%s/%s_%s_%s_train_idx.npy"%(dir_, target_prefix, nsplit, nsamples))
test_target_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_target_idx)


train_target_data = target_data[train_target_idx]
train_target_label = target_label[train_target_idx]

test_target_data = target_data[test_target_idx]
test_target_label = target_label[test_target_idx]

test_target_data_unl = target_data[test_target_idx]



print("TRAININg ID SELECTED")
print("train_target_data ",train_target_data.shape)
print("train_target_label ",train_target_label.shape)
sys.stdout.flush()

n_classes = len(np.unique(source_label))

source_data, source_label = shuffle(source_data, source_label)
train_target_data, train_target_label = shuffle(train_target_data, train_target_label)


#DATALOADER SOURCE
x_train_source = torch.tensor(source_data, dtype=torch.float32)
y_train_source = torch.tensor(source_label, dtype=torch.int64)

#dataset_source = TensorDataset(x_train_source, y_train_source)
dataset_source = MyDataset(x_train_source, y_train_source, transform=transform)
dataloader_source = DataLoader(dataset_source, shuffle=True, batch_size=TRAIN_BATCH_SIZE)

#DATALOADER TARGET TRAIN
x_train_target = torch.tensor(train_target_data, dtype=torch.float32)
y_train_target = torch.tensor(train_target_label, dtype=torch.int64)


dataset_train_target = MyDataset(x_train_target, y_train_target, transform=transform)
dataloader_train_target = DataLoader(dataset_train_target, shuffle=True, batch_size=TRAIN_BATCH_SIZE)

#DATALOADER TARGET UNLABELLED
x_train_target_unl = torch.tensor(test_target_data_unl, dtype=torch.float32)

dataset_train_target_unl = MyDataset_Unl(x_train_target_unl, transform)
dataloader_train_target_unl = DataLoader(dataset_train_target_unl, shuffle=True, batch_size=TRAIN_BATCH_SIZE)

#DATALOADER TARGET TEST
x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
dataset_test_target = TensorDataset(x_test_target, y_test_target)
dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=512)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("source_data.shape[1] ",source_data.shape[1])
print("target_data.shape[1] ",target_data.shape[1])
sys.stdout.flush()
model = ORDisModel(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
model = model.to(device)


loss_fn = nn.CrossEntropyLoss()
loss_fn_noReduction = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
#scl = SupervisedContrastiveLoss()



# Loop through the data
valid_f1 = 0.0
i = 0
model_weights = []
ema_weights = None

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    history_k = []
    tot_loss = 0.0
    tot_ortho_loss = 0.0
    tot_fixmatch_loss = 0.0
    den = 0
    for x_batch_source, y_batch_source in dataloader_source:
        optimizer.zero_grad()
        x_batch_target, y_batch_target = next(iter(dataloader_train_target))

        x_batch_target_unl, x_batch_target_unl_aug = next(iter(dataloader_train_target_unl))

        x_batch_source = x_batch_source.to(device)
        y_batch_source = y_batch_source.to(device)
        
        x_batch_target = x_batch_target.to(device)
        y_batch_target = y_batch_target.to(device)

        x_batch_target_unl = x_batch_target_unl.to(device)
        x_batch_target_unl_aug = x_batch_target_unl_aug.to(device)

        emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl, emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl = model([x_batch_source, x_batch_target])
        pred_task = torch.cat([task_source_cl, task_target_cl],dim=0)
        pred_dom = torch.cat([dom_source_cl, dom_target_cl],dim=0)
        y_batch = torch.cat([y_batch_source, y_batch_target],dim=0)
        y_batch_dom = torch.cat([torch.zeros_like(y_batch_source), torch.ones_like(y_batch_target)],dim=0)
        
        loss_pred = loss_fn(pred_task, y_batch)
        loss_dom = loss_fn( pred_dom, y_batch_dom)
        
        inv_emb = torch.cat([emb_source_inv, emb_target_inv])
        spec_emb = torch.cat([emb_source_spec, emb_target_spec])

        '''
        y_inv_labels = np.concatenate([y_batch_source.cpu().detach().numpy(), y_batch_target.cpu().detach().numpy()],axis=0)
        mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(inv_emb, y_inv_labels, np.zeros_like(y_inv_labels), scl, epoch)
        '''
        norm_inv_emb = nn.functional.normalize(inv_emb)
        norm_spec_emb = nn.functional.normalize(spec_emb)
        loss_ortho = torch.sum( norm_inv_emb * norm_spec_emb, dim=1)
        loss_ortho = torch.mean(loss_ortho)
        

        ##### FIXMATCH ###############
        model.target.train()
        unl_target_inv, unl_target_spec, pred_unl_target_dom, pred_unl_target = model.forward_source(x_batch_target_unl, 1)
        unl_target_aug_inv, unl_target_aug_spec, pred_unl_target_strong_dom, pred_unl_target_strong = model.forward_source(x_batch_target_unl_aug, 1)

        with torch.no_grad():
            pseudo_labels = torch.softmax(pred_unl_target, dim=1)
            max_probs, targets_u = torch.max(pseudo_labels, dim=1)
            mask = max_probs.ge(TH_FIXMATCH).float()

        u_pred_loss = (F.cross_entropy(pred_unl_target_strong, targets_u, reduction="none") * mask).mean()

        pred_unl_dom = torch.cat([pred_unl_target_strong_dom,pred_unl_target_dom],dim=0)
        u_loss_dom = loss_fn(pred_unl_dom, torch.ones(pred_unl_dom.shape[0]).long().to(device))

        #unlabeled_loss_dom_aug = (F.cross_entropy(pred_unl_target_strong_dom, torch.ones(pred_unl_target_strong_dom.shape[0]).long().to(device), reduction="none") ).mean()
        #unlabeled_loss_dom_orig = (F.cross_entropy(pred_unl_target_dom, torch.ones(pred_unl_target_dom.shape[0]).long().to(device), reduction="none") ).mean()
        #u_loss_dom = ( unlabeled_loss_dom_aug + unlabeled_loss_dom_orig) / 2

        unl_inv = torch.cat([unl_target_inv,unl_target_aug_inv],dim=0)
        unl_spec = torch.cat([unl_target_spec,unl_target_aug_spec],dim=0)
        u_loss_ortho = torch.mean( torch.sum( unl_inv * unl_spec, dim=1) )

        '''
        norm_unl_target_inv = F.normalize(unl_target_inv)
        norm_unl_target_spec = F.normalize(unl_target_spec)
        norm_unl_target_aug_inv = F.normalize(unl_target_aug_inv)
        norm_unl_target_aug_spec = F.normalize(unl_target_aug_spec)
        unlabeled_loss_ortho_orig = torch.mean( torch.sum( norm_unl_target_inv * norm_unl_target_spec, dim=1) )
        unlabeled_loss_ortho_aug = torch.mean( torch.sum( norm_unl_target_aug_inv * norm_unl_target_aug_spec, dim=1) )
        u_loss_ortho = (unlabeled_loss_ortho_orig + unlabeled_loss_ortho_aug) / 2
        '''
        ##### FIXMATCH ###############
        
        ###### NEGATIVE LOSS ######
        k = get_kTop(pred_unl_target_strong.detach(), pred_unl_target.detach())        
        history_k.append(k)
        if k == n_classes:
            neg_learn_loss = torch.tensor(0)
        else:
            neg_learn_loss = nl_loss(pred_unl_target_strong, pred_unl_target.detach(), k , device)
        ###########################
        
        loss = loss_pred + loss_dom + loss_ortho + u_pred_loss + u_loss_dom + u_loss_ortho + neg_learn_loss
        
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        
        tot_loss+= loss.cpu().detach().numpy()
        tot_ortho_loss+=loss_ortho.cpu().detach().numpy()
        tot_fixmatch_loss = u_pred_loss
        den+=1.

        #torch.cuda.empty_cache()

    end = time.time()
    pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    
    ####################### EMA #####################################
    f1_val_ema = 0
    if epoch >= WARM_UP_EPOCH_EMA:
        ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)
        current_state_dict = model.state_dict()
        model.load_state_dict(ema_weights)
        pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
        f1_val_ema = f1_score(labels_valid, pred_valid, average="weighted")
        model.load_state_dict(current_state_dict)
    ####################### EMA #####################################
    
    print("TRAIN LOSS at Epoch %d: WITH TOTAL LOSS %.4f acc on TEST TARGET SET (ORIG) %.2f (EMA) %.2f with train time %d"%(epoch, tot_loss/den, 100*f1_val, 100*f1_val_ema, (end-start)))    
    print("history K", np.bincount(history_k))
    sys.stdout.flush()

dir_name = dir_+"/OUR"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

output_file = dir_name+"/%s_%s_%s.pth"%(target_prefix, nsplit, nsamples)
model.load_state_dict(ema_weights)
torch.save(model.state_dict(), output_file)




