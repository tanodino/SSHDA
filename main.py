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
import random
from typing import Sequence
import torchcontrib
from collections import OrderedDict

def modify_weights(model, ghost_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ghost_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ghost_weights[k] + (1-alpha) * current_weights_npy[k]
    
    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )
    
    return current_weights, current_weights_npy

def retrieveModelWeights(model):
    ##### REASONING FLAT MINIMA #####
    state_dict = model.state_dict()
    to_save = OrderedDict()
    for k in state_dict:
        to_save[k] = state_dict[k].cpu().detach().numpy()
    return to_save



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



class MyRotateTransform():
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MyDataset_Unl(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]        
        x_transform = self.transform(self.data[index])
        
        return x, x_transform
    
    def __len__(self):
        return len(self.data)



class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)





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

def rescale(data):
    min_ = np.percentile(data,2)
    max_ = np.percentile(data,98)
    return np.clip( (data - min_) / (max_ - min_), 0, 1.)


def dataAugRotate(data, labels, axis):
    new_data = []
    new_label = []
    for idx, el in enumerate(data):
        for k in range(4):
            new_data.append( np.rot90(el, k, axis) )
            new_label.append( labels[idx] )
    return np.array(new_data), np.array(new_label)


dir = sys.argv[1]
source_prefix = sys.argv[2]
target_prefix = sys.argv[3]
nsamples = sys.argv[4]
nsplit = sys.argv[5]

source_data = np.load("%s/%s_data_filtered.npy"%(dir,source_prefix) )
target_data = np.load("%s/%s_data_filtered.npy"%(dir,target_prefix) )
source_label = np.load("%s/%s_label_filtered.npy"%(dir,source_prefix) )
target_label = np.load("%s/%s_label_filtered.npy"%(dir,target_prefix) )

print("data loaded")
print("source_data ",source_data.shape)
print("target_data ",target_data.shape)
print("source_label ",source_label.shape)
print("target_label ",target_label.shape)
sys.stdout.flush()
train_target_idx = np.load("%s/%s_%s_%s_train_idx.npy"%(dir, target_prefix, nsplit, nsamples))
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
#source_data, source_label = dataAugRotate(source_data, source_label, (1,2))
#train_target_data, train_target_label = dataAugRotate(train_target_data, train_target_label, (1,2))

print("AFTER DATA AUGMENTATION")
print("source_data ",source_data.shape)
print("train_target_data ",train_target_data.shape)
print("source_label ",source_label.shape)
print("train_target_label ",train_target_label.shape)
sys.stdout.flush()
n_classes = len(np.unique(source_label))


train_batch_size = 512#1024#512

source_data, source_label = shuffle(source_data, source_label)
train_target_data, train_target_label = shuffle(train_target_data, train_target_label)

#source_data = source_data[0:50000]
#source_label = source_label[0:50000]


#DATALOADER SOURCE
x_train_source = torch.tensor(source_data, dtype=torch.float32)
y_train_source = torch.tensor(source_label, dtype=torch.int64)

#dataset_source = TensorDataset(x_train_source, y_train_source)
angle = [0, 90, 180, 270]
transform_source = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5)
    ])
dataset_source = MyDataset(x_train_source, y_train_source, transform=transform_source)
dataloader_source = DataLoader(dataset_source, shuffle=True, batch_size=train_batch_size)

#DATALOADER TARGET TRAIN
x_train_target = torch.tensor(train_target_data, dtype=torch.float32)
y_train_target = torch.tensor(train_target_label, dtype=torch.int64)

transform_target = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5)
    ])

#dataset_train_target = TensorDataset(x_train_target, y_train_target)
dataset_train_target = MyDataset(x_train_target, y_train_target, transform=transform_target)
dataloader_train_target = DataLoader(dataset_train_target, shuffle=True, batch_size=train_batch_size)

#DATALOADER TARGET UNLABELLED
x_train_target_unl = torch.tensor(test_target_data_unl, dtype=torch.float32)

transform_target_unl = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5),
    T.RandomApply([T.ColorJitter()], p=0.5)
    ])

dataset_train_target_unl = MyDataset_Unl(x_train_target_unl, transform_target_unl)
dataloader_train_target_unl = DataLoader(dataset_train_target_unl, shuffle=True, batch_size=train_batch_size)

#DATALOADER TARGET TEST
x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
dataset_test_target = TensorDataset(x_test_target, y_test_target)
dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=512)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
model = resnet18(weights=None)
#model = convnext_tiny(weights=None)
model.conv1 = nn.Conv2d(train_data.shape[1], 64, kernel_size=7, stride=2, padding=3,bias=False)
model._modules["fc"]  = nn.Linear(in_features=512, out_features=len(np.unique(train_label)) )
'''
#x_opt, x_sar = x
#The forward method of the ORDisModel class takes as input a pair of optical and sar data
#more precisely, we adopt a training strategy in which we have a balanced number of optical and sar samples for each batch
model = None
opt_dim = None
sar_dim = None
'''
if source_prefix == "MS":
    opt_dim = source_data.shape[1]
    sar_dim = target_data.shape[1]
else:
    opt_dim = target_data.shape[1]
    sar_dim = source_data.shape[1]
'''

print("source_data.shape[1] ",source_data.shape[1])
print("target_data.shape[1] ",target_data.shape[1])
sys.stdout.flush()
model = ORDisModel(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
model = model.to(device)

#decay = 0.999
#ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
#ema_model = AveragedModel(model)



learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
loss_fn_noReduction = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
#optimizer = torchcontrib.optim.SWA(base_optimizer)
scl = SupervisedContrastiveLoss()


epochs = 500#300
# Loop through the data
valid_f1 = 0.0
margin = .3
decreasing_coeff = 0.95
i = 0
model_weights = []
ghost_weights = None
momentum_ema = 0.9
th_pseudo_label = .95
for epoch in range(epochs):
    start = time.time()
    model.train()
    tot_loss = 0.0
    tot_ortho_loss = 0.0
    tot_fixmatch_loss = 0.0
    den = 0
    for x_batch_source, y_batch_source in dataloader_source:
        #print("ciao")
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

        y_inv_labels = np.concatenate([y_batch_source.cpu().detach().numpy(), y_batch_target.cpu().detach().numpy()],axis=0)

        mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(inv_emb, y_inv_labels, np.zeros_like(y_inv_labels), scl, epoch)
        
        norm_inv_emb = nn.functional.normalize(inv_emb)
        norm_spec_emb = nn.functional.normalize(spec_emb)
        loss_ortho = torch.mean( torch.sum( norm_inv_emb * norm_spec_emb, dim=1) )
        loss_ortho = torch.maximum( loss_ortho - margin, torch.tensor(0) )

        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        
        ########## CONSISTENCY LOSS ##########

        #emb_unl_target, _, _, pred_unl_target = model.forward_source(x_batch_target_unl, 1)
        _, _, _, pred_unl_target = model.forward_source(x_batch_target_unl, 1)
        #emb_unl_target_aug, _, _, pred_unl_target_aug = model.forward_source(x_batch_target_unl_aug, 1)
        _, _, _, pred_unl_target_aug = model.forward_source(x_batch_target_unl_aug, 1)
        pred_unl_target = torch.softmax(pred_unl_target,dim=1).detach()
        pred_unl_target_aug = torch.softmax(pred_unl_target_aug,dim=1)
        
        loss_consistency_pred = torch.mean( torch.sum( torch.abs(pred_unl_target - pred_unl_target_aug), dim=1) )
        
        ''' FIXMATCH '''
        '''
        pseudo_labels = torch.softmax(pred_unl_target,dim=1).cpu().detach().numpy()
        max_value = np.amax(pseudo_labels, axis=1)
        ind_var = (max_value > th_pseudo_label).astype("int")
        ind_var = torch.tensor(ind_var).to(device)
        pseudo_labels_tensor = torch.tensor(np.argmax(pseudo_labels,axis=1), dtype=torch.int64).to(device)
        #print("pseudo_labels_tensor.shape ",pseudo_labels_tensor.shape)
        loss_fix_match = torch.sum( ind_var * loss_fn_noReduction( pred_unl_target_aug,  pseudo_labels_tensor ) )
        #print("loss_fix_match",loss_fix_match)
        loss_fix_match = loss_fix_match / ( torch.sum(ind_var) + torch.finfo(torch.float32).eps )
        #print("loss_fix_match",loss_fix_match)
        '''
        #norm_emb_unl_target = nn.functional.normalize(emb_unl_target)
        #norm_emb_unl_target_aug = nn.functional.normalize(emb_unl_target_aug)
        #loss_consistency_emb = torch.mean( 1 - torch.sum(norm_emb_unl_target * norm_emb_unl_target_aug, dim=1) )
        

        loss_consistency = loss_consistency_pred #+ loss_consistency_emb
        #loss_consistency = loss_fix_match #loss_consistency_pred
        ########################################
        
        loss = loss_pred + loss_dom + mixdl_loss_supContraLoss + 0.00001 * l2_reg + loss_ortho + loss_consistency
        
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        
        tot_loss+= loss.cpu().detach().numpy()
        tot_ortho_loss+=loss_ortho.cpu().detach().numpy()
        tot_fixmatch_loss = loss_consistency.cpu().detach().numpy()
        den+=1.

        #torch.cuda.empty_cache()
        
    if int(tot_ortho_loss/den * 1000) == 0:
        previous_margin = margin
        margin = margin * decreasing_coeff
        print("\T\T\T MARGIN decreasing from %f to %f"%(previous_margin,margin))

    #MANUAL IMPLEMENTAITON OF THE EMA OPERATION
    if epoch >= 50:
        current_weights, ghost_weights = modify_weights(model, ghost_weights, momentum_ema)
        model.load_state_dict(current_weights)

    end = time.time()
    pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    #ema_pred_valid, ema_labels_valid = evaluation(ema_model, dataloader_test_target, device, source_prefix)
    #f1_val_ema = f1_score(ema_labels_valid, ema_pred_valid, average="weighted")
    #print("TRAIN LOSS at Epoch %d: %.4f with acc on TEST TARGET SET %.2f with (EMA) acc on TETS TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_val,100*f1_val_ema, (end-start)))    
    print("TRAIN LOSS at Epoch %d: %.4f with ORTHO LOSS %.4f with CONSISTENCY LOSS %.4f acc on TEST TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, tot_ortho_loss/den, tot_fixmatch_loss/den, 100*f1_val, (end-start)))    
    sys.stdout.flush()
    


update_bn(dataloader_source, dataloader_train_target, model)
pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
f1_val = f1_score(labels_valid, pred_valid, average="weighted")
print("-> -> -> -> FINAL PERF AFTER BN STATISTICS UPDATE %f"%f1_val)


#path = "prova.pth"
#torch.save(swa_model.state_dict(), path)
#model.load_state_dict(torch.load(path))



#optimizer.swap_swa_sgd()
#pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
#f1_val = f1_score(labels_valid, pred_valid, average="weighted")
#print("SWA MODEL FINAL ACCURACY ON TEST TARGET SET %.2f"%(100*f1_val))    

