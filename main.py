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

def modify_weights(model, ghost_weights, alpha, epoch):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    
    if ghost_weights is None:
        ghost_weights = OrderedDict()
        for k in state_dict:
            current_weights_npy[k] = state_dict[k].cpu().detach().numpy()
    else:
        for k in state_dict:
            temp_weights = state_dict[k].cpu().detach().numpy()
            current_weights_npy[k] = alpha * ghost_weights[k] + (1-alpha) * temp_weights
    '''
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()
    '''
    
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
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class MyRotateTransform():
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


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



w_size = 128 
#resize = T.Resize((w_size, w_size), interpolation=T.InterpolationMode.BICUBIC)
#
#x_train_source = resize(x_train_source)
#print("\tAFTER RESIZING ",x_train_source.shape)

#dataset_source = TensorDataset(x_train_source, y_train_source)
angle = [0, 90, 180, 270]
transform_source = T.Compose([
    T.Resize(w_size,antialias=True), 
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
swa_model = torch.optim.swa_utils.AveragedModel(model)

#decay = 0.999
#ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
#ema_model = AveragedModel(model)



learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
#optimizer = torchcontrib.optim.SWA(base_optimizer)
scl = SupervisedContrastiveLoss()


epochs = 50#300
# Loop through the data
valid_f1 = 0.0
margin = .3
decreasing_coeff = 0.95
i = 0
model_weights = []
ghost_weights = None
momentum_ema = 0.99
for epoch in range(epochs):
    start = time.time()
    model.train()
    tot_loss = 0.0
    tot_ortho_loss = 0.0
    den = 0
    for x_batch_source, y_batch_source in dataloader_source:
        #print("ciao")
        optimizer.zero_grad()
        x_batch_target, y_batch_target = next(iter(dataloader_train_target))

        x_batch_source = x_batch_source.to(device)
        y_batch_source = y_batch_source.to(device)
        
        x_batch_target = x_batch_target.to(device)
        y_batch_target = y_batch_target.to(device)

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

        '''
        joint_embedding = torch.cat([inv_emb, emb_source_spec, emb_target_spec],dim=0)
        dom_mix_labels = np.concatenate([np.zeros(inv_emb.shape[0]), np.ones(emb_source_spec.shape[0]), np.ones(emb_target_spec.shape[0])*2],axis=0)
        
        
        #y_mix_labels = np.concatenate([y_inv_labels, np.ones(emb_source_spec.shape[0]), np.ones(emb_target_spec.shape[0]) ],axis=0)
        y_mix_labels = np.concatenate([y_inv_labels, y_batch_source.cpu().detach().numpy(), y_batch_target.cpu().detach().numpy() ],axis=0)
        
        mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)
        '''
        mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(inv_emb, y_inv_labels, np.zeros_like(y_inv_labels), scl, epoch)
        
        norm_inv_emb = nn.functional.normalize(inv_emb)
        norm_spec_emb = nn.functional.normalize(spec_emb)
        loss_ortho = torch.mean( torch.sum( norm_inv_emb * norm_spec_emb, dim=1) )
        loss_ortho = torch.maximum( loss_ortho - margin, torch.tensor(0) )

        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        #loss = loss_pred + loss_dom + mixdl_loss_supContraLoss + loss_ortho + 0.00001 * l2_reg
        loss = loss_pred + loss_dom + mixdl_loss_supContraLoss + 0.00001 * l2_reg + loss_ortho
        #loss = loss_pred + mixdl_loss_supContraLoss + loss_ortho + 0.00001 * l2_reg
        
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        
        #SWA : average at each iteration
        #if i > 10 and i % 5 == 0:
        #    optimizer.update_swa()
        #i+=1
        if epoch > 0:
            swa_model.update_parameters(model)

        tot_loss+= loss.cpu().detach().numpy()
        tot_ortho_loss+=loss_ortho.cpu().detach().numpy()
        den+=1.

    if int(tot_ortho_loss/den * 1000) == 0:
        previous_margin = margin
        margin = margin * decreasing_coeff
        print("\T\T\T MARGIN decreasing from %f to %f"%(previous_margin,margin))

    #MANUAL IMPLEMENTAITON OF THE EMA OPERATION
    current_weights, ghost_weights = modify_weights(model, ghost_weights, momentum_ema, epoch)
    model.load_state_dict(current_weights)

    end = time.time()
    pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    #ema_pred_valid, ema_labels_valid = evaluation(ema_model, dataloader_test_target, device, source_prefix)
    #f1_val_ema = f1_score(ema_labels_valid, ema_pred_valid, average="weighted")
    #print("TRAIN LOSS at Epoch %d: %.4f with acc on TEST TARGET SET %.2f with (EMA) acc on TETS TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_val,100*f1_val_ema, (end-start)))    
    print("TRAIN LOSS at Epoch %d: %.4f with ORTHO LOSS %.4f acc on TEST TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, tot_ortho_loss/den, 100*f1_val, (end-start)))    
    sys.stdout.flush()
    

#path = "prova.pth"
#torch.save(swa_model.state_dict(), path)
#model.load_state_dict(torch.load(path))



#optimizer.swap_swa_sgd()
#pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
#f1_val = f1_score(labels_valid, pred_valid, average="weighted")
#print("SWA MODEL FINAL ACCURACY ON TEST TARGET SET %.2f"%(100*f1_val))    

