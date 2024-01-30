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
def evaluation(model, dataloader, device, source_prefix):
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

train_target_idx = np.load("%s/%s_%s_%s_train_idx.npy"%(dir, target_prefix, nsplit, nsamples))
test_target_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_target_idx)

train_target_data = target_data[train_target_idx]
train_target_label = target_label[train_target_idx]

test_target_data = target_data[test_target_idx]
test_target_label = target_label[test_target_idx]

print("TRAININg ID SELECTED")
print("train_target_data ",train_target_data.shape)
print("train_target_label ",train_target_label.shape)

source_data, source_label = dataAugRotate(source_data, source_label, (1,2))
train_target_data, train_target_label = dataAugRotate(train_target_data, train_target_label, (1,2))

print("AFTER DATA AUGMENTATION")
print("source_data ",source_data.shape)
print("train_target_data ",train_target_data.shape)
print("source_label ",source_label.shape)
print("train_target_label ",train_target_label.shape)

n_classes = len(np.unique(source_label))


train_batch_size = 512#1024#512

source_data, source_label = shuffle(source_data, source_label)
train_target_data, train_target_label = shuffle(train_target_data, train_target_label)

#source_data = source_data[0:50000]
#source_label = source_label[0:50000]


#DATALOADER SOURCE
x_train_source = torch.tensor(source_data, dtype=torch.float32)
y_train_source = torch.tensor(source_label, dtype=torch.int64)

import torchvision.transforms as T 

#w_size = 128 
#resize = T.Resize((w_size, w_size), interpolation=T.InterpolationMode.BICUBIC)
print("\tBEFORE RESIZING ",x_train_source.shape)
#x_train_source = resize(x_train_source)
print("\tAFTER RESIZING ",x_train_source.shape)

dataset_source = TensorDataset(x_train_source, y_train_source)
dataloader_source = DataLoader(dataset_source, shuffle=True, batch_size=train_batch_size)

#DATALOADER TARGET TRAIN
x_train_target = torch.tensor(train_target_data, dtype=torch.float32)
y_train_target = torch.tensor(train_target_label, dtype=torch.int64)
dataset_train_target = TensorDataset(x_train_target, y_train_target)
dataloader_train_target = DataLoader(dataset_train_target, shuffle=True, batch_size=train_batch_size)

#DATALOADER TARGET TEST
x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
dataset_test_target = TensorDataset(x_test_target, y_test_target)
dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=128)

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

model = ORDisModel(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
model = model.to(device)
#decay = 0.999
#ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
#ema_model = AveragedModel(model)



learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
scl = SupervisedContrastiveLoss()


epochs = 300
# Loop through the data
valid_f1 = 0.0
for epoch in range(epochs):
    start = time.time()
    model.train()
    tot_loss = 0.0
    den = 0
    for x_batch_source, y_batch_source in dataloader_source:
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
        joint_embedding = torch.cat([inv_emb, emb_opt_spec, emb_sar_spec],dim=0)
        dom_mix_labels = np.concatenate([np.zeros(inv_emb.shape[0]), np.ones(emb_opt_spec.shape[0]), np.ones(emb_sar_spec.shape[0])*2],axis=0)
        
        
        #y_mix_labels = np.concatenate([y_inv_labels, np.ones(emb_opt_spec.shape[0]), np.ones(emb_sar_spec.shape[0]) ],axis=0)
        y_mix_labels = np.concatenate([y_inv_labels, y_batch_opt.cpu().detach().numpy(), y_batch_sar.cpu().detach().numpy() ],axis=0)
        
        mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)
        '''
        mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(inv_emb, y_inv_labels, np.zeros_like(y_inv_labels), scl, epoch)
        norm_inv_emb = nn.functional.normalize(inv_emb)
        norm_spec_emb = nn.functional.normalize(spec_emb)
        loss_ortho = torch.mean( torch.sum( norm_inv_emb * norm_spec_emb, dim=1) )


        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss_pred + loss_dom + mixdl_loss_supContraLoss + loss_ortho + 0.00001 * l2_reg
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        #EMA model update
        #ema_model.update_parameters(model)
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.

    end = time.time()
    pred_valid, labels_valid = evaluation(model, dataloader_test_target, device, source_prefix)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    #ema_pred_valid, ema_labels_valid = evaluation(ema_model, dataloader_test_target, device, source_prefix)
    #f1_val_ema = f1_score(ema_labels_valid, ema_pred_valid, average="weighted")
    #print("TRAIN LOSS at Epoch %d: %.4f with acc on TEST TARGET SET %.2f with (EMA) acc on TETS TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_val,100*f1_val_ema, (end-start)))    
    print("TRAIN LOSS at Epoch %d: %.4f with acc on TEST TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_val, (end-start)))    
    sys.stdout.flush()


