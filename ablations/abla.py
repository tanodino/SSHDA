### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
from model_pytorch import ORDisModel
import time
from sklearn.metrics import f1_score
import torchvision.transforms as T 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import sys
sys.path.append('..')

from functions import MyDataset_Unl, MyDataset, cumulate_EMA, transform, TRAIN_BATCH_SIZE, LEARNING_RATE, MOMENTUM_EMA, EPOCHS, TH_FIXMATCH, WARM_UP_EPOCH_EMA
import functions
import os


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
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

##########################
# MAIN FUNCTION: TRAINING
##########################
def main():
    dir_ = sys.argv[1]
    source_prefix = sys.argv[2]
    target_prefix = sys.argv[3]
    nsamples = sys.argv[4]
    nsplit = sys.argv[5]
    abla_number = int(sys.argv[6])

    source_data = np.load("%s/%s_data_filtered.npy"%(dir_,source_prefix) )
    target_data = np.load("%s/%s_data_filtered.npy"%(dir_,target_prefix) )
    source_label = np.load("%s/%s_label_filtered.npy"%(dir_,source_prefix) )
    target_label = np.load("%s/%s_label_filtered.npy"%(dir_,target_prefix) )

    train_target_idx = np.load("%s/%s_%s_%s_train_idx.npy"%(dir_, target_prefix, nsplit, nsamples))
    test_target_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_target_idx)

    train_target_data = target_data[train_target_idx]
    train_target_label = target_label[train_target_idx]

    test_target_data = target_data[test_target_idx]
    test_target_label = target_label[test_target_idx]

    test_target_data_unl = target_data[test_target_idx]

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

    model = ORDisModel(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    # Loop through the data
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

            unl_inv = torch.cat([unl_target_inv,unl_target_aug_inv],dim=0)
            norm_unl_inv = F.normalize(unl_inv)
            unl_spec = torch.cat([unl_target_spec,unl_target_aug_spec],dim=0)
            norm_unl_spec = F.normalize(unl_spec)
            u_loss_ortho = torch.mean( torch.sum( norm_unl_inv * norm_unl_spec, dim=1) )

            loss = None
            if abla_number == 1:
                loss = loss_pred
            elif abla_number == 2:
                loss = loss_pred + loss_ortho + loss_dom
            elif abla_number == 3:
                loss = loss_pred + loss_ortho + loss_dom + u_loss_ortho + u_loss_dom 
            elif abla_number == 4:
                loss = loss_pred + loss_ortho + loss_dom  + u_pred_loss 
            elif abla_number == 5:
                loss = loss_pred + loss_dom + u_loss_dom + u_pred_loss
            elif abla_number == 6:
                loss = loss_pred + loss_ortho + u_loss_ortho + u_pred_loss
            else:
                loss = loss_pred + loss_ortho + loss_dom + u_loss_ortho + u_loss_dom + u_pred_loss
            
            loss.backward() # backward pass: backpropagate the prediction loss
            optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
            
            tot_loss+= loss.cpu().detach().numpy()
            tot_ortho_loss+=loss_ortho.cpu().detach().numpy()
            tot_fixmatch_loss+=u_pred_loss.cpu().detach().numpy()
            den+=1.

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

    dir_name = dir_+"/OUR_abla_%d"%abla_number
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    output_file = dir_name+"/%s_%s_%s.pth"%(target_prefix, nsplit, nsamples)
    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)

if __name__ == "__main__":
    main()

