import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from SSHIDA import SSHIDA
#from torchvision.models import resnet18
from sklearn.metrics import f1_score
from functions import TRAIN_BATCH_SIZE
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
        _, pred = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels


dir_ = sys.argv[1]
source_prefix = sys.argv[2]
target_prefix = sys.argv[3]
#abla_id = int(sys.argv[5])
#nsamples = sys.argv[4]
#nsplit = sys.argv[5]
nsplits = 5

source_data = np.load("%s/%s_data_filtered.npy"%(dir_,source_prefix) )
target_data = np.load("%s/%s_data_filtered.npy"%(dir_, target_prefix))
target_label = np.load("%s/%s_label_filtered.npy"%(dir_, target_prefix))

n_classes = len(np.unique(target_label))


model = SSHIDA(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)


tot_mean = []
tot_std = []

for nsamples in [25, 50, 100, 200]:
    acc_f1_nsample = []
    for nsplit in range(nsplits):
        
        path = dir_+"/SSHIDA/%s_%s_%s.pth"%(target_prefix, nsplit, nsamples)
        print(path)
        if not os.path.exists(path):
            continue

        train_idx = np.load("%s/%s_%s_%s_train_idx.npy"%(dir_, target_prefix, nsplit, nsamples))
        test_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_idx)
        test_data = target_data[test_idx]
        test_label = target_label[test_idx]

        #DATALOADER TEST
        x_test = torch.tensor(test_data, dtype=torch.float32)
        y_test = torch.tensor(test_label, dtype=torch.int64)
        dataset_test = TensorDataset(x_test, y_test)
        dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

        model.load_state_dict(torch.load(path,map_location=torch.device(device)))
        model.eval()
        pred, labels = evaluation(model, dataloader_test, device)
        f1_val = f1_score(labels, pred, average="weighted")
        acc_f1_nsample.append(f1_val)

    print("%.2f $\pm$ %.2f"%(np.mean(acc_f1_nsample)*100, np.std(acc_f1_nsample)*100 ))    
    tot_mean.append( np.mean(acc_f1_nsample) )
    tot_std.append( np.std(acc_f1_nsample) )

st = []
for i in range(len(tot_mean)):
    st.append("%.2f $\pm$ %.2f"%(tot_mean[i]*100, tot_std[i]*100))

print( " & ".join(st) )
