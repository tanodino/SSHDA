import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model_pytorch import ORDisModel
from torchvision.models import resnet18
from sklearn.metrics import f1_score
from functions import TRAIN_BATCH_SIZE

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


dir_ = sys.argv[1]
method = sys.argv[2]
target_prefix = sys.argv[3]
nsamples = sys.argv[4]
#nsplit = sys.argv[5]
nsplits = 5

data = np.load("%s/%s_data_filtered.npy"%(dir_, target_prefix))
label = np.load("%s/%s_label_filtered.npy"%(dir_, target_prefix))

n_classes = len(np.unique(label))

model = resnet18(weights=None)
model.conv1 = nn.Conv2d(data.shape[1], 64, kernel_size=7, stride=2, padding=3,bias=False)
model._modules["fc"]  = nn.Linear(in_features=512, out_features=n_classes )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

acc_f1 = []
for nsplit in range(nsplits):
    train_idx = np.load("%s/%s_%s_%s_train_idx.npy"%(dir_, target_prefix, nsplit, nsamples))
    test_idx = np.setdiff1d(np.arange(data.shape[0]), train_idx)
    test_data = data[test_idx]
    test_label = label[test_idx]

    #DATALOADER TEST
    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_label, dtype=torch.int64)
    dataset_test = TensorDataset(x_test, y_test)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

    path = dir_+"/"+method+"/%s_%s_%s.pth"%(target_prefix, nsplit, nsamples)
    model.load_state_dict(torch.load(path))
    model.eval()
    pred, labels = evaluation(model, dataloader_test, device)
    f1_val = f1_score(labels, pred, average="weighted")
    acc_f1.append(f1_val)

print("F1 %.2f +- %.2f"%(np.mean(acc_f1)*100, np.std(acc_f1)*100 ))