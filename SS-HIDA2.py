import sys
import numpy as np
import sys
from sklearn.utils import shuffle
import time
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad
from torchvision.models import resnet18
import torchvision.transforms as T 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from functions import MyDataset_Unl, MyDataset, cumulate_EMA, transform, TRAIN_BATCH_SIZE, LEARNING_RATE, MOMENTUM_EMA, EPOCHS, TH_FIXMATCH, WARM_UP_EPOCH_EMA
#import functions
from model_pytorch import FC_Classifier_NoLazy
import os

# Implementation based on: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/wdgrl.py

# Parameters from HIDA code https://github.com/mihailoobrenovic/HIDA 
# params['lr_wd_D'] = 1e-3  # Learning rate for training the Wasserstein distance domain critic
# params['gp_param'] = 10   # Gradient penalty regularization parameter when training the domain critic
# params['l2_param'] = 1e-5 # L2 regularization parameter when training the feature extractor and the classifier
# params['wd_param'] = 0.1  # The weight parameter of the Wasserstein distance loss in the total loss equation in train_adapter
# params['lr'] = 1e-4
LR = 1e-4
LR_DC = 1e-3
WEIGHT_DECAY = 1e-5
GP_PARAM = 10
DC_PARAM = 0.1
ITER_DC = 10 # Inner iterations for domain-critic optimization
ITER_CLF = 1 # Inner iterations for classifier optimization


def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred, _ = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

def gradient_penalty(critic, h_s, h_t):
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class SSHIDA(torch.nn.Module):
    def __init__(self, input_channel_source=4, input_channel_target=2, num_classes=10):
        super(SSHIDA, self).__init__()

        source_model = resnet18(weights=None) #resnet50(weights=None)
        source_model.conv1 = nn.Conv2d(input_channel_source, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.source = nn.Sequential(*list(source_model.children())[:6])

        target_model = resnet18(weights=None) #resnet50(weights=None)
        target_model.conv1 = nn.Conv2d(input_channel_target, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.target = nn.Sequential(*list(target_model.children())[:6])

        common_model = resnet18(weights=None) #resnet50(weights=None)
        self.common = nn.Sequential(*list(common_model.children())[6:-1])

        # self.task_cl = nn.Linear(in_features=512, out_features=n_classes)
        self.task_cl = FC_Classifier_NoLazy(input_dim=512, n_classes=num_classes)


    def forward(self, x, from_source=False):
        if from_source:
            common_emb = self.source(x)
        else:
            common_emb = self.target(x)
        
        emb = self.common(common_emb).view(x.shape[0], -1) #squeeze()

        # Classifier branch
        pred_cl = self.task_cl(emb)

        return emb, pred_cl



##########################
# MAIN FUNCTION: TRAINING
##########################
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



print("TRAINING ID SELECTED")
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
dataloader_train_target = DataLoader(dataset_train_target, shuffle=True, batch_size=TRAIN_BATCH_SIZE//2)

#DATALOADER TARGET UNLABELLED
x_train_target_unl = torch.tensor(test_target_data_unl, dtype=torch.float32)

dataset_train_target_unl = MyDataset_Unl(x_train_target_unl, transform)
dataloader_train_target_unl = DataLoader(dataset_train_target_unl, shuffle=True, batch_size=TRAIN_BATCH_SIZE//2)

#DATALOADER TARGET TEST
x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
dataset_test_target = TensorDataset(x_test_target, y_test_target)
dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=512)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("source_data.shape[1] ",source_data.shape[1])
print("target_data.shape[1] ",target_data.shape[1])
sys.stdout.flush()


model = SSHIDA(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
model = model.to(device)

critic = nn.Sequential(
    nn.Linear(512,100),
    # nn.BatchNorm1d(100), # set to False by default
    nn.ReLU(),
    # nn.Dropout(p=dropout_prob), # set to False
    nn.Linear(100, 1)
).to(device)

clf_criterion = nn.CrossEntropyLoss()
critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_DC)
clf_optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) 


# Loop through the data
valid_f1 = 0.0
ema_weights = None

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    tot_loss = 0.0
    tot_dc_loss = 0.0
    den = 0

    for x_batch_source, y_batch_source in dataloader_source:
        if x_batch_source.shape[0]< TRAIN_BATCH_SIZE:
            continue # To avoid errors on pairing source/target samples

        # Load batch data
        x_batch_target, y_batch_target = next(iter(dataloader_train_target))
        x_batch_target_unl, x_batch_target_unl_aug = next(iter(dataloader_train_target_unl))

        x_batch_source = x_batch_source.to(device)
        y_batch_source = y_batch_source.to(device)
        
        x_batch_target = x_batch_target.to(device)
        y_batch_target = y_batch_target.to(device)

        x_batch_target_unl = x_batch_target_unl.to(device)
        # x_batch_target_unl_aug = x_batch_target_unl_aug.to(device) # unused

        x_batch_target_all = torch.cat((x_batch_target,x_batch_target_unl_aug),dim=0)

        # Train critic
        set_requires_grad(model, requires_grad=False)
        set_requires_grad(critic, requires_grad=True)
        with torch.no_grad():
            h_s, _ = model(x_batch_source, from_source=True)
            h_t, _ = model(x_batch_target_all)
        for _ in range(ITER_DC):
            gp = gradient_penalty(critic, h_s, h_t)

            critic_s = critic(h_s)
            critic_t = critic(h_t)
            wasserstein_distance = critic_s.mean() - critic_t.mean()

            critic_cost = -wasserstein_distance + GP_PARAM*gp

            critic_optim.zero_grad()
            critic_cost.backward()
            critic_optim.step()

            tot_dc_loss+=critic_cost.cpu().detach().numpy()
 
        # Train classifier
        set_requires_grad(model, requires_grad=True)
        set_requires_grad(critic, requires_grad=False)
        for _ in range(ITER_CLF):
            emb_s, pred_s = model(x_batch_source, from_source=True)
            emb_t, pred_t = model(x_batch_target)
            emb_t_unl, _  = model(x_batch_target_unl)

            # Classification loss
            pred = torch.cat((pred_s, pred_t),dim=0)
            y_batch = torch.cat((y_batch_source, y_batch_target),dim=0)
            loss_pred = clf_criterion(pred, y_batch)
            # Alternative: give same weight to source and target (regardless of nb labeled samples)
            # loss_pred = clf_criterion(pred_s, y_batch_source) + clf_criterion(pred_t, y_batch_target)

            emb_t_all = torch.cat((emb_t, emb_t_unl),dim=0) # all target embeddings (labelled + unlabelled)
            wasserstein_distance = critic(emb_s).mean() - critic(emb_t_all).mean()

            loss = loss_pred + DC_PARAM * wasserstein_distance
            clf_optim.zero_grad()
            loss.backward()
            clf_optim.step()

        tot_loss+= loss.cpu().detach().numpy()
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
    
    print("TRAIN LOSS at Epoch %d: WITH TOTAL LOSS %.4f (dc_loss %.4f). Acc on TEST TARGET SET (ORIG) %.2f (EMA) %.2f with train time %d"%(epoch, tot_loss/den, tot_dc_loss/den, 100*f1_val, 100*f1_val_ema, (end-start)))    
    sys.stdout.flush()

dir_name = "./results/HIDA"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

output_file = dir_name+"/%s_%s_%s.pth"%(target_prefix, nsplit, nsamples)
model.load_state_dict(ema_weights)
torch.save(model.state_dict(), output_file)