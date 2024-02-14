import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
import time
from sklearn.metrics import f1_score
from torchvision.models import resnet18
import torchvision.transforms as T 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from functions import MyDataset_Unl, MyDataset, cumulate_EMA, transform, TRAIN_BATCH_SIZE, LEARNING_RATE, MOMENTUM_EMA, EPOCHS, TH_FIXMATCH, WARM_UP_EPOCH_EMA
#import functions
from model_pytorch import FC_Classifier_NoLazy
import os

# Parameters from HIDA code https://github.com/mihailoobrenovic/HIDA 
# params['lr_wd_D'] = 1e-3  # Learning rate for training the Wasserstein distance domain critic
# params['gp_param'] = 10   # Gradient penalty regularization parameter when training the domain critic
# params['l2_param'] = 1e-5 # L2 regularization parameter when training the feature extractor and the classifier
# params['wd_param'] = 0.1  # The weight parameter of the Wasserstein distance loss in the total loss equation in train_adapter
# params['lr'] = 1e-4
WEIGHT_DECAY = 1e-5
GP_PARAM = 10
DC_PARAM = 0.1

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

def gradient_penalty(model, h_s, h_t):
    # taken from: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/wdgrl.py
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = model.critic(interpolates)
    gradients = torch.autograd.grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        #print(alpha)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -ctx.alpha
        return output, None

def grad_reverse(x,alpha):
    return GradReverse.apply(x,alpha)


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

        self.dom_critic = nn.Sequential(
            nn.Linear(512,100),
            # nn.BatchNorm1d(100), # set to False by default
            nn.ReLU(),
            # nn.Dropout(p=dropout_prob), # set to False
            nn.Linear(100, 1)
        )


    def forward_train(self, x, from_source=False):
        #self.train()
        if from_source:
            common_emb = self.source(x)
        else:
            common_emb = self.target(x)
        
        emb = self.common(common_emb).squeeze()

        # Classifier branch
        pred_cl = self.task_cl(emb)

        # Domain-critic branch
        emb_grl = grad_reverse(emb, alpha=1.)
        dom_crit = self.dom_critic(emb_grl)

        return emb_grl, pred_cl, dom_crit
    
    def forward(self,x):
        #self.eval()
        common_emb = self.target(x)
        emb = self.common(common_emb).squeeze()
        pred_cl = self.task_cl(emb)

        return pred_cl
    
    def critic(self,emb):
        #self.eval()
        dom_crit = self.dom_critic(emb)

        return dom_crit    


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


model = SSHIDA(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
# loss_fn_noReduction = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Loop through the data
valid_f1 = 0.0
ema_weights = None
torch.autograd.set_detect_anomaly(True)

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    tot_loss = 0.0
    tot_dc_loss = 0.0
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
        x_batch_target_unl_aug = x_batch_target_unl_aug.to(device) # unused

        emb_source, pred_cl_source, dom_crit_source = model.forward_train(x_batch_source,from_source=True)
        emb_target, pred_cl_target, dom_crit_target = model.forward_train(x_batch_target)
        emb_target_unl, _, dom_crit_target_unl = model.forward_train(x_batch_target_unl)

        # Classification loss
        pred_task = torch.cat((pred_cl_source, pred_cl_target),dim=0)
        y_batch = torch.cat((y_batch_source, y_batch_target),dim=0)
        
        loss_pred = loss_fn(pred_task, y_batch)

        # Domain-critic loss (+ gradient penalty)
        dom_crit_target_all = torch.cat((dom_crit_target, dom_crit_target_unl),dim=0)
        dom_crit_all = torch.cat((dom_crit_source, dom_crit_target, dom_crit_target_unl),dim=0)
        emb_all = torch.cat((emb_source, emb_target, emb_target_unl),dim=0)
        # gradient penalization
        dc_grad_source = torch.autograd.grad(dom_crit_source, emb_source,
                                      grad_outputs=torch.ones_like(dom_crit_source),
                                      retain_graph=True, create_graph=True)[0] 
        dc_grad_target = torch.autograd.grad(dom_crit_target, emb_target,
                                      grad_outputs=torch.ones_like(dom_crit_target),
                                      retain_graph=True, create_graph=True)[0] 
        dc_grad_target_unl = torch.autograd.grad(dom_crit_target_unl, emb_target_unl,
                                      grad_outputs=torch.ones_like(dom_crit_target_unl),
                                      retain_graph=True, create_graph=True)[0] 
        dc_grad = torch.cat((dc_grad_source, dc_grad_target, dc_grad_target_unl),dim=0)
        # dc_grad = torch.autograd.grad(dom_crit_all, emb_all,
        #                               grad_outputs=torch.ones_like(dom_crit_all),
        #                               retain_graph=True, create_graph=True)[0] # emb_whole (contains also linear combinations)
        # dc_slopes = torch.sqrt(torch.sum(dc_grad**2, dim=1))
        # dc_grad_penalty = torch.mean((dc_slopes - 1.0)**2)
        dc_slopes = dc_grad.norm(2, dim=1)
        dc_grad_penalty = ((dc_slopes - 1)**2).mean()        
        if torch.isnan(dc_grad_penalty):
            print('dc_grad_penalty is NaN!!!')
            print(dc_slopes)
            exit()        
        # wasserstein estimate
        wd_loss = torch.mean(dom_crit_source) - torch.mean(dom_crit_target_all)
        # total domain-critic
        dc_loss = - wd_loss + GP_PARAM*dc_grad_penalty

        if torch.isnan(dc_loss):
            print('dc_loss is NaN!!!')
            exit()

        # Final loss
        loss = loss_pred + DC_PARAM*dc_loss
        # In the code, they use the following: (in the paper, it is as above)
        # l2_loss = sum(torch.norm(p)**2 for p in train.parameters() if 'bias' not in p[0])
        # loss_rest = loss_pred + WEIGHT_DECAY*l2_loss + DC_PARAM*wd_loss # for all modules except the domain critic layers
        # loss_dc = dc_loss # for all modules except the domain critic layers

        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
  
        tot_loss+= loss.cpu().detach().numpy()
        tot_dc_loss+=dc_loss.cpu().detach().numpy()
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