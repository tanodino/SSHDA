import numpy as np
from sklearn.utils import shuffle

def rescale(data):
    min_ = np.percentile(data, 2)
    max_ = np.percentile(data, 98)
    return np.clip( (data - min_) / (max_ - min_), 0, 1.)

def getIdxVal(sub_hashCl2idx, val):
    idx = []
    for k in sub_hashCl2idx.keys():
        temp = sub_hashCl2idx[k]
        idx.append(temp[0:val])
    return np.concatenate(idx, axis=0)


def get_idxPerClass(hashCl2idx, max_val):
    sub_hashCl2idx = {}
    for k in hashCl2idx.keys():
        temp = hashCl2idx[k]
        temp = shuffle(temp)
        sub_hashCl2idx[k] = temp[0:max_val]
    return sub_hashCl2idx


def extractWriteTrainIdx(nrepeat, nsample_list, hashCl2idx, prefix):
    max_val = nsample_list[-1]
    for i in range(nrepeat):
        sub_hashCl2idx = get_idxPerClass(hashCl2idx, max_val)
        for val in nsample_list:
            idx = getIdxVal(sub_hashCl2idx, val)
            np.save("%s_%d_%d_train_idx.npy"%(prefix,i,val), idx) 


def getHash2classes(labels):
    hashCl2idx = {}
    for v in np.unique(labels):
        idx = np.where(labels == v)[0]
        idx = shuffle(idx)
        hashCl2idx[v] = idx
    return hashCl2idx
        
def writeFilteredData(prefix, data, label):
    np.save("%s_data_filtered.npy"%prefix,data)
    np.save("%s_label_filtered.npy"%prefix,label)

#READ DATA
data_r = np.load("RESISC45_data.npy").astype("float32")
data_e = np.load("EuroSAT_data.npy").astype("float32")

label_r = np.load("RESISC45_labels.npy")
label_e = np.load("EuroSAT_labels.npy")

#RESCALE DATA BETWEEN 0 and 1 per band
for i in range(data_r.shape[1]):
    data_r[:,i,:,:] = rescale(data_r[:,i,:,:])

for i in range(data_e.shape[1]):
    data_e[:,i,:,:] = rescale(data_e[:,i,:,:])


#WRITE DATA
writeFilteredData("RESISC45", data_r, label_r)
writeFilteredData("EURO", data_e, label_e)

r_hashCl2idx = getHash2classes(label_r)
e_hashCl2idx = getHash2classes(label_e)

#EXTRACT 10 time TRAIN IDX INCREASING THE NUMBER OF SAMLPE PER CLASS FROM 50 TO 400
nrepeat = 5
nsample_list = np.arange(50,401,50)
extractWriteTrainIdx(nrepeat, nsample_list, r_hashCl2idx, "RESISC45")
extractWriteTrainIdx(nrepeat, nsample_list, e_hashCl2idx, "EURO")



