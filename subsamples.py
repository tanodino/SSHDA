import sys
import glob
import numpy as np
from sklearn.utils import shuffle


def getNewTrainIdx(labels, train_idx, val):
    newTrainIdx = []
    hashMap = {}
    for l in np.unique(labels):
        hashMap[l] = []
    for v in train_idx:
        hashMap[labels[v]].append(v)
    for l in hashMap.keys():
        temp = shuffle( hashMap[l] )
        newTrainIdx.append( temp[0:val] )
    newTrainIdx = np.concatenate(newTrainIdx,axis=0)
    return newTrainIdx
        



dir_name = sys.argv[1]
prefix_source = sys.argv[2]
#MS_1_300_train_idx.npy
query_string = "_50_"
val = 25
substitute_string = "_%d_"%val

fileNames = glob.glob("%s/%s*%s*py"%(dir_name,prefix_source,query_string))
for fName in fileNames:
    labels = np.load("%s/%s_label_filtered.npy"%(dir_name,prefix_source))
    train_idx = np.load(fName)
    newTrainIdx = getNewTrainIdx(labels, train_idx, val)
    newFileName = fName.replace(query_string, substitute_string)
    print(newFileName)
    print(newTrainIdx.shape)
    print("============")
    np.save(newFileName, newTrainIdx)