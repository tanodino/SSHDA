import glob
import rasterio as rio
import numpy as np

'''
######## CREATE SAR DATASET #######
'''

'''
directory = "EuroSAT-SAR"
files = glob.glob(directory+"/*")
#print(files)
#exit()
hash_class = {}
class_label = []
dataset = []
for f in files:
    
    class_name = f.split("/")[-1]
    if class_name not in hash_class:
        hash_class[class_name] = len(hash_class)
    
    fileNames = glob.glob("%s/*.tif"%f)
    print("PROCESSING %s with %d imgs"%(f,len(fileNames)))
    for fs in fileNames:
        src = rio.open(fs)
        data = src.read()
        dataset.append( np.moveaxis(data,(0,1,2),(2,0,1)) )
        class_label.append( hash_class[class_name]  )
        src.close()

dataset = np.array(dataset)
class_label = np.array(class_label)
np.save("SAR_data.npy",dataset)
np.save("labels.npy",class_label)

'''
######## CREATE MS DATASET - R G B NIR #######

######## CREATE SAR DATASET #######

directory = "EuroSAT-MS"
#bands2select = [2,3,4,9] # B2, B3, B4, B8a
files = glob.glob(directory+"/*")
hash_class = {}
class_label = []
dataset = []
for f in files:
    print("PROCESSING %s"%f)
    class_name = f.split("/")[-1]
    if class_name not in hash_class:
        hash_class[class_name] = len(hash_class)
    
    fileNames = glob.glob("%s/*.tif"%f)
    for fs in fileNames:
        '''
        src = rio.open(f)
        data = []
        for b in bands2select:
            temp_data = src.read(b)
            data.append(temp_data)
        src.close()
        data = np.stack(data,axis=-1)
        #print(data.shape)
        #exit()
        dataset.append( data )
        class_label.append( hash_class[class_name]  )
        src.close()
        '''
        src = rio.open(fs)
        data = src.read()
        dataset.append( np.moveaxis(data,(0,1,2),(2,0,1)) )
        class_label.append( hash_class[class_name]  )
        src.close()

dataset = np.array(dataset)
class_label = np.array(class_label)
np.save("MS_data.npy",dataset)
np.save("MS_labels.npy",class_label)