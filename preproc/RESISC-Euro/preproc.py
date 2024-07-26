import glob
import rasterio as rio
import numpy as np
from PIL import Image

######## CREATE RESISC45 DATASET #######
'''
'''
directory = "subRESISC45"
hash_Resisc45= {"dense_residential":0, "forest":1, "freeway":2, "industrial_area":3, "lake":4, "meadow":5, "rectangular_farmland":6, "river":7}

dataset = []
class_label = []
for k in hash_Resisc45.keys():
    fileNames = glob.glob(directory+"/"+k+"/*.jpg")
    for fName in fileNames:
        img = Image.open(fName)
        img = np.array(img)
        img = np.moveaxis(img,(0,1,2),(1,2,0)).astype("float32")
        dataset.append(img)
        class_label.append(hash_Resisc45[k])

dataset = np.array(dataset)
class_label = np.array(class_label)

np.save("RESISC45_data.npy",dataset)
np.save("RESISC45_labels.npy",class_label)

#exit()
#'''
#'''

######## CREATE MS DATASET - R G B NIR #######

######## CREATE SAR DATASET #######
directory_eurosat = "../EuroSAT_OPT_SAR/EuroSAT-MS/"
hash_EuroSat = {"Residential":0, "Forest":1, "Highway":2, "Industrial":3, "SeaLake":4, "Pasture":5, "AnnualCrop":6, "PermanentCrop":6, "River":7}
directory = "EuroSAT-MS"

class_label = []
dataset = []
for k in hash_EuroSat.keys():
    fileNames = glob.glob(directory_eurosat+"/"+k+"/*.tif")
    for fName in fileNames:
        src = rio.open(fName)
        dataset.append( src.read() )
        src.close()
        class_label.append( hash_EuroSat[k]  )

dataset = np.array(dataset)
class_label = np.array(class_label)
np.save("EuroSAT_data.npy",dataset)
np.save("EuroSAT_labels.npy",class_label)