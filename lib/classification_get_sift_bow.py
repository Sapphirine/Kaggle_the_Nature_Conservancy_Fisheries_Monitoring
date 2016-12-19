import os
import joblib
import numpy as np
from cv2 import *

# <====================================================================================================>
# utils
def preProcessImages(image_paths):
    descriptors= []
    for image_path in image_paths:
        im = imread(image_path)
        kpts = feature_det.detect(im)
        kpts, des = feature_det.compute(im, kpts)
        descriptors.append(des)
    return descriptors


def getImagedata(feature_det,bow_extract,path):
    im = imread(path)
    featureset = bow_extract.compute(im, feature_det.detect(im))
    return featureset

# <====================================================================================================>
# get all the data and correspondinf class
os.chdir('/Users/pengfeiwang/Desktop/f/data/')

image_paths = glob.glob('./output/*/*.jpg')
fish_species = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
image_classes = [0] * len(image_paths)

for i,j in enumerate(fish_species):
	image_classes = np.add(image_classes, [i if re.search(j, k) else 0 for k in image_paths])

image_classes = image_classes.tolist()

# <====================================================================================================>
# draw the sift features and use bow to reduce dimension
feature_det = xfeatures2d.SIFT_create()
descriptors = preProcessImages(image_paths)

descriptors_none = []
for i, j in enumerate(descriptors):
    if j == None:
        descriptors_none.append(i)

descriptors = [i for i in descriptors if i!= None]
image_classes = [image_classes[i] for i in range(len(image_classes)) if i not in descriptors_none]
image_paths = [image_paths[i] for i in range(len(image_paths)) if i not in descriptors_none]

for des in descriptors:
    bow_train.add(des)

matcher = BFMatcher(NORM_L2)
bow_extract  = BOWImgDescriptorExtractor(feature_det,matcher)
bow_train = BOWKMeansTrainer(500)
voc = bow_train.cluster()
bow_extract.setVocabulary(voc)

joblib.dump((voc), "fullvoc.pkl", compress=3)

# <====================================================================================================>
# get features from the training data based on the vocabulary & approximate nearest neighbour
# features are used as the training data 
traindata = []  

for path in image_paths:
    featureset = getImagedata(feature_det,bow_extract,path)
    traindata.append(featureset)

traindata = np.array(traindata).reshape(len(np.array(traindata)), -1)
joblib.dump((traindata), "train_data.pkl")


