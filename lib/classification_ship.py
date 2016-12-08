import glob, os, cv2
import multiprocessing
import pandas as pd
import numpy as np
from sklearn import cluster
from scipy.misc import *


os.chdir('/Users/pengfeiwang/Desktop/f/data/')

ship_file_path = glob.glob('./train/*/*.jpg')
ship_dta = np.array([imread(i) for i in ship_file_path])

shapes = np.array([str(i.shape) for i in ship_dta])
pd.Series(shapes).value_counts()

def compare(args):
    img, img2 = args
    img = (img - img.mean()) / img.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.mean(np.abs(img - img2))

train = [imresize(i, (224, 224)) for i in ship_dta]


pool = multiprocessing.Pool(8)
distances = np.zeros((len(train), len(train)))

for i, img in enumerate(train): 
    all_imgs = [(img, f) for f in train]
    dists = pool.map(compare, all_imgs)
    distances[i, :] = dists

cls = cluster.DBSCAN(metric='precomputed', min_samples=5, eps=0.6)
y = cls.fit_predict(distances)

print('Cluster sizes:')
print(pd.Series(y).value_counts())


class_list = zip(y, ship_file_path)
class_path = [filter(lambda x: x[0]==i, class_list) for i in range()]




