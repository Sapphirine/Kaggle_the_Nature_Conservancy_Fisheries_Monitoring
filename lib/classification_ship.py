import glob, os
import multiprocessing
import pandas as pd
import numpy as np
from sklearn import cluster
from scipy.misc import *


os.chdir('/Users/pengfeiwang/Desktop/f/data/')


ship_file_path = glob.glob('./train/*/*.jpg')
ship_file_path = ship_file_path[:500]
ship_dta = np.array([imread(i) for i in ship_file_path])
shapes = np.array([str(i.shape) for i in ship_dta])


def compare(args):
    img, img2 = args
    img = (img - img.mean()) / img.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.mean(np.abs(img - img2))


train = [imresize(i, (224, 224)) for i in ship_dta]
pool = multiprocessing.Pool(8)
distances = np.zeros((len(train), len(train)))

for i, img in enumerate(train): 
    if i%100==0:
        print i
    all_imgs = [(img, f) for f in train]
    dists = pool.map(compare, all_imgs)
    distances[i, :] = dists


cls = cluster.DBSCAN(metric='precomputed', min_samples=5, eps=0.6)
y = cls.fit_predict(distances)
print(pd.Series(y).value_counts())

# group by into dict
d = {}
class_list = zip(y, ship_file_path)
for x, y in class_list:
    d.setdefault(x, []).append(y)

pickle.dump(d,open('./output/ship_class.pkl','wb'))
d = pickle.load(open('./output/ship_class.pkl','rb'))



# find the relationship btn ship and crop image size
crop_train_path = glob.glob('./output/*/*.jpg')

def stats_crop_image(crop_path_list):
	ship_dta = np.array([imread(i) for i in crop_path_list])
	shapes = np.array([i.shape for i in ship_dta])
	shape_mean = shapes.mean(axis=0)
	quantiles = np.percentile(shapes, [0,25,50,75,100], axis=0)
	print "-"*45
	print "Shape Mean " + str(shape_mean)
	print "Quantiles " 
	print quantiles

def find_the_crop(d_list):
	find_path = []
	for i in d_list:
		for j in crop_train_path:
			if i.split('/')[-1] in j:
				find_path.append(j)
	return(find_path)

crop_path = find_the_crop(d[0])
stats_crop_image(crop_path)
