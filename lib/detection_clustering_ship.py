import glob
import os,time, pickle
import multiprocessing
import pandas as pd
import numpy as np
from sklearn import cluster
from scipy.misc import *
from random import shuffle
from shutil import copyfile

# <=====================================================================================>
# util
def compare(args):
    img, img2 = args
    img = (img - img.mean()) / img.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.mean(np.abs(img - img2))


def calculate_distance(all_ship_path):
    all_ship_dta = np.array([imread(i) for i in all_ship_path])
    train = [imresize(i, (128, 128)) for i in all_ship_dta]
    pool = multiprocessing.Pool(8)
    distances = np.zeros((len(train), len(train)))
    start = time.time()
    for i, img in enumerate(train):
        if i % 100 == 0:
            print i
            end = time.time()
            print (end-start)/60.
        all_imgs = [(img, f) for f in train]
        dists = pool.map(compare, all_imgs)
        distances[i, :] = dists
    return(distances)


def clf_dict(classification_list, corres_path):
    '''predict_result_list, path => dictionary'''
    d = {}
    class_list = zip(classification_list, corres_path)
    for i, j in class_list:
        d.setdefault(i, []).append(j)
    return(d)


def save_local(d,output_path):
    '''save the image according to the dict.keys'''
    for j in d.keys():
        for i in d[j]:
            out_dir = output_path + str(j)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            copyfile(i, out_dir + '/' + str(i.split('/')[-1]))


def print_cls_result(cls, distances):
    y = cls.fit_predict(distances)
    print(pd.Series(y).value_counts()) 
    return(y) 

# <=====================================================================================>
# change the dir
os.chdir('/Users/pengfeiwang/Desktop/f/data/')

# load the images
ship_file_path = glob.glob('./train/*/*.jpg')
ship_test_path = glob.glob('./test_stg1/*.jpg')
shuffle(ship_file_path)
ship_file_path = ship_file_path[:1500]
all_ship_path = ship_file_path + ship_test_path
label = len(ship_file_path)*[0] + len(ship_test_path)*[1]


# <=====================================================================================>
# calculate the distances among images
all_ship_dta = np.array([imread(i) for i in all_ship_path])
shapes = np.array([str(i.shape) for i in all_ship_dta])

distances = calculate_distance(all_ship_path)
pickle.dump(distances, open('./ship_distances.pkl', 'wb'))


# <=====================================================================================>
# classify the ship with DBSCAN

distances = pickle.load(open('./ship_distances.pkl', 'rb'))

## classifier I
cls = cluster.DBSCAN(metric='precomputed', min_samples=5, eps=0.5, leaf_size=40)
y = print_cls_result(cls, distances)
# d = clf_dict(y, all_ship_path) # for whole
d = clf_dict(y[1500:], ship_test_path) # for test 
# save to loacal
ship_output_path = '/Users/pengfeiwang/Desktop/c1/'
save_local(d, ship_output_path)


## classifier II
unc_file = glob.glob(ship_output_path + str(-1) + '/' + '*.jpg')
distances2 = calculate_distance(unc_file)
cls2 = cluster.DBSCAN(metric='precomputed', min_samples=5, eps=0.75, leaf_size=40)
y2 = print_cls_result(cls2, distances2)
d2 = clf_dict(y2, unc_file)
out_dir = '/Users/pengfeiwang/Desktop/c1/c2/'
save_local(d2, out_dir)

# find the dim of images to help combine
name=[i for i in os.listdir('/Users/pengfeiwang/Desktop/c1/') if i[0] != '.']
for i in name:
    print i,imread(glob.glob('/Users/pengfeiwang/Desktop/c1/'+str(i)+'/*.jpg')[0]).shape



# <=====================================================================================>
# find the relationship btn ship and crop image size

crop_train_path = glob.glob('./output_v1/*/*.jpg')

def stats_crop_image(crop_path_list):
    ship_dta = np.array([imread(i) for i in crop_path_list])
    shapes = np.array([i.shape for i in ship_dta])
    shape_mean = shapes.mean(axis=0)
    quantiles = np.percentile(shapes, [0, 25, 50, 75, 100], axis=0)
    print "-" * 45
    print "Shape Mean " + str(shape_mean)
    print "Quantiles "
    print quantiles
    return(shape_mean)


def find_the_crop(class_dict, crop_fish_path):
    find_path = []
    for i in class_dict:
        for j in crop_fish_path:
            if i.split('/')[-1] in j:
                find_path.append(j)
    return(find_path)

mean_list = []
for i in list(set(y)):
    print i
    try:
        crop_path = find_the_crop(d[i], crop_train_path)
        mean_list.append([i, stats_crop_image(crop_path)])
    except:
        print "Cannot get the infomation"



