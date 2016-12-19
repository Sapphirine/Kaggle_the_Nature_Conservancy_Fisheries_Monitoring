from scipy.misc import *
import re
import os

'''
train the haar cascade
'''

def get_file_path(path):
    path_list = []
    for root, dirs, files in os.walk(path):
        for f in files:
            fullpath = os.path.join(root, f)
            if os.path.splitext(fullpath)[1] == '.jpg':
                path_list.append(fullpath)
    return(path_list)


base = '/Users/pengfeiwang/Desktop/f/'
path = base + 'data/output/'
neg_path = base + 'data/train/NoF/'
neg_out_path = base + 'data/output/NoF/'

pos_name = get_file_path(path)
neg_name = os.listdir(neg_path)

# pisitive instance file
for i, j in enumerate(pos_name):
    train = []
    train[:] = imresize(imread(j), [128, 128, 3])
    imsave(j, train)
    with open(base + 'data/output/pos.info', 'ab') as g:
        g.write(j + ' 1 0 0 128 128 \n')

# negative instance file
for i in neg_name:
    if i[0] != '.':
        i_name = neg_out_path + i
        neg = []
        neg[:] = imresize(imread(neg_path + i), [128, 128, 3])
        imsave(i_name, neg)
        with open(base + 'data/output/neg.info', 'ab') as g:
            g.write(i_name + '\n')

# in bash
# cd into the location of pos.info
opencv_createsamples -info pos.info -num 4471 -w 128 -h 128 -vec /Users/pengfeiwang/Desktop/f/data/output/fish.vec
opencv_createsamples -vec fish.vec -w 128 -h 128
opencv_traincascade -data data -vec fish.vec -bg neg_info.txt -numStages 10 -numPos 4000 -numNeg 465 -w 128 -h 128 -featureType LBP

# numPos should be less than the vec number, which is 4471



