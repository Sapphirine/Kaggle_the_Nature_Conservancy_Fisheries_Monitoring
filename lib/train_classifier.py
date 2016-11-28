from scipy.misc import *
import re
import os


def get_file_path(path):
    path_list = []
    for root, dirs, files in os.walk(path):
        for f in files:
            fullpath = os.path.join(root, f)
            if os.path.splitext(fullpath)[1] == '.jpg':
                path_list.append(fullpath)
    return(path_list)

path = '/Users/pengfeiwang/Desktop/f/data/output/'
neg_path = '/Users/pengfeiwang/Desktop/f/data/train/NoF'

pos_name = get_file_path(path)
neg_name = os.listdir(neg_path)

# resize and save
for i, j in enumerate(pos_name):
    train[:] = imresize(imread(j), [128, 128, 3])
    imsave(j, train)
    with open('/Users/pengfeiwang/Desktop/f/data/output/pos.info', 'ab') as g:
        g.write(j + ' 1 0 0 128 128 \n')

for i in neg_name:
    with open('/Users/pengfeiwang/Desktop/f/data/output/neg.info', 'ab') as g:
        g.write(i + '\n')

# in bash
# opencv_createsamples -info /Users/pengfeiwang/Desktop/f/data/output/pos.info -num ? -w 128 -h 128 -vec /Users/pengfeiwang/Desktop/f/data/output/fish.vec
# opencv_createsamples -vec /Users/pengfeiwang/Desktop/f/data/output/fish.vec -w 128 -h 128
# opencv_traincascade -data output/file/dir -vec \
# /Users/pengfeiwang/Desktop/f/data/output/fish.vec -bg \
# /Users/pengfeiwang/Desktop/f/data/output/neg.info -numPos ? -numNeg ? -w \
# 128 -h 128 -featureType LBP
