import cv2
import os

base_path = '/Users/pengfeiwang/Desktop/f/data/train/'
negative_path = '/Users/pengfeiwang/Desktop/f/data/train/NoF'


def get_file_path(base_path):
    path_list = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            fullpath = os.path.join(root, f)
            if os.path.splitext(fullpath)[1] == '.jpg':
                path_list.append(fullpath)
    return(path_list)

full_name = get_file_path(base_path)
negative_name = [os.path.join(negative_path, i) for i in get_file_path(negative_path)]
positive_name = list(set(full_name) - set(negative_name))
