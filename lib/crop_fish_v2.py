import numpy as np
import cv2
import json
import glob
import os
import urllib2

os.chdir('/Users/pengfeiwang/Desktop/f/data/')
LABELS_DIR = 'labels/'
TRAIN_DIR = 'train/'
OUTPUT_DIR = 'output1/'

LABELS_LINKS = [
    'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/5461/yft_labels.json?sv=2015-12-11&sr=b&sig=jiIUyzzgmF4sOl011pUortWzaZUMTr1jkTc4T1DivOE%3D&se=2016-12-02T14%3A58%3A24Z&sp=r',
    'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/5459/shark_labels.json?sv=2015-12-11&sr=b&sig=hyNNmI0DQ4WoY3BPa9J1Xmz8fsT1kLW3G5oQukd0LQw%3D&se=2016-12-02T14%3A57%3A51Z&sp=r',
    'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/5463/lag_labels.json?sv=2015-12-11&sr=b&sig=6VyFZYEqQpl7c7eyg03dUCVWA0X11ilfdC3MXf6DvT0%3D&se=2016-12-02T14%3A58%3A57Z&sp=r',
    'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/5460/dol_labels.json?sv=2015-12-11&sr=b&sig=%2FzAamolJIInQulkg7YriwpnAPKrFv25%2BcUIv2tozczU%3D&se=2016-12-02T14%3A58%3A10Z&sp=r',
    'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/5458/bet_labels.json?sv=2015-12-11&sr=b&sig=0tIpoSMwlSeLoXeHo74QhL9Ot7PLJRI48wN2Mw5LWmY%3D&se=2016-12-02T14%3A57%3A28Z&sp=r',
    'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/5462/alb_labels.json?sv=2015-12-11&sr=b&sig=p7iWS9hmDOOFsoI5pdJD4mQ3JebgWfZMgLTGbEJvHR0%3D&se=2016-12-02T14%3A58%3A38Z&sp=r',
]


def download_labels(LABELS_DIR, LABELS_LINKS):
    if not os.path.isdir(LABELS_DIR):
        os.mkdir(LABELS_DIR)
    for link in LABELS_LINKS:
        label_filename = link.split('?')[0].split('/')[-1]
        print("Downloading " + label_filename)
        f = urllib2.urlopen(link)
        with open(LABELS_DIR + label_filename, 'wb') as local_file:
            local_file.write(f.read())


def make_cropped_dataset(LABELS_DIR, OUTPUT_DIR):
    label_files = glob.glob(LABELS_DIR + '*.json')
    for file in label_files:
        process_labels(file, OUTPUT_DIR)


def process_labels(label_file, OUTPUT_DIR):
    file_name = os.path.basename(label_file)
    class_name = file_name.split("_")[0]
    if not os.path.isdir(OUTPUT_DIR + class_name.upper()):
        os.mkdir(OUTPUT_DIR + class_name.upper())
    print("Processing " + class_name + " labels")
    with open(label_file) as data_file:
        data = json.load(data_file)
    for img_data in data:
        img_file = TRAIN_DIR + class_name.upper() + '/' + img_data['filename']
        img = cv2.imread(img_file)
        # We will crop only images with both heads and tails present for cleaner dataset
        k = len(img_data['annotations'])
        for i in range(k):
            top_x = max(0,img_data['annotations'][i]['x'])
            top_y = max(0,img_data['annotations'][i]['y'])
            img_width = img_data['annotations'][i]['width']
            img_height = img_data['annotations'][i]['height']
            bot_x = top_x + img_width
            bot_y = top_y + img_height
            top_x, bot_x, top_y, bot_y = int(top_x), int(bot_x), int(top_y), int(bot_y)
            img = img[top_y:bot_y+1, top_x:bot_x+1, :] 
            cv2.imwrite(OUTPUT_DIR + class_name.upper() + '/' + str(i) + '_' + img_data['filename'], img)


download_labels(LABELS_DIR, LABELS_LINKS)
make_cropped_dataset(LABELS_DIR, OUTPUT_DIR)




