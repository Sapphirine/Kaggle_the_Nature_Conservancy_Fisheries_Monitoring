import cv2
import os
import pickle
import numpy as np
from scipy.misc import *
import matplotlib.pyplot as plt

'''
detect the fish using haar cascade
'''

os.chdir('/Users/pengfeiwang/Desktop/f/fisheries/')

location = pickle.load(open('/Users/pengfeiwang/Desktop/location.pkl', 'rb'))
fish_cascade = cv2.CascadeClassifier('./output/redfish.xml')

# get the training images in the clustering
train_path = glob.glob('/Users/pengfeiwang/Desktop/f/data/train/*/*.jpg')
train_name = [i.rsplit('/', 1)[1] for i in train_path]
clustering_path = glob.glob('/Users/pengfeiwang/Desktop/c1/*/*.jpg')
input_path = [i for i in clustering_path if i.rsplit('/', 1)[1] in train_name]
output_path = '/Users/pengfeiwang/Desktop/a/'

for i in input_path:
	output_name = i.split('/')[-1]
	clustering_class = input_path.split('/')[-2]
	a, b, c, d = 0
	a, b = location[clustering_class][0]
	c, d = location[clustering_class][1]
	image = cv2.imread(i)
	image = image[a:c,b:d]
	img = imresize(image, [64, 64, 3])
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	fish = fish_cascade.detectMultiScale(gray, 1.3, 5)
	for j, (x,y,w,h) in enumerate(fish):
	    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)  # color thickness
	    crop_fish = img[y:y+h, x:x+w]
	    cv2.imwrite(os.path.join(output_path, str(j)+ '_' + output_name), crop_fish, )
	cv2.imshow('img',img)
	plt.show()		    

cv2.destroyAllWindows()


