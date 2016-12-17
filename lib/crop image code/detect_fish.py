import cv2
import os
import numpy as np
from scipy.misc import *
import matplotlib.pyplot as plt

os.chdir('/Users/pengfeiwang/Desktop/f/fisheries/')
fish_cascade = cv2.CascadeClassifier('./output/redfish.xml')

input_path = '/Users/pengfeiwang/Desktop/f/data/train/ALB/'
output_path = '/Users/pengfeiwang/Desktop/a/'
file_name = [input_path + os.listdir(input_path)[i] for i in range(len(os.listdir(input_path)))]

for i in file_name:
	output_name = i.split('/')[-1]
	if output_name != '.DS_Store':
		print output_name
		img = imresize(cv2.imread(i), [64, 64, 3])
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		fish = fish_cascade.detectMultiScale(gray, 1.3, 5)
		for j, (x,y,w,h) in enumerate(fish):
		    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  # color thickness
		    crop_fish = img[y:y+h, x:x+w]
		    cv2.imwrite(os.path.join(output_path,str(j)+'_'+output_name),crop_fish,)
		cv2.imshow('img',img)
		plt.show()		    
