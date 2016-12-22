# The Nature Conservancy Fisheries Monitoring

## Group ID: 201612-13

## Team Member 
+ Kewen Zhang  (kz2246@columbia.edu)
+ Pengfei Wang (pw2406@columbia.edu)
+ Yueying Teng (yt2495@columbia.edu)

## Introduction
This project is inspired by Kaggle.com "the-nature-conservancy-fisheries-monitoring" [Link](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring).

## Environments
+ Operating System: macOS Sierra
+ Language: Python2.7
+ Software: Pyspark, Opencv3

## Outline
### Train Haar-feature-based cascade classifiers
+ [detection_crop_train_fish.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/detection_crop_train_fish.py): Crop fish in the training set
+ [detection_train_haar_classifier.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/detection_train_haar_classifier.py): Train Cascade classifier
+ [detection_generate_negative.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/detection_generate_negative.py): Generate negative instances
+ [detection_clustering_ship.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/detection_clustering_ship.py): Cluster the fishery ships to denoise the raw images
+ [detection_crop_window.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/detection_crop_window.py): Get window of each fishery ships

### BoW Model
+ [classification_get_sift_bow.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/classification_get_sift_bow.py): Extract SIFT features and map into bag-of-words model
+ [classification_mllib_pyspark.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/classification_mllib_pyspark.py): Machine Learning with Pyspark


### Convolutional Neural Network
+ [classification_cnn.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/classification_cnn.py): Construct Convolutional Neural Network model
+ [classification_cnn_LAG.py](https://github.com/Sapphirine/Kaggle_the_Nature_Conservancy_Fisheries_Monitoring/blob/master/lib/classification_cnn_LAG.py): Improve result by re-weighting

## Technical Presentation 
https://youtu.be/dlaS8nACQ7w

