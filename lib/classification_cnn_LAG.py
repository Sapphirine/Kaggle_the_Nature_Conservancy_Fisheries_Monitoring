import os
import glob
import cv2
import datetime
import numpy as np
import pandas as pd
import time
import warnings

from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

class LAG_CNN():
    def __init__(self):
        # Read train data
        train_data = []
        train_id = []
        train_target = []
        start_time = time.time()

        print('Read train images')
        folders = ['DOL', 'other']
        for fld in folders:
            index = folders.index(fld)
            print('Load folder {} (Index: {})'.format(fld, index))
            path = os.path.join('/Users/Kevin/Desktop/fish_model/red_crop_train', fld, '*.jpg')
            files = glob.glob(path)
            for fl in files:
                flbase = os.path.basename(fl)
                img = get_im_cv2(fl)
                train_data.append(img)
                train_id.append(flbase)
                train_target.append(index)
            print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))

        # Normalize train data

        print('Convert to numpy...')
        train_data = np.array(train_data, dtype=np.uint8)
        train_target = np.array(train_target, dtype=np.uint8)

        print('Reshape...')
        train_data = train_data.transpose((0, 3, 1, 2))

        print('Convert to float...')
        train_data = train_data.astype('float32')
        train_data = train_data / 255
        train_target = np_utils.to_categorical(train_target, 8)

        print('Train shape:', train_data.shape)
        print(train_data.shape[0], 'train samples')

        # CNN Model Building
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, 48, 48), dim_ordering='th'))
        model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
        model.add(Flatten())
        model.add(Dense(96, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        # CNN Training
        nfolds = 20
        batch_size = 16
        nb_epoch = 30
        random_state = 51
        yfull_train = dict()
        kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
        num_fold = 0
        sum_score = 0
        models = []
        for train_index, test_index in kf:
            X_train = train_data[train_index]
            Y_train = train_target[train_index]
            X_valid = train_data[test_index]
            Y_valid = train_target[test_index]
            num_fold += 1
            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            print('Split train: ', len(X_train), len(Y_train))
            print('Split valid: ', len(X_valid), len(Y_valid))
            callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),]
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
                    callbacks=callbacks)
            predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
            score = log_loss(Y_valid, predictions_valid)
            print('Score log_loss: ', score)
            sum_score += score * len(test_index)
            # Store valid predictions
            for i in range(len(test_index)):
                yfull_train[test_index[i]] = predictions_valid[i]
            models.append(model)

        score = sum_score/len(train_data)
        print("Log_loss train independent avg: ", score)
        info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)


        # Read test data
        path = os.path.join('/Users/Kevin/Desktop/fish_model/red_crop_test', '*.jpg')
        files = sorted(glob.glob(path))

        test_data = []
        test_id = []
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            test_data.append(img)
            test_id.append(flbase)

        # Nomilize test data
        start_time = time.time()
        test_data, test_id = load_test()

        test_data = np.array(test_data, dtype=np.uint8)
        test_data = test_data.transpose((0, 3, 1, 2))

        test_data = test_data.astype('float32')
        test_data = test_data / 255

        print('Test shape:', test_data.shape)
        print(test_data.shape[0], 'test samples')
        print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))

        # CNN Prediction
        batch_size = 16
        num_fold = 0
        yfull_test = []
        test_id = []
        nfolds = len(models)
        for i in range(nfolds):
            model = models[i]
            num_fold += 1
            print('Start KFold number {} from {}'.format(num_fold, nfolds))
            test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
            yfull_test.append(test_prediction)
        test_res = merge_several_folds_mean(yfull_test, nfolds)
        info_string = 'loss_' + info_string \
                    + '_folds_' + str(nfolds)
        create_submission(test_res, test_id, info_string)

    
    def get_im_cv2(self,path):
        img = cv2.imread(path)
        resized = cv2.resize(img, (48, 48), cv2.INTER_LINEAR)
        return resized

    def create_submission(self, predictions, test_id, info):
        result1 = pd.DataFrame(predictions, columns=['LAG', 'other'])
        result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
        now = datetime.datetime.now()
        sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
        result1.to_csv(sub_file, index=False)

    def dict_to_list(self, d):
        ret = []
        for i in d.items():
            ret.append(i[1])
        return ret

    def merge_several_folds_mean(self, data, nfolds):
        a = np.array(data[0])
        for i in range(1, nfolds):
            a += np.array(data[i])
        a /= nfolds
        return a.tolist()

    def get_validation_predictions(self, train_data, predictions_valid):
        pv = []
        for i in range(len(train_data)):
            pv.append(predictions_valid[i])
        return pv
    
if __name__=='__main__':
    LAG_CNN()