#!/usr/bin/env python

# -*- coding: utf-8 -*-



import os

import sys

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt



import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten

from keras.layers import Conv2D,MaxPooling2D

from keras.preprocessing.image import array_to_img,img_to_array,load_img

from keras import backend as K

from sklearn.model_selection import train_test_split

from keras.models import load_model

from keras.callbacks import ModelCheckpoint



# ======= hypter param =====

batch_size = 10

epochs = 50

# ==========================



path=os.getcwd()+'/data/'



class_count = 0

folder_list=os.listdir(path)



for folder in folder_list:



  class_count = class_count+1



NUM_CLASSES = class_count

IMAGE_SIZE = 28



# Loss

def plot_history_loss(fit):

    # Plot the loss in the history

    axL.plot(fit.history['loss'],label="loss for training")

    axL.plot(fit.history['val_loss'],label="loss for validation")

    axL.set_title('model loss')

    axL.set_xlabel('epoch')

    axL.set_ylabel('loss')

    axL.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=10)



# Accuracy

def plot_history_accuracy(fit):

    # Plot the loss in the history

    axR.plot(fit.history['accuracy'],label="loss for training")

    axR.plot(fit.history['val_accuracy'],label="loss for validation")

    axR.set_title('model accuracy')

    axR.set_xlabel('epoch')

    axR.set_ylabel('accuracy')

    axR.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=10)



if __name__ == '__main__':



    count=0

    folder_list = sorted(os.listdir(path))



    train_image = []

    train_label = []

    test_image = []

    test_label = []

    X = []

    Y = []



    label = 'label.txt'

    

    f = open(label, 'w')

    for folder in folder_list:

        subfolder = os.path.join(path,folder)

        file_list = sorted(os.listdir(subfolder))



        filemax = 0



        i = 0



        for file in file_list:



            i = i + 1



            img = img_to_array(load_img('./data/' + folder + '/' + file,target_size=(28,28)))

            X.append(img)

            Y.append(count)

        

        label_name = folder + ' ' + str(count) + '\n'

        f.write(label_name)



        count +=1



    X = np.asarray(X)

    Y = np.asarray(Y)

    X = X.astype('float32')

    X = X / 255.0



    Y = np_utils.to_categorical(Y, NUM_CLASSES)



    train_image, test_image, train_label, test_label = train_test_split(X,Y,test_size=0.20)

    

    f.close()

    print(u'画像読み込み終了')



    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)



    model = Sequential()

    model.add(Conv2D(32,kernel_size=(3,3),

                     activation='relu',

                     padding='same', 

                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation='softmax'))



    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer=keras.optimizers.Adadelta(),

                  metrics=['accuracy']

                  )



    chkpt = './model_28.h5'

    cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, 

                            save_best_only=True, mode='auto')



    history = model.fit(train_image, train_label,

              batch_size=batch_size,

              epochs=epochs,

              verbose=1,

              validation_data=(test_image, test_label),

              callbacks=[cp_cb],

              )



    model.summary()



    score = model.evaluate(test_image, test_label, verbose=0)



    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))



    plot_history_loss(history)

    plot_history_accuracy(history)



    fig.savefig('./loss_acc.png')

    plt.close()
