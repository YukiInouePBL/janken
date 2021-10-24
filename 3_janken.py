#!/usr/bin/env python



import os

import sys

import numpy as np

import tensorflow as tf



import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten

from keras.layers import Conv2D,MaxPooling2D

from keras.preprocessing.image import array_to_img,img_to_array,load_img

from keras import backend as K

from sklearn.model_selection import train_test_split

from keras.models import load_model

import time

import random



import RPi.GPIO as GPIO

import picamera



i = 0

label_name = []



label = 'label.txt'



f = open(label,'r')

for line in f:

  line = line.rstrip()

  l = line.rstrip()

  label_name.append(l)

  i = i + 1



NUM_CLASSES = i

IMAGE_SIZE = 28



if __name__ == '__main__':

    test_image = []

    

    # model read

    model = load_model('./model_28.h5')

    model.summary()

    

    # for Raspberry Pi



    GPIO.cleanup()



    #for Raspberry Pi

    GPIO.setmode(GPIO.BCM)

    port1 = 24 # switch



    GPIO.setup(port1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)



    print('Ready!')

    try:

        while True:

            if GPIO.input(port1):

                with picamera.PiCamera() as camera:

                    camera.resolution = (128, 96)

                    camera.start_preview()

                    camera.capture('./tmp.jpg')

                

                img = img_to_array(load_img('./tmp.jpg' , target_size=(28,28)))

                test_image.append(img)

                test_image = np.asarray(test_image)

                test_image = test_image.astype('float32')

                test_image = test_image / 255.0

    

                predictions = model.predict_classes(test_image)

                my_pose = predictions[0]

                #自分の出したものの確認

                if my_pose == 0:
                    print("私の出した手はグーです")
                elif my_pose ==1:
                    print("私の出した手はチョキです")
                else:
                    print("私の出した手はパーです")

                #相手の出したものの確認

                com = random.randint(0,2)

                if com == 0:
                    print("相手の出した手はグーです")
                elif com ==1:
                    print("相手の出した手はチョキです")
                else:
                    print("相手の出した手はパーです")

                #勝敗の判定

                judge = (my_pose - com + 3) % 3

                if judge == 2:
                    print("あなたの勝ちです")
                elif judge == 0:
                    print("あいこです")
                else:
                    print("あなたの負けです")

                test_image = []

            time.sleep(0.01)


    except KeyboardInterrupt:

        GPIO.cleanup()
