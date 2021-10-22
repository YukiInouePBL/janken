import glob

import time

import os

import io



# for Raspberry Pi

import RPi.GPIO as GPIO

import picamera



GPIO.cleanup()



#for Raspberry Pi

GPIO.setmode(GPIO.BCM)

port1 = 17 # gu

port2 = 27 # choki

port3 = 22 # pa



GPIO.setup(port1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GPIO.setup(port2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GPIO.setup(port3, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)



count = 1

flag = 0



# File remove

file_list = glob.glob('./data/0_gu/*')

for file in file_list:

    os.remove(file)

file_list = glob.glob('./1_choki/*')

for file in file_list:

    os.remove(file)

file_list = glob.glob('./data/2_pa/*')

for file in file_list:

    os.remove(file)



print('Ready!')



try:

    while True:

        sc = str(count)

        ssc = sc.zfill(4)

        #GPIOの17,27,22がオンになったら、画像を取り込んで認識を開始

        if GPIO.input(port1):

            label = '0_gu'

            flag = 1

        elif GPIO.input(port2):

            label = '1_choki'

            flag = 1

        elif GPIO.input(port3):

            label = '2_pa'

            flag = 1

            

        if flag ==1 :

            print(ssc + ':' + label)

            with picamera.PiCamera() as camera:

                    camera.resolution = (128, 96)

                    camera.start_preview()

                    camera.capture('./data/'+label+'/'+label+ssc+'.jpg')

            count +=1

            flag = 0



        time.sleep(0.01)



except KeyboardInterrupt:

    GPIO.cleanup()
