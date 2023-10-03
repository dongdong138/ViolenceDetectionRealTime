import cv2 
from threading import Thread
import numpy as np
import os 
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Input
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.models import Model

def SELayer(layer):
    pool=tf.math.reduce_mean(layer, axis=[2,3,4], keepdims=True, name=None)
    flatt=Flatten()(pool)
    fc1=tf.keras.layers.Dense((int(layer.shape[1])/2), activation='relu',kernel_initializer='he_uniform')(flatt)
    fc2=tf.keras.layers.Dense(layer.shape[1], activation='relu',kernel_initializer='he_uniform')(fc1)
    sig=tf.keras.layers.Activation('sigmoid')(fc2)
    mul=tf.keras.layers.Multiply()([layer,tf.reshape(sig,shape=(-1,layer.shape[1],1,1,1))])
    return mul

def conv3Dnet_a(layer,k_size,pool_size):
    conv=Conv3D(kernel_size=k_size,filters=5,activation='relu',kernel_initializer='he_uniform')(layer)
    drop1=Dropout(0.1)(conv)
    norm=tf.keras.layers.BatchNormalization()(drop1)
    act=tf.keras.layers.Activation('relu')(norm)
    pool=tf.keras.layers.MaxPooling3D(pool_size=pool_size)(act)
    return pool

def conv3Dnet_b(layer,k1_size,k2_size,pool_size):
    conv1=Conv3D(kernel_size=k1_size,filters=3,activation='relu',kernel_initializer='he_uniform')(layer)
    drop1=Dropout(0.1)(conv1)
    norm=tf.keras.layers.BatchNormalization()(drop1)
    act1=tf.keras.layers.Activation('relu')(norm)
    conv2=Conv3D(kernel_size=k2_size,filters=3,activation='relu',kernel_initializer='he_uniform')(act1)
    drop2=Dropout(0.1)(conv2)
    act2=tf.keras.layers.Activation('relu')(drop2)
    pool=tf.keras.layers.MaxPooling3D(pool_size=pool_size)(act2)
    return pool

tf.keras.backend.clear_session()
input=Input((20,224,224,3))
con_net_1=conv3Dnet_a(input,k_size=(3,5,5),pool_size=(2,2,2))
se_1=SELayer(con_net_1)
con_net_2=conv3Dnet_a(se_1,k_size=(2,3,3),pool_size=(2,2,2))
se_2=SELayer(con_net_2)
con_net_3=conv3Dnet_a(se_2,k_size=(1,2,2),pool_size=(1,2,2))
se_3=SELayer(con_net_3)
con_net_4=conv3Dnet_b(se_3,k1_size=(1,2,2),k2_size=(1,2,2),pool_size=(1,2,2))
se_4=SELayer(con_net_4)
con_net_5=conv3Dnet_b(se_4,k1_size=(1,2,2),k2_size=(1,2,2),pool_size=(1,2,2))
se_5=SELayer(con_net_5)
flatt=Flatten()(se_5)
drop3=Dropout(0.4)(flatt)
fc1=tf.keras.layers.Dense(512, activation='relu',kernel_initializer='he_uniform')(drop3)
drop4=Dropout(0.4)(fc1)
fc2=tf.keras.layers.Dense(100, activation='relu',kernel_initializer='he_uniform')(drop4)
drop5=Dropout(0.3)(fc2)
out=tf.keras.layers.Dense(1, activation='sigmoid')(drop5)
model=Model(inputs=input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
                loss='binary_crossentropy',metrics=['accuracy'])

model.load_weights('model.hdf5')

def prediction():
    global pred_var
    global org
    global color
    k=0
    run=True
    while run:
        try:
            frames=np.load('frame'+str(k)+'.npy')
            image=tf.image.per_image_standardization(frames)
            image=tf.reshape(image,shape=(1,20,224,224,3))
            pred=model.predict(image)
            print(pred[0])
            if pred[0]>0.4:
                pred_var='VIOLENCE'
                color = (0, 0,255)
                org = (120, 50)
            else:
                pred_var='NO VIOLENCE'
                color = (255, 0, 0)
                org = (100, 50)
            os.remove('frame'+str(k)+'.npy')
            k=k+1
        except:
            continue    
    cv2.destroyAllWindows()

import cv2
import numpy as np
from threading import Thread

def load_video():
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1
    thickness = 4
    count=0
    s=0
    vid = cv2.VideoCapture('video1.mp4')
    frames=[]
    success=True
    while success :
        success,frame= vid.read()
        image = frame
        if success==False:
            break
        frame = cv2.resize(frame,(224,224), interpolation=cv2.INTER_AREA)
        frame = np.reshape(frame, (224,224,3))
        image = cv2.putText(image, pred_var, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Video',image )
        if count%3==0:
            frames.append(frame)
        count=count+1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        if len(frames)==20:
            np.save('frame'+str(s)+'.npy',np.array(frames))
            frames=[]
            s=s+1
    vid.release()
    cv2.destroyAllWindows()

pred_var='NO VIOLENCE'
org = (100, 50) 
color = (255, 0, 0)
t2= Thread(target = load_video)
t1= Thread(target = prediction)
t2.start()
t1.start()
cv2.destroyAllWindows()