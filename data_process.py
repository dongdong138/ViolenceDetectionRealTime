import cv2 
import numpy as np
import tqdm
import glob
from random import shuffle
import tensorflow as tf
import sys

def vid_bright(vid):
    vid=tf.image.adjust_brightness(vid, delta=0.5)
    return vid
def vid_flip1(vid):
    vid=tf.image.flip_left_right(vid)
    return vid
def vid_flip2(vid):
    vid=tf.image.flip_up_down(vid)
    return vid
def vid_con(vid):
    vid=tf.image.adjust_contrast(vid, 5)
    return vid
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_video(addr,aug):
    vidcap = cv2.VideoCapture(addr)
    count = 0
    frames=[]
    success=True
    while success and count<20:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*250))
        success,frame= vidcap.read()
        frame = cv2.resize(frame,(224,224), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (224,224,3))
        frames.append(frame)  
        count += 1

    frames = np.array(frames)
    vidcap.release()
    if aug==1:
        aug_vid=vid_bright(frames)
        return frames,aug_vid
    if aug==2:
        aug_vid=vid_flip1(frames)
        return frames,aug_vid
    if aug==3:
        aug_vid=vid_flip2(frames)
        return frames,aug_vid
    if aug==4:
        aug_vid=vid_con(frames)
        return frames,aug_vid
    else:
        return frames,frames
    
def createDataRecord(out_filename, addrs, labels,aug=0,train=False):
    writer = tf.io.TFRecordWriter(out_filename)
    for i in tqdm.tqdm(range(len(addrs))):
        vid,aug_vid = load_video(addrs[i],aug)
        vid = np.uint8(vid)
        label = labels[i]
        feature = {
            'image_raw': _bytes_feature(vid.tostring()),
            'label': _int64_feature(label)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())
        if aug != 0:
            aug_vid = np.uint8(aug_vid)
            label = labels[i]
            feature = {
                'image_raw': _bytes_feature(aug_vid.tostring()),
                'label': _int64_feature(label)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        if i==399 and train==True:
            aug=2
        if i==799 and train==True:
            aug=3
        if i==1199 and train==True:
            aug=4
        if i>1599 and train==True:
            aug=0
    writer.close()
    sys.stdout.flush()

train_path = 'RWF-2000/train/*/*.avi'
test_path='RWF-2000/val/*/*.avi'

addrs_train = glob.glob(train_path)
addrs_test = glob.glob(test_path)
labels_train = [0 if addr.split('/')[-2]=='NonFight' else 1 for addr in addrs_train]  # 0 = no fight, 1 = fight
labels_test = [0 if addr.split('/')[-2]=='NonFight' else 1 for addr  in addrs_test]

c = list(zip(addrs_train, labels_train))
shuffle(c)
addrs_train, labels_train = zip(*c)

c = list(zip(addrs_test, labels_test))
shuffle(c)
addrs_test, labels_test = zip(*c)

createDataRecord('train.tfrecords', addrs_train, labels_train,aug=1,train=True)
createDataRecord('val.tfrecords',addrs_test, labels_test)