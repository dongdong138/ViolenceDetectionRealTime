import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Input
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.models import Model

def parser(record):
    keys_to_features = {
        "image_raw": tf.io.FixedLenFeature([], tf.string),
        "label":     tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    image = tf.io.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.reshape(image, shape=[20,224, 224, 3])
    image=tf.image.per_image_standardization(image)
    label = tf.cast(parsed["label"], tf.int32)

    return {'input_1': image}, label

def input_fn(filenames,train,buffer_size=32,batch_size=16):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset   

def train_input_fn():
    return input_fn(filenames=["train.tfrecords"],train=True,buffer_size=128)

def val_input_fn():
    return input_fn(filenames=["val.tfrecords"],train=False,buffer_size=32)

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

callback = tf.keras.callbacks.ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

model.load_weights('model.hdf5')

train = train_input_fn()
val = val_input_fn()
model.summary()
model.fit(train,steps_per_epoch=80,initial_epoch=0,validation_data=val,epochs=50,callbacks=callback)