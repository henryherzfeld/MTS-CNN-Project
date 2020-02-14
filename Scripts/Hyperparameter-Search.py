import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():    
    def load_data():    
        n_balls = len(os.listdir('/home/mkeller2013/MTS_NN_Project/Data/Balls/Balls'))
        n_strikes = len(os.listdir('/home/mkeller2013/MTS_NN_Project/Data/Strikes/Strikes'))
        n_samples = n_balls + n_strikes
        
        data=np.empty((n_samples*15, 115, 110))
        count=0

        # traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk('/home/mkeller2013/MTS_NN_Project/Data'):       
            for i, file in enumerate(files):        
                path = os.path.join(root, file) 
                img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)	
                data[count] = img        
                count=count+1
                print("Loaded file "+str(count)+ " of "+str(15*n_samples)+ " ")
        return data, n_samples, n_balls, n_strikes

    def label_data(n_samples, n_balls, n_strikes):
        diff_labels = np.empty((n_samples, 2))

        for i in range(0, n_samples):
            if i < n_balls:
                temp = np.array((1, 0))      
            else:
                temp = np.array((0, 1))      

            diff_labels[i] = temp
        return diff_labels

    def get_frame_differences(frames, method="default", thresholding=False):
        acc = []
        i = 0
        n = len(frames)
                
        if method == "reverse":
            while i < n-1:
                acc.append(frames[i]-frames[i+1])
                i += 1

        elif method == "multiframe":
            while i < n-2:
                acc.append(cv2.absdiff(frames[i], frames[i+2]))
                i += 1
                    
        else:
            while i < n-1:
                acc.append(frames[i+1]-frames[i])
                i += 1
                        
        if thresholding:
            acc2 = []
            for frame in acc:
                _, thresh_im = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
                acc2.append(thresh_im)
            return acc2
                    
        return acc    

    def create_difference_data_array(data, n_samples):
        temp = np.empty((15, 115, 110))
        diff_data = np.empty((n_samples+1, 14, 115, 110))

        count = 0

        for i, frame in enumerate(data):
            temp[i % 15] = frame

            if not (i+1) % 15:
                count += 1

                differences = get_frame_differences(temp, method='default')
                diff_data[count] = differences

        diff_data = diff_data[0:n_samples]

        diff_data = np.moveaxis(diff_data, 1, 3)
        return diff_data
    data, n_samples, n_balls, n_strikes=load_data()
    
    diff_labels=label_data(n_samples, n_balls, n_strikes)
    
    diff_data=create_difference_data_array(data, n_samples)

    train_data, test_data, train_labels, test_labels=train_test_split(diff_data, diff_labels, test_size=0.3, random_state=12345, shuffle=True)
    train_data, val_data, train_labels, val_labels=train_test_split(train_data, train_labels, test_size=0.2, random_state=12345, shuffle=True)

    train_data=train_data.astype('float32')
    val_data=val_data.astype('float32')
    test_data=test_data.astype('float32')
    train_data /= 255
    val_data /= 255
    test_data /= 255
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def model(train_data, train_labels, val_data, val_labels):
    #Xception=tf.keras.applications.xception.Xception(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2) 
    #VGG16=tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #VGG19=tf.keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #ResNet50=tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #ResNet101=tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #ResNet152=tf.keras.applications.resnet_v2.ResNet152V2(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #Inception=tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #InceptionResNet=tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #MobileNet=tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #DenseNet121=tf.keras.applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #DenseNet169=tf.keras.applications.densenet.DenseNet169(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #DenseNet201=tf.keras.applications.densenet.DenseNet201(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    #NASNetLarge=tf.keras.applications.nasnet.NASNetLarge(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    NASNetMobile=tf.keras.applications.nasnet.NASNetMobile(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    
    base_model=NASNetMobile

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)    
    predictions = tf.keras.layers.Dense(2, activation= 'softmax')(x)
    model = tf.keras.models.Model(inputs = base_model.input, outputs = predictions)
    optim=tf.keras.optimizers.Adam(lr={{choice([10**-i for i in range (1, 6)])}})

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)
    
    model.fit(train_data, train_labels, batch_size=1, epochs=10, validation_data=(val_data, val_labels), shuffle=True)
    score, acc = model.evaluate(val_data, val_labels)
    print('Val accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}    
    
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

train_data, train_labels, val_data, val_labels, test_data, test_labels=data()

os.chdir('/home/mkeller2013/MTS_NN_Project/Scripts')

best_run, best_model=optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials(), notebook_name=None)
print(best_run)
print(best_model.evaluate(test_data, test_labels, batch_size=1))