import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

os.chdir('/home/mkeller2013/MTS_NN_Project')

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

def load_data(n_samples):
    data=np.empty((n_samples*15, 115, 110))
    count=0

    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk('Data'):       
        for i, file in enumerate(files):        
            path = os.path.join(root, file) 
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)	
            data[count] = img        
            count=count+1
            print("Loaded file "+str(count)+ " of "+str(15*n_samples)+ " ")
    return data              

def preprocess_data(data):
    temp = np.empty((15, 115, 110))
    diff_data = np.empty((n_samples+1, 14, 115, 110))

    count = 0

    for i, frame in enumerate(data):
        temp[i % 15] = frame

        if not (i+1) % 15:
            count += 1

            differences = get_frame_differences(temp)
            diff_data[count] = differences

    diff_data = diff_data[0:n_samples]
    diff_data = np.moveaxis(diff_data, 1, 3)
    return diff_data

def create_labels():    
    diff_labels = np.empty((n_samples, 2))

    for i in range(0, n_samples):
        if i < n_balls:
            temp = np.array((1, 0))      
        else:
            temp = np.array((0, 1))      

        diff_labels[i] = temp
    return diff_labels

def split_data(data):
    train_data, test_data, train_labels, test_labels=train_test_split(diff_data, diff_labels, test_size=0.3, train_size=0.7, shuffle=True)
    return train_data, test_data, train_labels, test_labels

def set_memory_growth():
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

def create_builtin_model():
    #Xception=tf.keras.applications.xception.Xception(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2) 
    VGG16=tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
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
    #NASNetMobile=tf.keras.applications.nasnet.NASNetMobile(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)
    
    base_model=VGG16
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)    
    predictions = tf.keras.layers.Dense(2, activation= 'softmax')(x)
    adam = tf.keras.optimizers.Adam(lr=0.00001)

    model = tf.keras.models.Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_experiment(model, runs, folds, train_data, train_labels, test_data, test_labels):
    for i in range(0, runs):
        print("Run ", i+1)
        count=1    

        for train_index, val_index in KFold(folds, shuffle=True).split(train_data):
            train_data, val_data=train_data[train_index], train_data[val_index]
            train_labels, val_labels=train_labels[train_index], val_labels[val_index]

            model.save_weights("weights.h5")

            print("Fold ", count)
                                                       
            hist=model.fit(train_data, train_labels, epochs=50, batch_size=16, validation_data = (val_data, val_labels), verbose=1, shuffle=True)
            test_results=model.evaluate(test_data, test_labels, batch_size=16)

            print(test_results)

            model.load_weights("weights.h5")

n_balls = len(os.listdir('Data/Balls/Balls'))
n_strikes = len(os.listdir('Data/Strikes/Strikes'))
n_samples = n_balls + n_strikes

data=load_data(n_samples)
diff_data=preprocess_data(data)
labels=create_labels()
train_data, test_data, train_labels, test_labels=split_data(diff_data)
set_memory_growth()
model=create_builtin_model()
run_experiment(model, 20, 10, train_data, train_labels, test_data, test_labels)

