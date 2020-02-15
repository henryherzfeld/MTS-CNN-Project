import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

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
    
    data=data.astype('float32')    
    data /= 255

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

def create_single_model(batch_size, train_data, val_data, train_labels, val_labels):
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
    adam = tf.keras.optimizers.Adam(lr=math.sqrt(batch_size)*(10**-5))
    model = tf.keras.models.Model(inputs = base_model.input, outputs = predictions)
    mc=tf.keras.callbacks.ModelCheckpoint('Models/best_model-VGG16.hdf5', save_best_only=True, monitor='val_accuracy')    
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.save_weights('Models/initial_weights.h5')

    model.fit(train_data, train_labels, epochs=30, batch_size=batch_size, validation_data=(val_data, val_labels), shuffle=True, callbacks=[mc])

    model=tf.keras.models.load_model('Models/best_model-VGG16.hdf5')
    # evaluate the model
    test_loss, test_acc = model.evaluate(val_data, val_labels, batch_size=batch_size, verbose=0)
    return model, test_acc

def ensemble_predictions(members, test_data):
	# make predictions
	predictions = [model.predict(test_data) for model in members]
	predictions = np.array(predictions)
	# sum across ensemble members
	summed = np.sum(predictions, axis=0)
	# argmax across classes
	result = np.argmax(summed, axis=1)
	return result
 
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, test_data, test_labels):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    en_predictions = ensemble_predictions(subset, test_data)
    en_predictions = tf.keras.utils.to_categorical(en_predictions)
    # calculate accuracy
    return accuracy_score(test_labels, en_predictions)

n_balls = len(os.listdir('Data/Balls/Balls'))
n_strikes = len(os.listdir('Data/Strikes/Strikes'))
n_samples = n_balls + n_strikes

data=load_data(n_samples)
diff_data=preprocess_data(data)
labels=create_labels()
set_memory_growth()
train_data, test_data, train_labels, test_labels=train_test_split(diff_data, labels, test_size=0.3, shuffle=True)

n_folds = 10
kfold = KFold(n_folds, True, 1)
# cross validation estimation of performance
scores, members = list(), list()

for train_index, test_index in kfold.split(train_data):
	# select samples
	train_data_cv, train_labels_cv = train_data[train_index], train_labels[train_index]
	val_data, val_labels = train_data[test_index], train_labels[test_index]
	# evaluate model
	model, test_acc = create_single_model(4, train_data_cv, val_data, train_labels_cv, val_labels)
	print('>%.3f' % test_acc)
	scores.append(test_acc)
	members.append(model)
# summarize expected performance
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, n_folds+1):
	ensemble_score = evaluate_n_members(members, i, test_data, test_labels)
	_, single_score = members[i-1].evaluate(test_data, test_labels, verbose=0)
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# plot score vs number of ensemble members
print('Accuracy %.3f (%.3f)' % (np.mean(single_scores), np.std(single_scores)))
x_axis = [i for i in range(1, n_folds+1)]
plt.plot(x_axis, single_scores, marker='o', linestyle='None')
plt.plot(x_axis, ensemble_scores, marker='o')
plt.show()
