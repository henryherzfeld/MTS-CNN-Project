#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('/home/mkeller2013/MTS_NN_Project')


# In[2]:


import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
import cv2

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


# In[3]:


def get_frame_differences(frames):
  n = len(frames)

  first_frame = frames[0]
  acc = np.empty((n-1, *first_frame.shape))

  for i, frame in enumerate(frames[1:]):
    diff=cv2.absdiff(first_frame, frame)    
    #if thresholding:
      #ret, diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)

    acc[i] = diff
    first_frame = frame

  return acc


# In[4]:


n_balls = len(os.listdir('Data/Balls/Balls'))
n_strikes = len(os.listdir('Data/Strikes/Strikes'))
n_samples = n_balls + n_strikes

print(n_balls)
print(n_strikes)
print(n_samples)


# In[5]:


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


# In[6]:


diff_labels = np.empty((n_samples, 2))

for i in range(0, n_samples):
    if i < n_balls:
      temp = np.array((1, 0))      
    else:
      temp = np.array((0, 1))      

    diff_labels[i] = temp


# In[7]:


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

for i in range(0, len(diff_data)):
    diff_data[i]=diff_data[i]/255


# In[8]:


diff_data = np.moveaxis(diff_data, 1, 3)


# In[9]:


from sklearn.model_selection import KFold

#train_data, test_data, train_labels, test_labels=train_test_split(diff_data, diff_labels, test_size=0.3, train_size=0.7)
#train_data, val_data, train_labels, val_labels=train_test_split(train_data, train_labels, test_size=0.1, train_size=0.9)


# In[10]:


base_model=applications.xception.Xception(include_top=False, weights=None, input_tensor=None, input_shape=(115, 110, 14), pooling=None, classes=2)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
adam = Adam(lr=0.00001)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


# In[11]:


from sklearn.metrics import confusion_matrix

runs=10
epochs=50
#train_acc_array=np.empty(epochs)
test_acc_array=np.empty(epochs)

folds=5

max_test_array=np.empty((10, folds))

for i in range(0, runs):
    print("Run ", i+1)
    count=1    

    for train_index, test_index in KFold(folds, shuffle=True).split(diff_data):
        train_data, test_data=diff_data[train_index], diff_data[test_index]
        train_labels, test_labels=diff_labels[train_index], diff_labels[test_index]

        model.save_weights("weights.h5")

        print("Fold ", count)

        for j in range(0, epochs):
            print("Epoch ", j+1)       

            hist=model.fit(train_data, train_labels, epochs=1, batch_size=1, validation_data = None)

            #train_acc_array[j]=hist.history['accuracy']

            test_results=model.evaluate(test_data, test_labels, batch_size=1)

            print("Test accuracy: ", test_results[1])  

            test_acc_array[j]=test_results[1]

            predictions=np.empty((len(test_data), 2))

            predictions_output=model.predict(test_data)

            for k in range (0, len(predictions_output)):
                if predictions_output[k][0]>predictions_output[k][1]:
                    predictions[k]=np.array((1, 0))
                else:
                    predictions[k]=np.array((0, 1))

            print(confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1)))

        max_test_array[i][count-1]=np.max(test_acc_array)
        count=count+1
    
        model.load_weights("weights.h5")


# In[ ]:


#plt.plot(train_acc_list)
#plt.plot(test_acc_list)
#plt.title('VGG-16-5 Fold Cross validation-model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.gcf().savefig("Results/VGG-16-5FCV.png")
#plt.show()


# In[ ]:


best_epoch_df=pd.DataFrame(max_test_array)
best_epoch_df.to_csv('Results/Tables/Xception-5FCV.csv')

