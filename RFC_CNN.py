import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import tensorflow as tf
import cv2 as cv2
import random
import scikitplot as skplt
import joblib
import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# download dataset
DATASET_PATH = 'trash'
data_dir = pathlib.Path(DATASET_PATH)

# define and print categories
CATEGORIES= ["Non-recyclable", "Recyclable"]
Categories = np.array(tf.io.gfile.listdir(str(data_dir)))
print('Categories:', Categories, '\n')

img_size = 180 # set image size (height and width)

dataset = [] # create empty dataset array

# ---Preprocessing images---

# populate array
for c in CATEGORIES:
    path = os.path.join(data_dir, c)
    print(c)
    print('================')
    for item in os.listdir(path): # enter dataset folder
        path2 = os.path.join(path, item)# create path
        print (item)
        #i = 1 # for limiting how many images
        for img in tqdm(os.listdir(path2)):# enter subfolder within each category
            images = cv2.imread(os.path.join(path2, img))# read images 
            images2 = cv2.resize(images, (img_size, img_size))# resiize images
            dataset.append([images2, CATEGORIES.index(c)])# add to dataset array with category index (0,1)
            #i = i + 1 # for limiting how many images
            #if (i > 301): # for limiting how many images
                #break # for limiting how many images

random.shuffle(dataset) # shuffle dataset

# create arrays to hold image and labels
images = []
labels = []

# split images and labels from dataset into each array
for image, label in dataset:
    images.append(image)
    labels.append(label)
    
print("Number of images: " + str(len(images)))

# ---Preprocessing RFC---

images_rfc = np.array(images).reshape(-1, (img_size*img_size*3)) # reshape and convert to numpy array for RFC
labels_rfc = np.array(labels) # convert to numpy array

images_rfc = images_rfc/255.0 # normalise

# split into training and test sets
images_rfc_train, images_rfc_test, labels_rfc_train, labels_rfc_test = train_test_split(images_rfc, labels_rfc, test_size = 0.7)


# -----Random Forest Classifier-----

rfc = RandomForestClassifier(n_jobs = -1, n_estimators=100) # initialise RFC with 100 trees
rfc.fit(images_rfc_train, labels_rfc_train) # train RFC

rfc.score(images_rfc_test, labels_rfc_test) # test RFC
pred = rfc.predict(images_rfc_test) # predict with RFC

# converts 0,1 to proper label from CATEGORIES
def get_labels(label_list):
    true_labels = []
    for label in label_list:
        true_labels.append(str(CATEGORIES[label]))
    return true_labels

# plots confusion matrix
plt.figure(figsize=(12,8))
f, ax=plt.subplots(1,1,figsize=(12,12))
skplt.metrics.plot_confusion_matrix(get_labels(labels_rfc_test), get_labels(pred), normalize='true', ax = ax)
plt.show()

# uses the classification report function from scikit-learn to produce a report with various evaluators
print(classification_report(get_labels(labels_rfc_test), get_labels(pred)))

#joblib.dump(rfc, "./random_forest_300.joblib") # saves RFC model

# -----Convolutional Neural Network-----

# ---Preprocessing CNN---
images_cnn = np.array(images).reshape(-1, img_size, img_size, 3) # reshapes image array to numpy array for CNN
labels_cnn = np.array(labels) # converts to numpy array

print("Tensor shape: " + str(images_cnn.shape)) # prints shape of array
images_cnn = images_cnn.astype('float32') # converts to float32
images_cnn = images_cnn/255.0# normalises

# split into train and test sets
images_cnn_train, images_cnn_test, labels_cnn_train, labels_cnn_test = train_test_split(images_cnn, labels_cnn, test_size = 0.7)

 # changes label array to categorical values for CNN prediction
labels_cnn_train = keras.utils.np_utils.to_categorical(labels_cnn_train)
labels_cnn_test = keras.utils.np_utils.to_categorical(labels_cnn_test)

# ---Model building---

input_shape = images_cnn.shape # defines input shape for CNN

# create a normalisation later (normalises into distribution around 0)
norm_layer =  tf.keras.layers.experimental.preprocessing.Normalization() 


model = models.Sequential([
    layers.Input(shape=input_shape[1:]), # input layer
    norm_layer, # normalises
    layers.Conv2D(128, 3, activation='relu'), # 2D convolutional layer, 32 output filters, kernel size 3, rectified linear unit function activation)
    layers.MaxPooling2D(pool_size=(2, 2)), # pools by going over the image in 2 by 2 'windows'
    layers.Flatten(), # flattens input
    layers.Dense(128, activation="relu"), # dense layer maps every input to every output, 128 output space, relu activation
    layers.Dropout(0.25), # reduces overfitting by dropping a quarter of input units
    layers.Dense(32, activation="relu"), # dense layer maps every input to every output, 32 output space, relu activation
    layers.Dropout(0.25), # reduces overfitting by dropping a quarter of input units
    layers.Dense(2, activation="softmax"), # final dense layers with as many outputs as labels (2), softmax activation
])

model.compile(
    loss='categorical_crossentropy', # computes loss
    optimizer='adam', # optimiser
    metrics=['accuracy']
)

model.summary() # prints model summary

history = model.fit(
    images_cnn_train, # input data
    labels_cnn_train, # input data
    batch_size=64, # batch size
    epochs=10, # training epochs
    validation_data=(images_cnn_test, labels_cnn_test), #validation data
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2), # early stopping to prevent long/unnecessary training times
)

# ---Metrics---

# evaluate train and test accuracy
_, train_acc = model.evaluate(images_cnn_train, labels_cnn_train)
_, test_acc = model.evaluate(images_cnn_test, labels_cnn_test)

print(f'\nTrain accuracy: {train_acc:.0%}')
print(f'Test accuracy: {test_acc:.0%}')

# plot loss curve
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

#model.save('trash_CNN/') # save CNN model