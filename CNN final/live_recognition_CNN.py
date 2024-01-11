import cv2
import keras
import pathlib
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# classes that the predictor will use to determine the class 
class_labels = [
    "Recyclable",
    "Non-Recycleable",
]

DATASET_PATH = 'CNN final/trash_CNN_300v5_69'
data_dir = pathlib.Path(DATASET_PATH)

# loading the trained Model 
loaded_model = tf.keras.models.load_model(data_dir)

# printing the model summary to check if the model has been loaded successfully 
loaded_model.summary()

# define a video capture object
vid = cv2.VideoCapture(0)

while True:
        # take a real time image of the frame (screen)
        _, frame = vid.read()

        # makes sure image is in RGB
        img = Image.fromarray(frame, 'RGB')

        # resizing into 128x128 because we trained the model with this image size.
        img = img.resize((180,180))

        # convert the image to a numpy array
        image_to_test = image.img_to_array(img)

        # add a fourth dimension to the image (since Keras expects a list of images, not a single image)
        list_of_images = np.expand_dims(image_to_test, axis=0)

        # make a prediction using the model
        results = loaded_model.predict(list_of_images)
        print(results) # print exact probability

        # Since we are only testing one image, we only need to check the first result
        single_result = results[0]
        
        # the single result contains the probability of both classes Recycleable and Non_Recycleable.
        # we will get a likelihood score for both 2  possible classes. 
        # find out which class had the highest score.
        most_likely_class_index = int(np.argmax(single_result))

        # using the index of the given of the most likely class , we use the index to find the class name from the class label array
        class_likelihood = single_result[most_likely_class_index]

        # get the name of the most likely class
        class_label = class_labels[most_likely_class_index]

        # overlay a rectangle and text indicating whether the item is recyclable or not
        cv2.rectangle(frame, (0,0), (300,30), (255,255,255), cv2.FILLED)
        cv2.putText(frame, "This item is {}".format(class_label), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0))

        cv2.imshow("Convolutional Neural Network", frame)
        key=cv2.waitKey(1)

        # exit program once 'Q' key is clicked 
        if key == ord('q'):
            break

# ends current video capture
vid.release()
cv2.destroyAllWindows()