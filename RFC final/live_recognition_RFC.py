import cv2
from pathlib import Path
from keras.preprocessing import image
import joblib as joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# classes that the predictor will use to determine the class 
class_labels = [
    "Recyclable",
    "Non-Recycleable",
]

# loading the trained Model 
loaded_rf = joblib.load('RFC final/random_forest_all.joblib')

# define a video capture object
vid = cv2.VideoCapture(0)

while True:
        # take a real time image of the frame (screen)
        _, frame = vid.read()
        img = Image.fromarray(frame, 'RGB')

        # resizing into 128x128 because we trained the model with this image size.
        img = img.resize((180,180))

        # convert the image to a numpy array
        image_to_test = image.img_to_array(img)
        
        # reshapes array into 2 dimensional for RFC
        images = np.array(image_to_test).reshape(-1, (180*180*3))
        images = images/255.0

        # make a prediction using the model
        results = loaded_rf.predict(images)
        print(results) # prints identified class number

        # since we are only testing one image, we only need to check the first result
        single_result = results[0]

        # get the name of the most likely class
        class_label = class_labels[single_result]

        # overlay a rectangle and text indicating whether the item is recyclable or not
        cv2.rectangle(frame, (0,0), (300,30), (255,255,255), cv2.FILLED)
        cv2.putText(frame, "This item is {}".format(class_label), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0))

        cv2.imshow("Random Forest Classifier", frame)
        key=cv2.waitKey(1)

        # exit program once 'Q' key is clicked 
        if key == ord('q'):
            break

# ends current video capture
vid.release()
cv2.destroyAllWindows()