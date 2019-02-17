

#                                     -- Preparing Data --

#importing modules for data preparation
import cv2
import numpy as np
import os
import random

# convert the labels into one_hot vectors
#the training data is just named image files without groundtruth labels
#we derive the labels from the filenames
def label_img(img):
    word_label = img.split(".")[-3]
    # eg: "dog.93.png" we split this into ["dog","93","png"], hence the [-3] index is "dog"
    if word_label = "cat":
        return [1,0]
        
    elif word_label == "dog":
        return [0,1]


#resize the images (they have different sizes and aspect ratios)
#50x50 is our desired resolution which is quite good for 2016 CNN standards
IMG_SIZE = 50

#specify the directory where to find the data
DATA_DIR = "/home/user/.../Downloads/PythonProjects/catsVSdogs/data"

#we write a function that loads data into a variable
def create_data():
    
    #make a list to append the data as 2-tuples (image, label)
    data = []
    
    #os.listdir(path) returns a list where the elements are the entries in the directory given by 'path'
    #in our case the elements are image files
    for img in  os.listdir(DATA_DIR):
        
        #we call the make labels function here
        label = label_img(img)
        
        #what does path join?
        path = os.path.join(DATA_DIR,img)
        
        #what does cv2.imread(path,__)
        #we make it greyscale to simplify 3 channel matrix into simple 2D matrix
        img = cv2.imread(path,cv2.IMREAD_GREYSCALE)
        
        #here we actually resize the image with the resize specified above
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        #now we append the list with the 2-tuples
        data.append([np.array(img),np.array(label)])
    
    #we shuffle the data (for epochs?)
    random.shuffle(data)
    return data
    

#we load data into a variable
data = create_data()

#we split data into training data and testing data
#the last 1000 of the 25000 images will be test data
train = [:-1000]
test = [-1000:]

#split the training data into input and label
#also reshape the input vector back into a 2D matrix image
X = [i[0] for i in train].reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]


#split the test data into input and label
test_x = [i[0] for i in test]
test_y = [i[1] for i in test]






#                                           -- Designing the Model --

#import modules
import tflearn

#hyperparamters
learningRate = 0.001


#define input layer (input_data refers to input layer) ( name="Input" is used in model.fit)
convnet = tflearn.layers.core.input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name="input")

#define first conv layer, and pooling layer
convnet = tflearn.layers.conv.conv_2d(convnet, 32, 2 , activation="relu")
convnet = tflearn.layers.conv.max_pool_2d(convnet,2)

#define second conv layer
convnet = tflearn.layers.conv.conv_2d(convnet, 64, 2 , activation="relu")
convnet = tflearn.layers.conv.max_pool_2d(convnet,2)

#unflatten (why is there a relu here?)
convnet = tflearn.layers.core.fully_connected(convnet, 1024, activation="relu")

#implement a dropout layer
convnet = tflearn.layers.core.dropout(convnet, 0.8)

#define output layer (which is just a fully connected) (are there no weights?)
convnet = tflearn.layers.core.fully_connected(convnet, 2, activation="softmax")

#define cost function and gradient descent method (name="targets" is used in model.fit)
convnet = tflearn.layers.estimator.regression(convnet, optimizer="adam", learning_rate=learningRate, loss="categorical_crossentropy", name="targets")

#declare a model object
model = tflearn.DNN(convnet , tensorboard_dir = "log")
#tensorboard will visualise the model
#tensorboard_dir is the directory were tensorboard will be saved

#let's give our declared model a name
MODEL_NAME = dogsVScats.model


#                                   -- TRAINING --

#feed the data variables as arguments into the model and train it
model.fit({"input":X} , {"targets":Y} , n_epoch=5 , validation_set = ({"input":test_x} , {"targets":test_y}) , snapshot_step=500, show_metric =True, run_id="MODEL_NAME")
#run_id is how we find the model in tensorboard


#tensorboard
#foo is just because you need some name
tensorboard --logdir=foo:/home/user/.../Downloads/PythonProjects/catsVSdogs/log
#this line will give you an web adress, when you put that into your browser
#it will open a tensorboard website with your results visualized


