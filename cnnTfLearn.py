#import modules
import tflearn
#import data
import tflearn.datasets.mnist as mnist

#load data into variables
X , Y , test_x , test_y = mnist.load_data(one_hot=True)

#reshape 784 input-vectors into original 28x28 matrices
X = X.reshape ([-1 , 28 , 28 ,1])
test_x = test_x.reshape ([-1 , 28 , 28 , 1])

#define input layer (input_data refers to input layer) ( name="Input" is used in model.fit)
convnet = tflearn.layers.core.input_data(shape=[None,28,28,1],name="input")

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
convnet = tflearn.layers.core.fully_connected(convnet, 10, activation="softmax")

#define cost function and gradient descent method (name="targets" is used in model.fit)
convnet = tflearn.layers.estimator.regression(convnet, optimizer="adam", learning_rate=0.01, loss="categorical_crossentropy", name="targets")

#declare a model object
model = tflearn.DNN(convnet)

#train the model and feed the data variables as arguments into the model
model.fit({"input":X} , {"targets":Y} , n_epoch=3 , validation_set = ({"input":test_x} , {"targets":test_y}) , snapshot_step=500, show_metric =True, run_id="mnist")

