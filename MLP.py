import numpy as np
import tensorflow as tf
import random
import pickle

#load data
import mnistLoader

training_data, validation_data, test_data = mnistLoader.load_data_wrapper()
#each is a 50k-tuple of x,y 2-tuples


#hyperparameters

sizes=[784,100,100,100,10]

learningRate=0.001

epochs=10

batch_size=5

# placeholders

x = tf.placeholder('float', [None,784])

y = tf.placeholder('float')

def feedForward(x):
    
    #declare and initialize weights and biases
    
    layers=[]
    
    for i in range(1,len(sizes)):
        
        layers.append({ 'weights' : tf.Variable(tf.random_normal([sizes[i-1],sizes[i]])) , 'biases' : tf.Variable(tf.random_normal([sizes[i]])) })
        
        
    #design forward propagation
    
    n1=tf.add( tf.matmul(x , layers[0]['weights']) , layers[0]['biases'])
    a1=tf.nn.relu(n1)
    
    n2=tf.add( tf.matmul(a1 , layers[1]['weights']) , layers[1]['biases'])
    a2=tf.nn.relu(n2)
    
    n3=tf.add( tf.matmul(a2 , layers[2]['weights']) , layers[2]['biases'])
    a3=tf.nn.relu(n3)
    
    n4=tf.add( tf.matmul(a3 , layers[3]['weights']) , layers[3]['biases'])
    
    return n4


def train_neural_network(x):
    
    prediction = feedForward(x)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        #variables are finally being initialized
        
        for epoch in range(epochs):
            #code below is executed for each epoch once
            
            random.shuffle(training_data)
            
            epoch_loss = 0
            #total Cost of this epoch
                
            #create a list mini_batches
            mini_batches = [training_data[k:k+batch_size] for k in range(0,len(training_data),batch_size)]
            
            for mini_batch in mini_batches:
                #code below is executed for each mini_batch once
                
                batch_x=np.zeros(shape=(len(mini_batch),1))
                batch_y=np.zeros(shape=(len(mini_batch),1))
                
                for i in range(len(mini_batch)):
                    batch_x[0][i]=mini_batch[i][0]
                    batch_y[0][i]=mini_batch[i][1]
                    
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x[1], y: batch_y[1]})
                #the optimizer and cost are only evaluated here
                #feed_dict feeds the placeholder the actual values of this batch
                
                epoch_loss += c
                #adding cost of this batch to the total cost of epoch
                
                
            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)
        
        
        #evaluation
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y})) #TODO
        
        
        
train_neural_network(x)


