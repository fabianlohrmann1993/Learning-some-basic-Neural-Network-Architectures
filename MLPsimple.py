#data & module import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)


#hyperparameters
sizes=[784,500,500,500,10]
batch_size = 100
learningRate=0.01
num_epochs=10

#placeholders
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])


#Build Model
def neural_network_model(data):
   
    layer=[] 
    for l in range(1,len(sizes)):
        
        layer.append( {'weights': tf.Variable(tf.random_normal([sizes[l-1], sizes[l]])) , \
                       'biases':tf.Variable(tf.random_normal([sizes[l]]))} )

    
    
    z1 = tf.add(tf.matmul(data,layer[0]['weights']), layer[0]['biases'])
    a1 = tf.nn.relu(z1)

    z2 = tf.add(tf.matmul(a1,layer[1]['weights']), layer[1]['biases'])
    a2 = tf.nn.relu(z2)

    z3 = tf.add(tf.matmul(a2,layer[2]['weights']), layer[2]['biases'])
    a3 = tf.nn.relu(z3)

    z4 = tf.matmul(a3,layer[3]['weights']) + layer[3]['biases']

    return z4


#Train Model
def train_neural_network(x):
    prediction = neural_network_model(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
    
    
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        #variables are finally being initialized

        for epoch in range(num_epochs): 
            #code below is executed for each epoch once
            epoch_loss = 0 
            #total Cost of this epoch
            for _ in range(int(mnist.train.num_examples/batch_size)): 
                #code below is executed for each batch once
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                #a batch from the training data is loaded into batch_x and...
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                #the optimizer and cost are only evaluated here
                #feed_dict feeds the placeholder the actual values of this batch
                epoch_loss += c 
                #adding cost of this batch to the total cost of epoch

            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)
        
        #Test Model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

#execute
train_neural_network(x)
