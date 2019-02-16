from read_file import read_file
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# testing code, in order to check if the data are read correctly.
# image_num, row_num, col_num, fcontent = read_file("data/t10k-images-idx3-ubyte")
# t = np.asarray( fcontent[0:784] ).reshape(28,28) 
# plt.imshow(t, cmap='gray')
# plt.show()

# read the training set labal file:
training_label_num, _, _, training_label_data = read_file( "data/train-labels-idx1-ubyte" )

# read the training set image file:
training_image_num, row_num, col_num, training_image_data = read_file( "data/train-images-idx3-ubyte" )

# read the test data set
test_label_num, _, _, test_label_data = read_file( "data/t10k-labels-idx1-ubyte" )
test_image_num, _, _, test_image_data = read_file( "data/t10k-images-idx3-ubyte" )

# check 1
if training_label_num != training_image_num:
    raise "Training Label number is not equal to image number!"
if test_label_num != test_image_num:
    raise "Testing Label number is not equal to image number!"

# Building Computation Graph:
classes = 10
X_training = tf.placeholder( tf.float32, shape = ( None, row_num*col_num ) )
Y_training = tf.placeholder( tf.float32, shape = ( None, classes ) )
W = tf.Variable( tf.random_normal([ row_num*col_num, classes ]), name="weights" )
b = tf.Variable( tf.random_normal([ classes ]), name="bias" )
logits = tf.matmul( X_training, W ) + b
# hypothesis = tf.nn.softmax( tf.matmul( X_training, W ) + b )
# cost = tf.reduce_mean( -tf.reduce_sum( Y_training * tf.log( hypothesis ), axis = 1  )  )
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( labels = Y_training, logits = (tf.matmul( X_training, W ) + b)  )
cost = tf.reduce_mean( cross_entropy )
# Note: In practice, You should NOT use the commented 2 lines to compute the cross entropy loss function. Because if logit is large, the exp() will be explosive, and hypothesis will be near zero. Therefore, the logarithm will be nan. 
# You can only use tf.nn.softmax_cross_entropy_with_logits_v2 to directly compute the cross entropy.
optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer( learning_rate = 0.003 ).minimize( cost )

# transfer label and image data to proper format
X = np.asarray( training_image_data ).reshape( training_image_num, row_num*col_num  )
Y = np.zeros( (training_label_num, classes) )
for i in range(classes):
    Y[ : , i ] = ( np.asarray(training_label_data) == (i+1) )
X_test = np.asarray( test_image_data ).reshape( test_image_num, row_num*col_num  )

#  feed data and run
batch_size = 512
batch_num = training_label_num // batch_size
y_cost = []
with tf.Session() as sess:
    # initializing of variables
    sess.run( tf.global_variables_initializer() )
    i_iter = 0
    for epoch in range(20):
        for i in range(batch_num):
            cost_val, _ = sess.run( [ cost, optimizer ], feed_dict = {  \
                                    X_training: X[ i*batch_size : (i+1)*batch_size, : ],\
                                    Y_training: Y[ i*batch_size : (i+1)*batch_size, : ] } )
            if ( i_iter % 20 == 0 ):
                print( "step "+str(i_iter)+": cost = "+str(cost_val)+"\n" )
            i_iter +=1
            y_cost.append( cost_val )
    # compute the accuracy
    logits_val = sess.run( logits, feed_dict = {X_training: X_test}  )
    max_index = sess.run( tf.argmax(logits_val, axis = 1) )
    accuracy = sess.run( tf.reduce_mean( tf.cast( (max_index + 1) == test_label_data , dtype=tf.float32 ) ) )
    print( "testing accuracy :"+str(accuracy) )

# plot cost-iter
plt.plot(y_cost)
plt.show() 
