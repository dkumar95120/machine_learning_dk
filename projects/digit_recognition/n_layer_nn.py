# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

# helper function to compute accuracy given predictions and labels
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
# helper function to generate a convolution of size nxm with a stride of nxm for input data and its weights
def conv2d(x, w, n, m):
  return tf.nn.conv2d(x, w, strides=[1, n, m, 1], padding='SAME')

# helper function to generate a maxpool rendition of the input data
def maxpool(x, n, m):
  return tf.nn.max_pool(x, ksize=[1, n, m, 1], strides=[1, n, m, 1], padding='SAME')

#helper function to build a weight variable of given sha  pe
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#helper function to build 
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def create_logits (X, weights, biases, dropout=False):
    #initialize relu_logits as the input dataset
    nlayer = len(weights.keys())
    cur_logits = X
    for layer in range(nlayer):
        logits = tf.matmul(cur_logits, weights[layer]) + biases[layer]
        # Transform logits computed from layer 1 using relu function to compute logits for layer2
        if layer < nlayer-1:
            cur_logits = tf.nn.relu(logits)
        # Dropout on hidden layers only
            if dropout:
                cur_logits = tf.nn.dropout(cur_logits, 0.75)

    # Apply activation function to transform logits from the last layer to generate predictions for training samples
    y_pred = tf.nn.softmax(logits)
    return y_pred, logits

# helper function to flatten 2d images to 1d tensor
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)




#Create a single hidden layer neural network using RELU and 1024 nodes
def n_layer_nn (X_train, y_train, X_valid, y_valid, X_test, y_test, image, num_classes, 
  batch_size=128, num_nodes=[1024], num_samples=0, num_steps = 1001, print_steps=100, dropout=False):
  beta = .01
  init_learning_rate = 0.1
  if num_samples ==0:
      num_samples = y_train.shape[0]

  graph = tf.Graph()
  #build nodes array and append it with the number of classes which is final number of nodes
  nodes_list = num_nodes
  nodes_list.append(num_classes)

  with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_X_train = tf.placeholder(tf.float32, shape=(batch_size, image[0] * image[1]))
      tf_y_train = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

      tf_X_valid = tf.constant(X_valid)
      tf_X_test  = tf.constant(X_test)
      # save weights and biases in a dictionary for each layer
      weights={}
      biases ={}
      # first dimesion of the weights will be the number of features in the training dataset which is the image size
      dim1 = image[0] * image[1]
      # build weights and biases for each layer
      for i, nodes in enumerate(nodes_list):
          weights[i] = tf.Variable(tf.truncated_normal([dim1, nodes]))
          biases[i]  = tf.Variable(tf.zeros([nodes]))
          w_l2_loss = tf.nn.l2_loss(weights[i])
          regularizers = regularizers + w_l2_loss if  i else w_l2_loss
          # Subsequent layer first dimension would the number of nodes in the previous layer
          dim1 = nodes

      y_train_pred, logits = create_logits (tf_X_train, weights, biases, dropout)

      # compute the loss comparing logits from layer 2 and the training labels
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y_train, logits=logits))
      # Loss function with L2 Regularization with beta
      loss = tf.reduce_mean(cross_entropy + beta * regularizers)

      # Optimize using Gradient Descent
      global_step = tf.Variable(0)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps=100000, decay_rate=0.96, staircase=True)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
      
      # Transform validation samples using the same transformation used above for training samples
      y_valid_pred, _ = create_logits (tf_X_valid, weights, biases)
      
      # Transform test samples using the same transformation used above for training samples
      y_test_pred, _ = create_logits (tf_X_test, weights, biases)


  with tf.Session(graph=graph) as session:
      # This is a one-time operation which ensures the parameters get initialized as
      # we described in the graph: random weights for the matrix, zeros for the
      # biases. 
      tf.global_variables_initializer().run()
      print ('\nstep#       Loss     Training Accuracy Validation Accuracy')
      for step in range(num_steps):
          offset = (step * batch_size) % (num_samples - batch_size)
          # Generate a minibatch from the training dataset for Stocastic Gradient Descent
          batch_X = X_train[offset:(offset + batch_size), :]
          batch_y = y_train[offset:(offset + batch_size), :]
          # Prepare a dictionary telling the session where to feed the minibatch.
          # The key of the dictionary is the placeholder node of the graph to be fed,
          # and the value is the numpy array to feed to it.
          feed_dict = {tf_X_train : batch_X, tf_y_train : batch_y}
          _, l, y_pred = session.run([optimizer, loss, y_train_pred], feed_dict=feed_dict)

          # Run the computations. We tell .run() that we want to run the optimizer,
          # and get the loss value and the training predictions returned as numpy
          # arrays.
          if (step % print_steps == 0):
              print('{0:5d} {1:10.2f} {2:15.2f} {3:15.2f}'.format(step, l, accuracy(y_pred, batch_y), 
                                                            accuracy(y_valid_pred.eval(), y_valid)))
              # Calling .eval() on valid_prediction is basically like calling run(), but
              # just to get that one numpy array. Note that it recomputes all its graph
              # dependencies.
      print('Test accuracy: {:.1f}'.format(accuracy(y_test_pred.eval(), y_test))) 
batch_size = 128
num_nodes= [4096]
image = [image_size,image_size]
nclass = 10
num_steps = 2501
print_steps = 500
dropout = False
num_samples=0 # try with full dataset first
n_layer_nn (train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, image, 
  nclass, batch_size, num_nodes, num_samples, num_steps, print_steps, dropout)

