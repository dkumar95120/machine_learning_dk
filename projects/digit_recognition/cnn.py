# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

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

image_size   = 28 #TODO: image size
num_labels   = 10 #TODO: number of different label types
num_channels = 1 #TODO: set as 1:grayscale, or 3: color (RGB)

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

batch_size = 128
patch_size = 14
depth      = 16
num_hidden = 128
maxpool    = True
dropout    = False
# build a convolutional model for the given input data, weights and biases
def model(X, weights, biases, maxpool=False, dropout=False):
	nlayer = len(weights.keys())
	hidden = X
	for layer in range(nlayer-2):
		hidden = tf.nn.relu(conv2d(hidden, weights[layer]) + biases[layer])
		if maxpool:
			hidden = max_pool_2x2(hidden)
		if (dropout):
			hidden = tf.nn.dropout(hidden, .75)

	shape = hidden.get_shape().as_list()
	hidden = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
	hidden = tf.nn.relu(tf.matmul(hidden, weights[nlayer-2]) + biases[nlayer-2])
	logits = tf.matmul(hidden, weights[nlayer-1]) + biases[nlayer-1]
	return logits

graph = tf.Graph()

with graph.as_default():
	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
	tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset  = tf.constant(test_dataset)

	# Variables.
	weights={}
	biases ={}
	# first dimesion of the weights will be the number of features in the training dataset which is the image size
	# build weights and biases for each layer

	weights[0] = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
	biases[0]  = tf.Variable(tf.zeros([depth]))
	weights[1] = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
	biases[1]  = tf.Variable(tf.constant(1.0, shape=[depth]))
	dim1       = depth*4
	weights[2] = tf.Variable(tf.truncated_normal([dim1, num_hidden], stddev=0.1))
	biases[2]  = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
	weights[3] = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
	biases[3]  = tf.Variable(tf.constant(1.0, shape=[num_labels]))

	# Build CNN model using specified weights with option to maxpool and dropout
	logits = model(tf_train_dataset, weights, biases, maxpool=True, dropout=True)

	# Training computation.
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
	# Loss function with L2 Regularization with beta
	l2_loss = True
	if l2_loss:
		beta = .01
		nlayer = len(weights.keys())
		for layer in range(nlayer):
			w_l2_loss = tf.nn.l2_loss(weights[layer])
			regularizers = regularizers + w_l2_loss if  layer else w_l2_loss
		loss = tf.reduce_mean(loss + beta * regularizers)

	# Build optimizer with exponential decay
	#optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
	init_learning_rate = 0.05
	global_step = tf.Variable(0)  # count the number of steps taken.
	learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps=100000, decay_rate=0.96, staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset, weights, biases, maxpool, dropout))
	test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, biases, maxpool, dropout))

num_steps = 5001
print_steps = 500
with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	print ('\nstep#       Loss     Training Accuracy Validation Accuracy')
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run(
		[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % print_steps == 0):
			print('{0:5d} {1:10.2f} {2:15.2f} {3:15.2f}'.format(step, l, accuracy(predictions, batch_labels),
																accuracy(valid_prediction.eval(), valid_labels)))
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))