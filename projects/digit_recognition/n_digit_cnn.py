from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'synthetic.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	X_train = save['train_data']
	y_train = save['train_labels']
	X_tests = save['test_data']
	y_tests = save['test_labels']
	del save  # hint to help gc free up memory
	print('Training set', X_train.shape, y_train.shape)
	print('Testing  set', X_tests.shape, y_tests.shape)

num_labels=11
ndigit = y_train.shape[1]  # the first one is the number of digits in that sample

# split training data into training and validation sets
nsample = y_train.shape[0]
ntrain = round(0.9*nsample)

X_input = X_train[:ntrain]
X_valid = X_train[ntrain:]

print('train data shape', X_input.shape)
print('valid data shape', X_valid.shape)
print('tests data shape', X_tests.shape)

y_input = y_train[:ntrain]
y_valid = y_train[ntrain:]

print('train label shape', y_input.shape)
print('valid label shape', y_valid.shape)
print('tests label shape', y_tests.shape)

patch_size = 5
conv_stride= 1
pool_size  = 2
pool_stride= 2
padding = 'SAME'

def accuracy(y, labels):
	return (100.0 * np.sum(np.argmax(y, 2).T == labels) / y.shape[1] / y.shape[0])
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, conv_stride, conv_stride, 1], padding=padding)

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_stride, pool_stride, 1], padding=padding)

# Create Variables Function
def weights_conv(shape, name):
    return tf.get_variable(shape=shape, name=name,
        initializer=tf.contrib.layers.xavier_initializer_conv2d())

def weights_fc(shape, name):
    return tf.get_variable(shape=shape, name=name,
        initializer=tf.contrib.layers.xavier_initializer())

def biases_var(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def model (X, weights, biases, nconv, dropout=False):
	#initialize relu_logits as the input dataset
	nlayer = len(weights.keys())
	hidden = X
	# build convolution layers
	for layer in range(nconv):
		hidden = tf.nn.relu(conv2d(hidden, weights[layer]) + biases[layer])
		hidden = max_pool_2x2(hidden)
		if (dropout):
			hidden = tf.nn.dropout(hidden, .8)

	shape  = hidden.get_shape().as_list()
	cur_logits = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

	print ('First FC full dimensions', cur_logits.shape)

	for layer in range(nconv, nlayer):
		logits = tf.matmul(cur_logits, weights[layer]) + biases[layer]
		# Transform logits computed from layer 1 using relu function to compute logits for layer2
		if layer < nlayer-1:
			cur_logits = tf.nn.relu(logits)
		# Dropout on hidden layers only
			if dropout:
				cur_logits = tf.nn.dropout(cur_logits, 0.8)

	return logits

# Create Function for Image Size: Pooling
# 3 Convolutions
# 2 Max Pooling
# final_image_size = output_size_pool(image_size, patch_size, conv_stride, pool_size, pool_stride, nlayer

def output_size_pool(input_size, conv_filter_size, conv_stride, pool_filter_size, pool_stride, nlayer):

    pad = 1.0 if padding == 'SAME' else 0.0

    output = input_size
    for layer in range(nlayer):
	    # After convolution
	    output = (((output - conv_filter_size + 2 * pad) / conv_stride) + 1.00)
	    # After pool
	    output = (((output - pool_filter_size + 2 * pad) / pool_stride) + 1.00)

    return int(output)

#Create a single hidden layer neural network using RELU and 1024 nodes
def n_layer_cnn (X_train, y_train, X_valid, y_valid, X_test, y_test, image, num_classes, ndigit,
				batch_size=128, num_samples=0, num_steps = 1001, print_steps=100, dropout=False):
	beta = .01
	if num_samples ==0:
		num_samples = X_train.shape[0]

	# reshape X_train, X_valid and X_test
	graph = tf.Graph()
	#build nodes array and append it with the number of classes which is final number of nodes

	batch_size = 64
	image_size = image[0]
	depth      = [ 8, 16, 32]
	num_hidden = [64]

	with graph.as_default():

		# Input data. For the training data, we use a placeholder that will be fed
		# at run time with a training minibatch.
		tf_X_train = tf.placeholder(tf.float32, shape=(batch_size, image[0], image[1], image[2]))
		tf_y_train = tf.placeholder(tf.int32, shape=(batch_size, ndigit))
		tf_X_valid = tf.constant(X_valid)
		tf_X_test  = tf.constant(X_test)

		# save weights and biases in a dictionary for each layer
		weights={}
		biases ={}
		logits ={}
		# build weights and biases for each layer for each of the possible 5 digits
		# first dimesion of the weights will be the number of features in the training dataset which is the image size
		loss = 0
		regularizers = 0
		nconv = len(depth)
		nfc   = len(num_hidden)
		nlayer = nconv + nfc + 1

		#build convolution dimensions
		conv_dim = [num_channels] + depth + [num_hidden[0]]
		print ("convolution dimensions", conv_dim)

		# build FC layers dimensions
		final_image_size = output_size_pool(image_size, patch_size, conv_stride, pool_size, pool_stride, len(depth))
		hidden_dim = [final_image_size*final_image_size*depth[-1]] + num_hidden + [num_classes]
		print ("hidden dimensions", hidden_dim)


		print ('final image size', final_image_size)
		for digit in range(ndigit):
			weights[digit] = {}
			biases[digit]  = {}

			# buld weights and biases for convolution layers first
			for layer in range(nconv):
				shape = [patch_size, patch_size, conv_dim[layer], conv_dim[layer+1]]
				weights[digit][layer] = weights_conv(shape,'w'+str(digit)+str(layer))
				biases[digit][layer]  = biases_var([depth[layer]],'b'+str(digit)+str(layer))
				w_l2_loss = tf.nn.l2_loss(weights[digit][layer])
				regularizers = regularizers + w_l2_loss

			# build weights and biases for fully connected layers next
			for layer in range (nconv,nlayer):
				i = layer - nconv # index to hidden layer
				shape = [hidden_dim[i], hidden_dim[i+1]]
				weights[digit][layer] = weights_fc(shape,'w'+str(digit)+str(layer))
				biases[digit][layer]  = biases_var([hidden_dim[i+1]], 'b'+str(digit)+str(layer))
				w_l2_loss = tf.nn.l2_loss(weights[digit][layer])
				regularizers = regularizers + w_l2_loss

			logits[digit] = model (tf_X_train, weights[digit], biases[digit], nconv, dropout)
			loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_y_train[:,digit], logits=logits[digit]))

		# Loss function with L2 Regularization with beta
		loss = tf.reduce_mean(loss + beta * regularizers)

		# Optimize using Gradient Descent
		global_step = tf.Variable(0)  # count the number of steps taken.
		init_learning_rate = 0.01
		learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps=100000, decay_rate=0.95, staircase=True)
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
		optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

		# Compute prediction for training samples 
		y_pred={}
		for digit in range(ndigit):
			y_pred[digit] = tf.nn.softmax(logits[digit])

		y_train_pred = tf.stack([y_pred[0],y_pred[1], y_pred[2], y_pred[3]])

		# Transform validation samples using the same transformation used above for training samples
		for digit in range(ndigit):
			valid_logits = model (tf_X_valid, weights[digit], biases[digit], nconv, dropout)
			y_pred[digit] = tf.nn.softmax(valid_logits)

		y_valid_pred = tf.stack([y_pred[0],y_pred[1], y_pred[2], y_pred[3]])

		# Transform test samples using the same transformation used above for training samples
		for digit in range(ndigit):
			test_logits = model (tf_X_test, weights[digit], biases[digit], nconv, dropout)
			y_pred[digit] = tf.nn.softmax(test_logits)

		y_test_pred = tf.stack([y_pred[0],y_pred[1], y_pred[2], y_pred[3]])


	with tf.Session(graph=graph) as session:
		# This is a one-time operation which ensures the parameters get initialized as
		# we described in the graph: random weights for the matrix, zeros for the
		# biases. 
		tf.global_variables_initializer().run()
		print ('\nstep#       Loss     Training Accuracy Validation Accuracy')

		y_temp       = np.zeros([batch_size, ndigit, num_labels], dtype=np.int32)
		y_valid_temp = np.zeros([y_valid.shape[0], y_valid.shape[1], num_labels], dtype=np.int32)
		y_test_temp  = np.zeros([y_test.shape[0], y_test.shape[1], num_labels], dtype=np.int32)

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
			# and get the loss value and the training predictions returned as numpy arrays.
			if (step % print_steps == 0):
				y_valid_eval = y_valid_pred.eval()

				print('{0:5d} {1:10.2f} {2:15.2f} {3:15.2f}'.format(step, l, accuracy(y_pred, batch_y), 
				accuracy(y_valid_eval, y_valid)))
				# Calling .eval() on valid_prediction is basically like calling run()
		y_test_eval = y_test_pred.eval()

		print('Test accuracy: {:.1f}'.format(accuracy(y_test_eval, y_test)))
	return y_test_eval, y_valid_eval

image_size = X_train.shape[1]
batch_size = 64
num_channels = X_train.shape[3]
image = [image_size, image_size, num_channels]
print ('image size',image)
nclass = num_labels
num_steps = 1001
print_steps = 100
ndigit = 4 # our synthetic data has only 4 digits
dropout = False
num_samples=0 # try with full dataset first

patch_size = 5
conv_stride= 1
pool_size  = 2
pool_stride= 2
padopt = padding
conv_params = [patch_size, conv_stride, pool_size, pool_stride, padopt]

y_test_pred, y_valid_pred = n_layer_cnn (X_input, y_input, X_valid, y_valid, X_tests, y_tests, image, nclass, ndigit,
									            batch_size, num_samples, num_steps, print_steps, dropout)

