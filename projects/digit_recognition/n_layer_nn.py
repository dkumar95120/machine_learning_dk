from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


def load_data(pickle_file):

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        X_train = save['train_data']
        y_train = save['train_labels']
        X_tests = save['test_data']
        y_tests = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', X_train.shape, y_train.shape)
        print('Testing  set', X_tests.shape, y_tests.shape)

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

    return X_input, y_input, X_valid, y_valid, X_tests, y_tests

def accuracy(y, labels):
    ntests = labels.shape[0]
    ndigit = labels.shape[1]

    y_pred = np.argmax(y, 2).T
    match=0
    for i in range(ntests):
        match += np.sum(labels[i]==y_pred[i])//ndigit

    label_accuracy = 100.*match/ntests
    digit_accuracy = 100.*np.sum(labels==y_pred)//(ntests*ndigit)
    # return label accuracy as this is the really what we want to make sure was predicted correctly
    return digit_accuracy, label_accuracy

def model (X, weights, biases, dropout=False):
    #initialize relu_logits as the input dataset
    nlayer = len(weights.keys())
    #cur_logits = X.reshape(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])
    cur_logits = X
    for layer in range(nlayer):
        logits = tf.matmul(cur_logits, weights[layer]) + biases[layer]
        # Transform logits computed from layer 1 using relu function to compute logits for layer2
        if layer < nlayer-1:
            cur_logits = tf.nn.relu(logits)
        # Dropout on hidden layers only
            if dropout:
                cur_logits = tf.nn.dropout(cur_logits, 0.75)

    return logits

def pack(y):
    nitems = len(y.keys())
    if nitems > 5:
        print ("too much to pack")
        y_pack = None
    elif nitems == 5:
        y_pack = tf.stack([y[0],y[1], y[2], y[3], y[4]])
    elif nitems == 4:
        y_pack = tf.stack([y[0],y[1], y[2], y[3]])
    elif nitems == 3:
        y_pack = tf.stack([y[0],y[1], y[2]])
    elif nitems == 2:
        y_pack = tf.stack([y[0],y[1]])
    elif nitems == 1:
        y_pack = y[0]
    else:
        print("nothing to pack")
        y_pack = None

    return y_pack

#Create a single hidden layer neural network using RELU and 1024 nodes
def n_layer_nn (X_train, y_train, X_valid, y_valid, X_test, y_test, image, num_labels, ndigit,
    batch_size=128, num_nodes=[1024], num_samples=0, num_steps = 1001, print_steps=100, dropout=False):
    if num_samples ==0:
        num_samples = X_train.shape[0]

    # reshape X_train, X_valid and X_test
    X_train = X_train.reshape(X_train.shape[0],-1)
    X_valid = X_valid.reshape(X_valid.shape[0],-1)
    X_test  = X_test.reshape(X_test.shape[0],-1)
    graph = tf.Graph()
    #build nodes array and append it with the number of classes which is final number of nodes
    nodes_list = num_nodes
    nodes_list.append(num_labels)

    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_X_train = tf.placeholder(tf.float32, shape=(batch_size, image[0] * image[1] * image[2]))
        tf_y_train = tf.placeholder(tf.int32, shape=(batch_size, ndigit))

        tf_X_valid = tf.constant(X_valid.reshape(X_valid.shape[0],-1))
        tf_X_test  = tf.constant(X_test.reshape(X_test.shape[0],-1))

        # save weights and biases in a dictionary for each layer
        weights={}
        biases ={}
        logits ={}
        # build weights and biases for each layer for each of the possible 5 digits
        # first dimesion of the weights will be the number of features in the training dataset which is the image size
        loss = 0
        regularizers = 0

        for digit in range(ndigit):
            dim1 = image[0] * image[1] * image[2]
            weights[digit] = {}
            biases[digit]  = {}
            for layer, nodes in enumerate(nodes_list):
                weights[digit][layer] = tf.Variable(tf.truncated_normal([dim1, nodes]))
                biases [digit][layer] = tf.Variable(tf.zeros([nodes]))
                w_l2_loss = tf.nn.l2_loss(weights[digit][layer])
                regularizers = regularizers + w_l2_loss
                # Subsequent layer first dimension would the number of nodes in the previous layer
                dim1 = nodes
            logits[digit] = model (tf_X_train, weights[digit], biases[digit], dropout)
            loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_y_train[:,digit], logits=logits[digit]))

        # Loss function with L2 Regularization with beta
        beta = 0.0001
        loss = tf.reduce_mean(loss + beta * regularizers)

        # Optimize using Gradient Descent
        global_step = tf.Variable(0)  # count the number of steps taken.
        init_learning_rate = 0.05
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps=100000, decay_rate=0.95, staircase=True)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Compute prediction for training samples 
        y_pred={}
        for digit in range(ndigit):
            y_pred[digit] = tf.nn.softmax(logits[digit])

        y_train_pred = pack(y_pred)

        # Transform validation samples using the same transformation used above for training samples
        for digit in range(ndigit):
            valid_logits = model (tf_X_valid, weights[digit], biases[digit], dropout)
            y_pred[digit] = tf.nn.softmax(valid_logits)
        y_valid_pred = pack(y_pred)

        # Transform test samples using the same transformation used above for training samples
        for digit in range(ndigit):
            test_logits = model (tf_X_test, weights[digit], biases[digit], dropout)
            y_pred[digit] = tf.nn.softmax(test_logits)
        y_test_pred = pack(y_pred)


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


synthetic = False
if synthetic:
    pickle_file = 'synthetic.pickle'
else:
    pickle_file = 'SVHN.pickle'

X_input, y_input, X_valid, y_valid, X_tests, y_tests = load_data(pickle_file)

ndigit      = y_tests.shape[1] # our synthetic data has only 4 digits
image       = [X_input.shape[1], X_input.shape[2], X_input.shape[3]]
batch_size  = 64
num_nodes   = [4096,512]
num_labels  = 11
num_steps   = 5001
print_steps = 500
dropout     = True
num_samples =0 # full dataset; otherwise specify the subset
print ('image size',image)
y_test_pred, y_valid_pred = n_layer_nn (X_input, y_input, X_valid, y_valid, X_tests, y_tests, image, num_labels, ndigit,
                                        batch_size, num_nodes, num_samples, num_steps, print_steps, dropout)