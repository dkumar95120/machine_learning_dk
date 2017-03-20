from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile
import helper
import numpy as np
from keras.utils import np_utils
import pickle
import problem_unittests as tests
import tensorflow as tf

num_labels=10

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

tests.test_folder_path(cifar10_dataset_folder_path)

# Explore the dataset
batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    x = x/255.
    return x
tests.test_normalize(normalize)

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    global num_labels
    y=np_utils.to_categorical(x, num_labels)
    return y

tests.test_one_hot_encode(one_hot_encode)

# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    im_size1, im_size2, im_size3 = image_shape
    input_shape = (None, im_size1, im_size2, im_size3)
    tf_X = tf.placeholder("float", shape=input_shape, name='x')
    return tf_X


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    tf_y = tf.placeholder("float", shape=(None, n_classes), name='y')
    return tf_y


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    keep_prob = tf.placeholder("float", name='keep_prob') #dropout (keep probability)
    return keep_prob


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

layer_conv=0
def weights_conv(shape, name):
    with tf.variable_scope('conv') as scope:
        try:
            v = tf.get_variable(shape=shape, name=name,
               initializer=tf.contrib.layers.xavier_initializer_conv2d())
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(shape=shape, name=name,
               initializer=tf.contrib.layers.xavier_initializer_conv2d())            
    return v

def biases_var(shape, name):
    v = tf.Variable(tf.constant(0.1, shape=shape), name=name)
    return v

def conv2d(x, W, conv_strides, padopt):
    conv_stride1, conv_stride2 = conv_strides
    return tf.nn.conv2d(x, W, strides=[1, conv_stride1, conv_stride2, 1], padding=padopt)

def max_pool(x, pool_ksize, pool_strides, padopt):
    pool_size1, pool_size2 = pool_ksize
    pool_stride1, pool_stride2 = pool_strides
    return tf.nn.max_pool(x, ksize=[1, pool_size1, pool_size2, 1], strides=[1, pool_stride1, pool_stride2, 1], padding=padopt)

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    global layer_conv
    # TODO: Implement Function
    padopt='SAME'
    x_shape = x_tensor.get_shape().as_list()
    print('tensor shape:',x_shape)
    print('conv_ksize', conv_ksize)
    print('conv_num_outputs', conv_num_outputs)
    print('conv_strides', conv_strides)
    print('pool_ksize', pool_ksize)
    print('pool_strides', pool_strides)
    c_size1, c_size2 = conv_ksize
    conv_shape = [c_size1, c_size2, x_shape[3], conv_num_outputs]
    print('conv_shape', conv_shape)

    w_c = weights_conv(conv_shape,'w_conv'+str(layer_conv))
    b_c = biases_var([conv_num_outputs],'b_conv'+str(layer_conv))

    hidden = tf.nn.relu(conv2d(x_tensor, w_c, conv_strides, padopt) + b_c)
    hidden = max_pool(hidden, pool_ksize, pool_strides, padopt)

    return hidden 

tests.test_con_pool(conv2d_maxpool)

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    shape  = x_tensor.get_shape().as_list()
    print(shape)
    x = tf.reshape(x_tensor, [-1, shape[1] * shape[2] * shape[3]])

    return x


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)

layer_fc = 0

def weights_fc(shape, name):
    with tf.variable_scope('fc') as scope:
        try:
            v = tf.get_variable(shape=shape, name=name,
                initializer=tf.contrib.layers.xavier_initializer())
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(shape=shape, name=name,
                initializer=tf.contrib.layers.xavier_initializer())            
    return v

def fully_conn(x_tensor, num_outputs, keep_prob=1.0):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    global layer_fc
    shape  = x_tensor.get_shape().as_list()
    print(shape)
    weights = weights_fc([shape[1], num_outputs],'w_fc'+str(layer_fc))
    biases  = biases_var([num_outputs], 'b_fc'+str(layer_fc))
    logits = tf.nn.relu(tf.matmul(x_tensor, weights) + biases)
    logits = tf.nn.dropout(logits, keep_prob)
    return logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    shape  = x_tensor.get_shape().as_list()
    print(shape)
    weights = weights_fc([shape[1], num_outputs],'w_o')
    biases  = biases_var([num_outputs], 'b_o')
    logits = tf.matmul(x_tensor, weights) + biases
    return logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    global layer_conv
    global layer_fc
    
    x_tensor         = x
    shape            = x_tensor.get_shape().as_list()
    print(shape)
    layer_conv       = 1              # convolution layer number
    conv_ksize       = (5, 5)
    conv_num_outputs = [8, 16, 32, 64]
    conv_strides     = (1, 1)
    pool_ksize       = (2, 2)
    pool_strides     = (2, 2)

    for i, conv_outputs in enumerate(conv_num_outputs):
        layer_conv = i+1
        x_tensor = conv2d_maxpool(x_tensor, conv_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
        shape            = x_tensor.get_shape().as_list()
        print('shape after conv layer',layer_conv, shape)
    
    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    x_tensor = flatten(x_tensor)
    shape    = x_tensor.get_shape().as_list()
    print('shape after flatening',shape)
    
    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    fc_nodes = [1024, 512, 128, 64]
    
    for i, num_outputs in enumerate(fc_nodes):
        layer_fc    = i+1          # Fully connected layer number
        x_tensor =  fully_conn(x_tensor, num_outputs, keep_prob)
        shape    = x_tensor.get_shape().as_list()
        print('shape after fully connected layer', layer_fc, shape)
    

    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    num_outputs = num_labels
    x_tensor = output(x_tensor, num_outputs)
    shape    = x_tensor.get_shape().as_list()
    print('shape after output layer', shape)
    
    # TODO: return output
    return x_tensor


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
image_size = (32, 32, 3)
x = neural_net_image_input(image_size)
y = neural_net_label_input(num_labels)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)

def print_stats(session, feature_batch, label_batch, cost, accuracy, keep_probability):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    global valid_features, valid_labels
    # TODO: Implement Function
    feed_train ={x : feature_batch, y : label_batch, keep_prob: keep_probability}
    feed_valid ={x : valid_features, y: valid_labels, keep_prob: keep_probability}
    print('loss: {0:10.2f} batch accuracy: {1:5.2f} valid accuracy:{2:5.2f}'.format(cost.eval(feed_train), 
                                                                           accuracy.eval(feed_train),
                                                                           accuracy.eval(feed_valid)))

# Tune Parameters
epochs = 20
batch_size = 64
keep_probability = 0.8

save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy, keep_probability)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

    import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()