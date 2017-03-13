from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from keras.utils import np_utils

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
nsample = y_train[0].shape[0]
ntrain = round(.9*nsample)
nvalid = nsample - ntrain
ntests = y_test[0].shape[0]

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
