import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import scipy.io as sio
import matplotlib.pyplot as plt

def reformat(X, Y, image_size, num_channels, num_labels):
	x = np.zeros((X.shape[3],image_size,image_size,num_channels),dtype=np.float32)
	for i in range(X.shape[3]):
		x[i,] = X[:,:,:,i]
	y = (np.arange(num_labels) == Y[:,None]).astype(np.float32)
	return x, y

def load_digits_dataset(mnist_data=True):
	if mnist_data:
		K.set_image_dim_ordering('th')
		# load data
		(X_train, y_train), (X_test, y_test) = mnist.load_data()

		print('Data Set      X Shape,       Y_shape')
		print('Training', X_train.shape, y_train.shape)
		print('Testing ', X_test.shape, y_test.shape)

		# reshape to be [samples][pixels][width][height]
		X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
		X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

		print('Training', X_train.shape, y_train.shape)
		print('Testing ', X_test.shape, y_test.shape)

		# normalize inputs from 0-255 to 0-1
		X_train = X_train / 255
		X_test = X_test / 255
		# one hot encode outputs
		y_train = np_utils.to_categorical(y_train)
		y_test = np_utils.to_categorical(y_test)
		num_classes = y_test.shape[1]
		print('\nFinal Data      X Shape,       Y_shape')
		print('Training', X_train.shape, y_train.shape)
		print('Testing ', X_test.shape, y_test.shape)
		input_shape = (1, 28, 28)
		print ('input shape', input_shape)
	else:
		K.set_image_dim_ordering('tf')
		train_data = sio.loadmat('train_32x32.mat')
		test_data  = sio.loadmat('test_32x32.mat')

		X_train = train_data['X']
		y_train = train_data['y'].reshape(-1)
		X_test = test_data['X']
		y_test = test_data['y'].reshape(-1)

		print('\n\nReal Set      X Shape,       Y_shape')
		print('Training', X_train.shape, y_train.shape)
		print('Testing ', X_test.shape, y_test.shape)

		image_size   = X_train.shape[0] #image size
		num_channels = X_train.shape[2] #set as 1:grayscale, or 3: color (RGB)
		num_classes = 10 # number of types of digits (0-9)

		# reshape to be [samples][pixels][width][height]
		# reshape data for input to CNN model

		X_train, y_train = reformat(X_train, y_train, image_size, num_channels, num_classes)
		X_test, y_test   = reformat(X_test, y_test, image_size, num_channels, num_classes)
		# normalize X values between 0 to 1
		X_train = X_train / 255
		X_test = X_test / 255
		print('\nFinal Data      X Shape,       Y_shape')
		print('Training', X_train.shape, y_train.shape)
		print('Testing ', X_test.shape, y_test.shape)
		input_shape = (image_size, image_size, num_channels)
		print ('input shape', input_shape)
	return X_train, y_train, X_test, y_test, input_shape, num_classes
#========================================================================================
def larger_model(input_shape, num_classes):
	# create model
	model = Sequential()
	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(16, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

X_train, y_train, X_test, y_test, input_shape, num_classes = load_digits_dataset (False)
# build the model
model = larger_model(input_shape, num_classes)
# Fit the model using 80% of training data and use remaining 20% for validation
n = round(.8*X_train.shape[0])
model.fit(X_train[:n], y_train[:n], validation_data=(X_train[n:], y_train[n:]), nb_epoch=2, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline accuracy: %.2f%%" % (scores[1]*100))