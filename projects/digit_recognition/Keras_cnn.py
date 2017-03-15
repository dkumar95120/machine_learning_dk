import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import scipy.io as sio
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

K.set_image_dim_ordering('tf')

def reformat(X, Y, image_size, num_channels, num_labels):
	x = np.zeros((X.shape[3],image_size,image_size,num_channels),dtype=np.float32)
	for i in range(X.shape[3]):
		x[i,] = X[:,:,:,i]
	for i in range(Y.shape[0]):
		Y[i] = Y[i] % 10    # to turn all 10 labels for zero to 0
	#y = (np.arange(num_labels) == Y[:,None]).astype(np.float32)
	return x, Y

def load_digits_dataset(multi_digit=True):
	if multi_digit:
		# load data
		pickle_file = 'SVHN.pickle'
		with open(pickle_file, 'rb') as f:
			save = pickle.load(f)
			X_train = save['train_data']
			y_train = save['train_labels']
			X_test = save['test_data']
			y_test = save['test_labels']
			del save  # hint to help gc free up memory
			print('Training set', X_train.shape, y_train.shape)
			print('Testing  set', X_test.shape, y_test.shape)

		num_classes = 11
		ndigit = y_test.shape[1]
		input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
		print ('input shape', input_shape)
	else:
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
		ndigit = 1

		# reshape to be [samples][pixels][width][height]
		# reshape data for input to CNN model

		X_train, y_train = reformat(X_train, y_train, image_size, num_channels, num_classes)
		X_test, y_test   = reformat(X_test, y_test, image_size, num_channels, num_classes)
		# normalize X values between 0 to 1
		X_train = 2.*X_train / 255 - 1.0
		X_test  = 2.*X_test / 255 - 1.0
		print('\nFinal Data      X Shape,       Y_shape')
		print('Training', X_train.shape, y_train.shape)
		print('Testing ', X_test.shape, y_test.shape)
		input_shape = (image_size, image_size, num_channels)
		print ('input shape', input_shape)
	return X_train, y_train, X_test, y_test, input_shape, num_classes, ndigit
#========================================================================================
def larger_model(input_shape, num_classes, ndigit):
	# create model
	model_input = x = Input(shape=input_shape)
	x = Convolution2D(32, 5, 5, border_mode='valid', input_shape=input_shape, activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Convolution2D(16, 3, 3, activation='relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	x = Dense(64, activation='relu')(x)

	# build a list of outputs for all digits
	outputs=[]
	for i in range(ndigit):
		outputs.append(Dense(num_classes, activation='softmax')(x))

	model = Model(input=model_input, output=outputs)

	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

X_train, y_train, X_test, y_test, input_shape, num_classes, ndigit = load_digits_dataset (True)
# build the model
model = larger_model(input_shape, num_classes, ndigit)
# Fit the model using 90% of training data and use remaining for validation
n = round(0.9*X_train.shape[0])
yt = []
yv = []
yf = []
for i in range(ndigit):
	yt.append(y_train[:n,i])
	yv.append(y_train[n:,i])
	yf.append(y_test[:,i])

model.fit(X_train[:n], yt, validation_data=(X_train[n:], yv), nb_epoch=10, batch_size=64, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, yf, verbose=0)
print("Overall accuracy: %.2f%%" % (scores[1]*100))