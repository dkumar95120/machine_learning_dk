import numpy as np
import cv2
import sys
import os
from   PIL import Image
from   six.moves import cPickle as pickle
from   scipy.misc import imresize
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py

img_size = 32

def show_bboxes (imagefile):
    im = cv2.imread(imagefile)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    # find contour in image
    # cv2.RETR_TREE retrieves the entire hierarchy of contours in image
    # if you only want to retrieve the most external contour
    # use cv.RETR_EXTERNAL
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    idx=0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 16 or h < 16: continue
        idx += 1
        roi=im[y:y+h,x:x+w]
        #cv2.imwrite(str(idx), roi)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.putText(im,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
    cv2.imshow('img',im)
    cv2.waitKey(0)    
    cv2.destroyAllWindows()

# The DigitStructFile is just a wrapper around the h5py data.  It basically references 
#     file_:            The input h5 matlab file
#     digitStructName   The h5 ref to all the file names
#     digitStructBbox   The h5 ref to all struc data
class DigitStructs:
    def __init__(self, file_, start_ = 0, end_ = 0):
        self.file_ = h5py.File(file_, 'r')
        self.names = self.file_['digitStruct']['name'][start_:end_] if end_ > 0 else self.file_['digitStruct']['name']
        self.bboxes = self.file_['digitStruct']['bbox'][start_:end_] if end_ > 0 else self.file_['digitStruct']['bbox']
        self.collectionSize = len(self.names)
        print("\n%s file structure contain %d entries" % (file_, self.collectionSize))

    def remove_anomaly_samples(self, data, max_class_length = 5):
        """
        Here we remove all data which has class length higher than specified value.
        """
        print("\nDataset size before update:", len(data))

        for i in range(len(data)):
            if i < len(data) and len(data[i]['label']) > max_class_length:
                print("\nAnomaly at index %d detected. Class size: %d" % (i, len(data[i]['label'])))
                del data[i]

        print("\nDataset after before update:", len(data))            
        return data
       
    def bbox_data(self, keys_):
        """
        Method handles the coding difference when there is exactly one bbox or an array of bbox. 
        """
        if (len(keys_) > 1):
            val = [self.file_[keys_.value[j].item()].value[0][0] for j in range(len(keys_))]
        else:
            val = [keys_.value[0][0]]
        return val
    # get_bbox returns a dict of data for the n(th) bbox. 
    def get_bbox(self, n):
        bbox = {}
        bb = self.bboxes[n].item()
        bbox['height'] = self.bbox_data(self.file_[bb]["height"])
        bbox['left']   = self.bbox_data(self.file_[bb]["left"])
        bbox['top']    = self.bbox_data(self.file_[bb]["top"])
        bbox['width']  = self.bbox_data(self.file_[bb]["width"])
        bbox['label']  = self.bbox_data(self.file_[bb]["label"])
        return bbox
    def getName(self, n):
        """
        Method returns the filename for the n(th) digitStruct. Since each letter is stored in a structure 
        as array of ANSII char numbers we should convert it back by calling chr function.
        """
        return ''.join([chr(c[0]) for c in self.file_[self.names[n][0]].value])

    def get_digit(self,n):
        s = self.get_bbox(n)
        s['name']=self.getName(n)
        return s

    def get_digits_data(self):
        """
        Method returns an array, which contains information about every image.
        This info contains: positions, labels 
        """
        return [self.get_digit(i) for i in range(self.collectionSize)]

    
    # Return a restructured version of the dataset (one object per digit in 'boxes').
    #
    #   Return a list of dicts :
    #      'filename' : filename of the samples
    #      'boxes' : list of dicts (one by digit) :
    #          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
    #          'left', 'top' : position of bounding box
    #          'width', 'height' : dimension of bounding box
    #
    # Note: We may turn this to a generator, if memory issues arise.
    def reformat(self): # getAllDigitStructure_ByDigit
        digit_data = self.get_digits_data()
        print("\nObject structure before transforming: ", digit_data[0])
        self.remove_anomaly_samples(digit_data)

        result = []
        for digit in digit_data:
            metadatas = []
            for i in range(len(digit['height'])):
                metadata = {}
                metadata['height'] = digit['height'][i]
                metadata['label']  = digit['label'][i]
                metadata['left']   = digit['left'][i]
                metadata['top']    = digit['top'][i]
                metadata['width']  = digit['width'][i]
                metadatas.append(metadata)

            result.append({ 'boxes':metadatas, 'name':digit["name"] })

        print("\nObject structure after transforming: ", result[0])

        return result

def prepare_images(samples, folder):
    print("Started preparing images...")
    ndigits = 5
    prepared_images = np.ndarray([len(samples), img_size, img_size, 1], dtype='float32')
    digit_labels    = np.zeros([len(samples), ndigits], dtype=int) * 10
    files = []
    for i in range(len(samples)):
        filename = samples[i]['name']
        filepath = os.path.join(folder, filename)
        image    = Image.open(filepath)
        boxes     = samples[i]['boxes']
        number_length = len(boxes)
        files.append(filepath)
        
        top = np.ndarray([number_length], dtype='float32')
        left = np.ndarray([number_length], dtype='float32')
        height = np.ndarray([number_length], dtype='float32')
        width = np.ndarray([number_length], dtype='float32')
        
        for j in range(number_length):
            # here we use j+1 since first entry used by label length
            digit_labels[i,j] = boxes[j]['label'] % 10                
            top[j]            = boxes[j]['top']
            left[j]           = boxes[j]['left']
            height[j]         = boxes[j]['height']
            width[j]          = boxes[j]['width']

        for j in range(number_length,ndigits):
            digit_labels[i,j] = 10

        img_min_top  = np.amin(top)
        img_min_left = np.amin(left)
        img_height   = np.amax(top)  + height[np.argmax(top)] - img_min_top
        img_width    = np.amax(left) + width[np.argmax(left)] - img_min_left

        img_left   = np.floor(img_min_left - 0.1 * img_width)
        img_top    = np.floor(img_min_top - 0.1 * img_height)
        img_right  = np.amin ([np.ceil(img_left + 1.2 * img_width), image.size[0]])
        img_bottom = np.amin ([np.ceil(img_top + 1.2 * img_height), image.size[1]])
            
        image = image.crop((img_left, img_top, img_right, img_bottom))
        image = image.resize([img_size, img_size], Image.ANTIALIAS) # Resize image to 32x32
        image = np.dot(np.array(image, dtype='float32'), [[0.2989],[0.5870],[0.1140]]) # Convert image to the grayscale
        image = (image)/128. - 1. # normalize and cerntralize images
        prepared_images[i] = image.reshape(img_size,img_size,1)
        
    print("Images cropped, resized, grayscaled and normalized")
    return prepared_images, digit_labels, files


from sklearn.utils import shuffle

def reformat(X, Y, image_size, num_channels, num_labels):
    x = np.zeros((X.shape[3],image_size,image_size,num_channels),dtype=np.float32)
    for i in range(X.shape[3]):
        x[i,] = X[:,:,:,i]
    y = (np.arange(num_labels) == Y[:,None]).astype(np.float32)
    return x, y

def synthesize_images_and_labels (X, y, img_size=32):
    # assuming imput shape X(nsample, img_size, img_size, nchannels) and y(nsample)
    X_blank = np.zeros((img_size,img_size,3), dtype=np.float32)
    #create a set of 4 digits while randomly splicing blanks
    ndigits = 4
    nsample = y.shape[0]
    y = y.reshape(-1)
    new_images = np.ndarray([nsample,img_size,img_size,1], dtype='float32')
    labels     = np.zeros((nsample,ndigits),dtype=np.int32)
    for i in range(nsample):
        # create a placeholder for the synthetic image
        x = np.zeros((ndigits,img_size,img_size,3),dtype=np.float32)
        # choose from 1 to ndigits with a minimum of 1 digit
        ndigit = np.random.choice(ndigits,1)[0] + 1
        for j in range(ndigit):
            k = i+j 
            if k >= nsample:          # wrap to get as many synthetic images as the sample data
                k = k % nsample
            x[j] = X[k]
            labels[i,j] = y[k] % 10
        # now pad with blank images
        for j in range(ndigit,ndigits):
            x[j] = X_blank            # the label for blank image is 10 (initialized)
            labels[i,j]=10
        # merge images in a 2x2 grid to create a 64x64 image
        x12   = np.concatenate((x[0], x[1]), axis=1)
        x34   = np.concatenate((x[2], x[3]), axis=1)
        image = np.concatenate((x12,  x34),  axis=0)
        image = np.dot(image, [[0.2989],[0.5870],[0.1140]]) # Convert image to the grayscale
        image = image.reshape(img_size*2, img_size*2)
        image = imresize(image, [img_size, img_size]) # Resize synthetic image to 32x32
        image = 2.*(image)/255. - 1.0
        new_images[i]  = image.reshape(img_size,img_size,1)

    return new_images, labels


def create_synthetic_images():
    train_data = sio.loadmat('train_32x32.mat')
    test_data  = sio.loadmat('test_32x32.mat')

    X_train = train_data['X']
    y_train = train_data['y'].reshape(-1)
    X_test  = test_data ['X']
    y_test  = test_data ['y'].reshape(-1)

    print('\n\nReal Set      X Shape,       Y_shape')
    print('Training', X_train.shape, y_train.shape)
    print('Testing ', X_test.shape, y_test.shape)

    image_size   = X_train.shape[0] #image size
    num_channels = X_train.shape[2] #set as 1:grayscale, or 3: color (RGB)
    num_classes = 10 # number of types of digits (0-9)

    # reshape to be [samples][pixels][width][height]

    X_train, _ = reformat(X_train, y_train, image_size, num_channels, num_classes)
    X_test, _  = reformat(X_test, y_test, image_size, num_channels, num_classes)

    print('\nFinal Data      X Shape,       Y_shape')
    X_train, y_train = synthesize_images_and_labels (X_train, y_train, image_size)
    print('Training', X_train.shape, y_train.shape)
    X_test,  y_test  = synthesize_images_and_labels (X_test,  y_test,  image_size)
    print('Testing ', X_test.shape, y_test.shape)
    input_shape = (image_size, image_size, 1)
    print ('input shape', input_shape)
    return X_train, y_train, X_test, y_test

build_synthetic_images=False
if build_synthetic_images:

    X_train, y_train, X_test, y_test = create_synthetic_images()

    pickle_file = 'synthetic.pickle'

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_data'  : X_train,
            'train_labels': y_train,
            'test_data'   : X_test,
            'test_labels' : y_test,
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

prepare_test_data = False
if prepare_test_data:
    file_ = '.\\train\\digitStruct.mat'
    dsf = DigitStructs(file_)
    train_data = dsf.reformat()

    file_ = '.\\test\\digitStruct.mat'
    dsf = DigitStructs(file_)
    test_data = dsf.reformat()

    train_data, train_labels, _ = prepare_images(train_data, '.\\train\\')
    print(train_data.shape, train_labels.shape)

    test_data, test_labels, test_filenames = prepare_images(test_data, '.\\test\\')
    print(test_data.shape, test_labels.shape)

    pickle_file = 'SVHN.pickle'

    try:
        f = open(pickle_file, 'wb')
        save = {
                'train_data'  : train_data,
                'train_labels': train_labels,
                'test_data'   : test_data,
                'test_labels' : test_labels,
                }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
        
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

def prepare_bbox_images(samples, folder):
    print("Started preparing bbox images...")
    ndigits = 5
    img_size=50 #final image size
    bbox_images    = np.ndarray([len(samples), img_size, img_size, 1], dtype='float32')
    bbox_labels    = np.zeros([len(samples), ndigits*4], dtype=np.uint8)
    files = []
    for i in range(len(samples)):
        filename = samples[i]['name']
        filepath = os.path.join(folder, filename)
        image    = Image.open(filepath)
        img      = np.asarray(image)
        img_h    = img.shape[0]
        img_w    = img.shape[1]
        boxes    = samples[i]['boxes']
        number_length = len(boxes)
        files.append(filepath)
        
        # rescale all bbox to within [0,100] dimension for the final image
        for j in range(number_length):
            k = 4*j

            x = round((img_size*boxes[j]['left'])/img_w)
            if x <= 0:
                x = 1
            bbox_labels[i,k+0] = x 

            w = round((img_size*boxes[j]['width'])/img_w)
            bbox_labels[i,k+2] = w if x+w < img_size else (img_size - x)


            y = round((img_size*boxes[j]['top'])/img_h)
            if y <= 0:
                y = 1
            bbox_labels[i,k+1] = y

            h = round((img_size*boxes[j]['height'])/img_h)
            bbox_labels[i,k+3] = h if y+h < img_size else img_size - y 

            x = bbox_labels[i,k+0]
            w = bbox_labels[i,k+2]
            y = bbox_labels[i,k+1]
            h = bbox_labels[i,k+3]
            if  x+w > img_size or y+h > img_size:
                print ("bounding box out of image boundary:", boxes[j])
                print ("x {}, w{}, y{}, h{}, img_w {}, img_h{}".format(x, y, w, h, img_w, img_h))

        image = image.resize([img_size, img_size], Image.ANTIALIAS) # Resize image to 32x32
        image = np.dot(np.array(image, dtype='float32'), [[0.2989],[0.5870],[0.1140]]) # Convert image to the grayscale
        image = (image)/128. - 1. # normalize and cerntralize images
        bbox_images[i] = image.reshape(img_size,img_size,1)
        
    print("Bounding box images resized, grayscaled and normalized")    
    return bbox_images, bbox_labels, files

prepare_bbox_data = True
if prepare_bbox_data:
    file_ = '.\\train\\digitStruct.mat'
    dsf = DigitStructs(file_)
    train_data = dsf.reformat()

    file_ = '.\\test\\digitStruct.mat'
    dsf = DigitStructs(file_)
    test_data = dsf.reformat()

    train_data, train_labels, _ = prepare_bbox_images(train_data, '.\\train\\')
    print(train_data.shape, train_labels.shape)

    test_data, test_labels, test_filenames = prepare_bbox_images(test_data, '.\\test\\')
    print(test_data.shape, test_labels.shape)

    pickle_file = 'SVHN_bboxes.pickle'

    try:
        f = open(pickle_file, 'wb')
        save = {
                'train_data'  : train_data,
                'train_labels': train_labels,
                'test_data'   : test_data,
                'test_labels' : test_labels,
                'test_files'  : test_filenames
                }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
        
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)
