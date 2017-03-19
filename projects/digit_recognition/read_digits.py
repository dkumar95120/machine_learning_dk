# Import the modules
import cv2
from keras.models import load_model
from   scipy.misc import imresize
import numpy as np
import sys

def read_digits(argv):
    images = len(argv) - 1
    if images == 0:
        print('Usage: {} imagefile1 imagefile2 ...'.format(argv[0]))
        return

    print('Number of images:', images)
    grayscale = [[0.574],[0.575],[0.574]]
    # Load a compiled model identical to the previous one
    model_file = 'k_cnn_model.h5'
    try:
        model = load_model(model_file)
    except Exception as e:
        print('Unable to load model from file', model_file, ':', e)
        raise

    img_size = 32
    ndigit   = 5
    cell = img_size//2

    for imagefile in argv[1:]:
        print('processing image', imagefile)
        # Read the input image as grayscale
        im = cv2.imread(imagefile)
        height, width, channels = im.shape

        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)        # convert to gray

        # apply Gaussian filtering
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

        # Threshold the image
        #ret, im_th = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY_INV)
        im_th = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

        # Find contours in the image
        _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        nrect = len(rects)
        print('found {} digits in the image {}'.format(nrect, imagefile))
        images = np.ndarray([nrect, img_size, img_size, 1], dtype='float32')

        # For each rectangular region, predict digits included (upto 5 supported for now)
        index = 0
        for i, rect in enumerate(rects):
            # Draw bounding box
            x,y,w,h = rect
            if h <= 16 or w <= 16:
                continue

            # Make the rectangular region around the digit and within image boundaries
            x1 = int(x - .1*w)
            x1 = 1 if x1 < 1 else x1
            y1 = int(y - .1*h)
            y1 = 1 if y1 < 1 else y1
            w1 = int(w * 1.2)
            w1 = (width - x1 -1) if (x1+w1) >= width else w1
            h1 = int(h * 1.2)
            h1 = (height -y1 -1) if (y1+h1) >= height else h1
            area = w1*h1
            if (area <= 0):
                print("found degenerate bounding box", x1, y1, w1, h1)
                continue

            img = im_gray[y1:,x1:][:h1,:w1]
            # Resize the image
            img = imresize(img, [img_size, img_size]) # Resize synthetic image to 32x32

            img = cv2.dilate(img, (3, 3))

            img = np.array([img],dtype='float32').reshape(img_size, img_size, -1)
            images[index]= 2*img/255. - 1.0
            rects[index] = x1, y1, w1, h1
            cv2.rectangle(im, (x1,y1), (x1+w1, y1+h1), (0, 255, 0), 2)
            index += 1

        print ("Total valid digits found", index)
        if not index:
            print("No sizable digits found")
            return
        y_pred = model.predict(images[:index])
        yp = np.argmax(y_pred, 2).T

        for i in range(index):
            txt = ''
            x,y,w,h = rects[i]
            xt= x + w//2
            yt = int(y+h)
            for j in yp[i]:
                txt = txt + str(j) if j < 10 else txt
            cv2.putText(im, txt, (xt,yt), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 0, 0), 1)

        cv2.imshow("Image with Numbers", im)
        cv2.waitKey()

if __name__ == "__main__":
    read_digits(sys.argv)