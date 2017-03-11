import numpy as np
import cv2
def get_image(imagefiles):
	nfile = len(imagefiles)
	img = np.zeros((nfile,32,32,3), dtype=np.float32)
	for i in range(nfile):
		img[i,] = cv2.imread(imagefiles[i])
	return img


def show_bboxes (imagefile):
	im = cv2.imread(imagefile)
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	contours, _ = cv2.findContours(gray, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	idx =0 
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if w < 32 or h < 32: continue
		idx += 1
		roi=im[y:y+h,x:x+w]
		cv2.imwrite(str(idx), roi)
		cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
		#cv2.putText(im,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
	cv2.imshow('img',im)
	cv2.waitKey(0)    
	cv2.destroyAllWindows()
