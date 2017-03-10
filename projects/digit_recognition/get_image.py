import numpy as np
def get_image(x,i):

	a=np.zeros((32,32,3),dtype=np.uint8)

	for j in range(32):
		for k in range(32):
			for l in range(3):
				a[j,k,l] = x[j,k,l,i]
	return a