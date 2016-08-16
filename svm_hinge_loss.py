import time
import numpy as np

def squash(x, delta):
	return np.maximum(0, x + delta)

def L_i(x, y, W):
	"""
	unvectorized version. Compute the multiclass svm loss for a single example (x,y)
	- x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
	with an appended bias dimension in the 3073-rd position (i.e. bias trick)
	- y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
	- W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
	"""
	delta = 1.0 			# see notes about delta later in this section
	scores = W.dot(x) 		# scores becomes of size 10 x 1, the scores for each class
	correct_class_score = scores[y]
	D = W.shape[0] 			# number of classes, e.g. 10
	loss_i = 0.0
	for j in range(D): 	# iterate over all wrong classes
		if j == y:
	  		continue
		# accumulate loss for the i-th example
		loss_i += max(0, scores[j] - correct_class_score + delta)

	return loss_i	

def L(X, y, W):
	"""
		Fully-vectorized implementation
		- X holds all the training examples as columns
		- y is array of integers specifying correct class
	  	- W holds the weights
	"""
	# grab number of images
	N = X.shape[1]
	# set desired threshold
	delta = 1.0

	# scores holds the score for each image as columns
	scores = W.dot(X)

	# grab scores of correct classes
	correct_classes = scores[y, np.arange(N)]

	# compute margins
	margins = np.maximum(0, scores - correct_classes + delta)

	# ignore the y-th position and only consider margin on max wrong class
	margins[y, np.arange(N)] = 0

	# compute loss column-wise
	losses = np.sum(margins, axis=0)
	
	# average out the loss
	loss = np.sum(losses) / N
	
	# return average loss
	return loss


"""
- 3 possible classes: dog cat horse
- 2 images of 4 pixels each
We thus have a matrix X of 4 rows and 2 columns
and a matrix W of 3 columns (3 classes) and 4 columns
Using the bias trick, we introduce a new dimension to X of 1's
and add the bias column to the weight matrix W
In conclusion, we have X (5x2) and W(3x5)
"""

# dim(X) = 5x2 = 2 training observations consisting of 5 pixel values each
X = np.array([
	[-15, 2], 
	[22, 11], 
	[-44, -35], 
	[56, 12], 
	[1.0, 1.0]
])
# dim(W) = 3x5 
W = np.array([
	[0.01, -0.05, 0.1, 0.05, 0.0], 
	[0.7, 0.2, 0.05, 0.16, 0.2], 
	[0.0, -0.45, -0.2, 0.03, -0.3]
])

y = np.array([2, 1])

losses = 0.0

start_time = time.time()
print("loss: ", L(X, y, W))
time1 = time.time() - start_time
print('Vectorized takes: {}'.format(time1))

start_time = time.time()
for i in range(X.shape[1]):
	losses += L_i(X[:, i], y[i], W)
print('loss = {}'.format(losses / X.shape[1]))
time2 = time.time() - start_time
print('Non-vectorized takes: {}'.format(time2))

print( ((time1 - time2) / time2) * 100 )
