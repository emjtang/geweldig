import glob
import code
import sys, pickle
import cv2
import numpy 
from sklearn import cluster
import random

# make sure you change the image directory
image_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/'

def keyboard(banner=None):
		''' Function that mimics the matlab keyboard command '''
		# use exception trick to pick up the current frame
		try:
				raise None
		except:
				frame = sys.exc_info()[2].tb_frame.f_back
		print "# Use quit() to exit :) Happy debugging!"
		# evaluate commands in current namespace
		namespace = frame.f_globals.copy()
		namespace.update(frame.f_locals)
		try:
				code.interact(banner=banner, local=namespace)
		except SystemExit:
				return

# number of "words" we want in the dictionary
n_clusters = 500

# speeded up robust features (speeded up version of SIFT)
# read here for more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
surf = cv2.SURF(400) 

'''
This function goes through all the images and extracts SURF features from each image.
Similar to NLP, where we want to be able to create a dictionary of words 
(and then the feature vector is the length of the dictionary, where each index corresponds to a word in the dict, 
and the # in each index of vector corresponds to the # of times that specific word occurs in the example)

We want to create a dictionary of these "image" words, by using kmeans to cluster all the extracted image descriptors
from each image to into n_clusters (or n words). The center of each cluster is the "word".

This will be our "bag of words" feature
'''
def createDictionary(image_files):
	for i, filename in enumerate(image_files):	
		img = cv2.imread(filename, 0) # load/read image
		kp, des = surf.detectAndCompute(img,None) # extract SURF features

		# stack all the features
		if (i == 0):
			all_des = des
		else:
			all_des = numpy.vstack((all_des, des))

	k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
	k_means.fit(all_des)

	labels = k_means.labels_
	values = k_means.cluster_centers_.squeeze()

	# not sure what the code below was for, maybe a sanity check for k_means?
	# for i, filename in enumerate(image_files):	
	# 	img = cv2.imread(filename, 0)
	# 	kp, des = surf.detectAndCompute(img,None)
	# 	if (i == 0):
	# 		blah = des
	# 	else:
	# 		blah = numpy.vstack((blah, des))
	# pred = k_means.predict(blah)

	with open('kmeans_500.pickle', 'w') as f:
		pickle.dump(k_means, f)

def main():
	all_files = glob.glob(image_dir + '*.jpg')
	createDictionary(all_files[:100])

if __name__ == '__main__':
	main()
