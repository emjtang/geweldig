import glob
import code, multiprocessing
import sys, pickle, os
import cv2
import numpy 
from sklearn import cluster

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

image_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/images/'

image_feat_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/image_feats/'
surf = cv2.SURF(400)
num_features = 500
k_means = pickle.load(open( "kmeans_500.pickle", "rb" ))
# create dictionary of 1000 words
num_black_imgs = 0

def assign_cluster(filename):
	image_id = os.path.splitext(os.path.basename(filename))[0]
	image_out = image_feat_dir + image_id + ".npy"
	if os.path.isfile(image_out):
		print "Already downloaded!"
		return
	print "Assigning cluster for ", image_id
	img = cv2.imread(filename, 0) 
	try:
		kp, des = surf.detectAndCompute(img,None)
		v = numpy.zeros(num_features)

		if (len(kp) > 0):
			labels = k_means.predict(des)
			for label in labels:
				v[label] = v[label] + 1
			# keyboard()
			v = v/float(sum(v))
		if (sum(v) == 0):
			print "black"
		numpy.save(image_out,v)
	except cv2.error as e:
		print e
		return

	# keyboard()
		# 304 by 128

def main():
	all_files = glob.glob(image_dir + '*.jpg')
	# assign_cluster(all_files[10])

	# for i, filename in enumerate(all_files):
	# 	assign_cluster(filename)
	# 	if (i == 0):
	# 		allFeats = feat
	# 	else:
	# 		allFeats = numpy.vstack((allFeats, feat))
		
	# keyboard()
	# image_file = image_dir + '958.jpg'
	# sanity_check(all_files[:100])
	# for k, feat in enumerate(allFeats): 
	# 	[(i, max(allFeats[k])) for i,j in enumerate(allFeats[k]) if allFeats[k][i] == max(allFeats[k])]
	pool = multiprocessing.Pool()
	pool.map(assign_cluster, all_files)
	pool.close()
	pool.join()  

if __name__ == '__main__':
	main()
# keyboard()

# SURF 
# to create bag of words features
# cluster features / cluster centers = words

# cluster the descriptors

# remember to norm it

# for filename in glob.glob(image_dir + '*.jpg'):
# 	keyboard()


		# go through all ids, and get image
