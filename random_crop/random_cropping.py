#18809
import os
from PIL import Image
import numpy as np

square = True

image_dir = 'data/r_crop_test/random_crop/'

new_image_dir = image_dir.replace("random_crop", "random_crop_new")
if not os.path.exists(new_image_dir):
    os.makedirs(new_image_dir)

count = 0
for (dirpath, dirnames, filenames) in os.walk(image_dir):
	for filename in filenames:
		
		img_file = os.path.join(dirpath,filename)
		new_dirpath = dirpath.replace("random_crop", "random_crop_new")
		
		if not os.path.exists(new_dirpath):
			os.makedirs(new_dirpath)

		new_img_file = img_file.replace("random_crop", "random_crop_new")
		
		'''File Modification -- Add in Random Crops'''
		if ".png" not in img_file and ".jpg" not in img_file and ".jpeg" not in img_file:
			continue
		
		orig_img = Image.open(img_file)
		old_size = orig_img.size #224x224
		if count == 0:
			print old_size


		new_size = (old_size[0]/5, old_size[1]/5)
		black_sq = Image.new("RGB", (new_size[0], new_size[1]))
		if count == 0:
			print black_sq.size

		file_version = 0
		for xpos in np.arange(0, old_size[0] - new_size[0] + 1, 4):
			for ypos in np.arange(0, old_size[1] - new_size[1] + 1, 4):
				img_copy = Image.open(img_file)
				img_copy.paste(black_sq, (xpos, ypos))
				new_ext = "_" + str(xpos) + "-" + str(ypos) + "."
				img_copy.save(new_img_file.replace(".", new_ext))
				file_version += 1

		count += 1
		if count == 1:
			break
	if count == 1:
		break