#18809

import os
from PIL import Image

square = False

#image_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/images/'
image_dir = 'images/'

new_image_dir = image_dir.replace("images", "resized_images")

if not os.path.exists(new_image_dir):
    os.makedirs(new_image_dir)

new_size = (224, 224)

img_files = os.listdir(image_dir)
for img_file in img_files:
	if ".png" not in img_file and ".jpg" not in img_file and ".jpeg" not in img_file :
		continue
	old_im = Image.open(image_dir + img_file)
	old_size = old_im.size

	if square:
		longest_side = 1.0 * max(old_size[0], old_size[1])
		new_size = (int(round(old_size[0]/longest_side * 224)),
			int(round(old_size[1]/longest_side * 224)))
		new_im = old_im.resize(new_size)
		newest_im.paste(new_im, ((224-old_size[0])/2,
	                      (2224-old_size[1])/2))
		newest_im.save(new_image_dir + img_file)
	else:
		longest_side = 1.0 * max(old_size[0], old_size[1])
		new_size = (int(round(old_size[0]/longest_side * 224)),
			int(round(old_size[1]/longest_side * 224)))
		newest_im = old_im.resize(new_size)
		newest_im.save(new_image_dir + img_file)
	#print longest_side
	#new_im = Image.new("RGB", (longest_side, longest_side)) 
	#new_im = Image.new("RGB", (longest_side-old_size[0]/longest_side * 224,
	#	longest_side-old_size[1]/longest_side * 224)) 
	# pad with black
	#new_im.paste(old_im, ((longest_side-ongest_side-old_size[0])/2,
	#                      (longest_side-old_size[1])/2))
	# resize to 224x224
	#newest_im = new_im.resize(new_size)
	#print old_size
	# new_size = (int(round(old_size[0]/longest_side * 224)),
	# 	int(round(old_size[1]/longest_side * 224)))
	# #print new_size
	# newest_im = old_im.resize(new_size)
	# newest_im.save(new_image_dir + img_file)
	
