#18809

import os
from PIL import Image


#image_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/images/'
image_dir = 'images/'

new_image_dir = image_dir.replace("images", "resized_images")

if not os.path.exists(new_image_dir):
    os.makedirs(new_image_dir)

new_size = (224, 224)

img_files = os.listdir(image_dir)
print "here"
for img_file in img_files:
	if ".png" not in img_file and ".jpg" not in img_file and ".jpeg" not in img_file :
		continue
	old_im = Image.open(image_dir + img_file)
	old_size = old_im.size


	longest_side = max(old_size[0], old_size[1])
	print longest_side
	new_im = Image.new("RGB", (longest_side, longest_side)) 
	# pad with black
	new_im.paste(old_im, ((longest_side-old_size[0])/2,
	                      (longest_side-old_size[1])/2))
	# resize to 224x224
	newest_im = new_im.resize(new_size)
	newest_im.save(new_image_dir + img_file)
	
