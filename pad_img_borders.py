#18809

import os
from PIL import Image


#image_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/images/'
image_dir = 'images/'

new_image_dir = image_dir.replace("images", "resized_images")

if not os.path.exists(new_image_dir):
    os.makedirs(new_image_dir)

#new_size = (18809, 18809)
new_size = (7378, 7378)

img_files = os.listdir(image_dir)

for img_file in img_files:
	if ".png" not in img_file and ".jpg" not in img_file:
		continue
	old_im = Image.open(image_dir + img_file)
	old_size = old_im.size

	new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
	new_im.paste(old_im, ((new_size[0]-old_size[0])/2,
	                      (new_size[1]-old_size[1])/2))
	new_im.save(new_image_dir + img_file)
	
