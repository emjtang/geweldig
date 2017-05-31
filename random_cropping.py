#18809

import os
from PIL import Image

square = True

#image_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/images/'
#image_dir = 'images/'
image_dir = '/home/tarabalakrishnan/geweldig/data/geweldig_photos_old/'
new_image_dir = image_dir.replace("_old", "")

if not os.path.exists(new_image_dir):
    os.makedirs(new_image_dir)

new_size = (299, 299)
new_width = new_size[0]
new_height = new_size[1]

cnt = 0
img_files = []
for r,d,f, in os.walk(image_dir):
	for file_name in f:
		if cnt == 0:
			print r, d, file_name
			cnt += 1
		img_files.append(os.path.join(r,file_name))

for img_file in img_files:
	if ".png" not in img_file and ".jpg" not in img_file and ".jpeg" not in img_file :
		continue
	old_im = Image.open(img_file)
	old_size = old_im.size

	if square:
		longest_side = 1.0 * max(old_size[0], old_size[1])
		new_size = (int(round(old_size[0]/longest_side * new_width)),
			int(round(old_size[1]/longest_side * new_width)))
		new_im = old_im.resize(new_size)
		newest_im = Image.new("RGB", (new_width, new_height))
		newest_im.paste(new_im, ((new_width-new_size[0])/2,
	                      (new_width-new_size[1])/2))
		newest_im.save(img_file.replace("_old", ""))
	else:
		longest_side = 1.0 * max(old_size[0], old_size[1])
		new_size = (int(round(old_size[0]/longest_side * new_width)),
			int(round(old_size[1]/longest_side * new_width)))
		newest_im = old_im.resize(new_size)
		newest_im.save(img_file)
	#print longest_side
	#new_im = Image.new("RGB", (longest_side, longest_side)) 
	#new_im = Image.new("RGB", (longest_side-old_size[0]/longest_side * new_width,
	#	longest_side-old_size[1]/longest_side * new_width)) 
	# pad with black
	#new_im.paste(old_im, ((longest_side-ongest_side-old_size[0])/2,
	#                      (longest_side-old_size[1])/2))
	# resize to new_widthxnew_width
	#newest_im = new_im.resize(new_size)
	#print old_size
	# new_size = (int(round(old_size[0]/longest_side * new_width)),
	# 	int(round(old_size[1]/longest_side * new_width)))
	# #print new_size
	# newest_im = old_im.resize(new_size)
	# newest_im.save(new_image_dir + img_file)
	
