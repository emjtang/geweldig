import code
import sys
import csv
import re
import random
from collections import Counter
from operator import itemgetter
import multiprocessing  
import urllib
import os


 
image_dir = 'data/images/'
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

def download_image(info):
    image_id, image_url, artist, folder = info
    artist = re.sub('[^0-9a-zA-Z]+', '', artist)
    artist_dir = image_dir +  folder + artist + "/"
    #if not os.path.exists(artist_dir):
     #   os.makedirs(artist_dir)

    if image_url == '':
        print "Uh oh, no image url to be found"
        return
    image_out = image_dir + image_id + ".jpg"
    if os.path.isfile(image_out):
        print "Already downloaded!"
        return
    print image_id + '\t' + image_url
    urllib.urlretrieve(image_url, artist_dir + image_id + ".jpg")

def main():
    filename = 'museum_data.csv'
    artist_total_counts = Counter()
    artist_counts = Counter()
    with open(filename) as listings_file:
        reader = csv.DictReader(listings_file)
        listings = list(reader);

        image_info = [(row['id'], row['image_url'], row['principalOrFirstMaker']) for row in listings]

    image_info_with_type = []
    for img in image_info:
        #print img
        image_id, image_url, artist = img
        folder = "train/"
        artist_total_counts[artist] += 1

    
    if not os.path.exists(image_dir + "train/"):
	os.makedirs(image_dir + "train/")
    if not os.path.exists(image_dir + "test/"):
        os.makedirs(image_dir + "test/")
    if not os.path.exists(image_dir + "val/"):
        os.makedirs(image_dir + "val/")
    for img in image_info:
        #print img
        image_id, image_url, artist = img
        folder = "train/"
        artist_counts[artist] += 1
        if artist_counts[artist] >= artist_total_counts[artist] * 0.8:
            folder = "test/"
        elif artist_counts[artist] >= artist_total_counts[artist] * 0.6:
            folder = "val/"
	artist = re.sub('[^0-9a-zA-Z]+', '', artist)
        artist_dir = image_dir +  folder + artist + "/"
        if not os.path.exists(artist_dir):
            os.makedirs(artist_dir)
        image_info_with_type.append((image_id, image_url, artist, folder))


    pool = multiprocessing.Pool()
    pool.map(download_image, image_info_with_type)
    pool.close()
    pool.join()   
    print('done')


if __name__ == '__main__':
  main()
