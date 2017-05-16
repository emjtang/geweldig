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

image_dir = '/Users/emjtang/Dropbox/_spr17/geweldig/data/images_top10/'
 
image_dir = '/home/tarabalakrishnan/geweldig/data/images_top10/'
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
    image_id, image_url, artist = info
    artist = re.sub('[^0-9a-zA-Z]+', '', artist)
    if image_url == '':
        print "Uh oh, no image url to be found"
        return
    image_out = image_dir + image_id + ".jpg"
    if os.path.isfile(image_out):
        print "Already downloaded!"
        return
    print image_id + '\t' + image_url
    artist_dir = image_dir + artist + "/"
    if not os.path.exists(artist_dir):
        os.makedirs(artist_dir)
    urllib.urlretrieve(image_url, artist_dir + image_id + ".jpg")

def main():
    filename = 'museum_data_top10.csv'
    with open(filename) as listings_file:
        reader = csv.DictReader(listings_file)
        listings = list(reader);

        image_info = [(row['id'], row['image_url'], row['principalOrFirstMaker']) for row in listings]

    pool = multiprocessing.Pool()
    pool.map(download_image, image_info)
    pool.close()
    pool.join()   
    print('done')


if __name__ == '__main__':
  main()
