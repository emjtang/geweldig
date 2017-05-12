# geweldig
CS231N Project - Rijksmuseum classification
Dream Team: Tara, Sarah, Emily

# Downloading the dataset
The dataset exists in a CSV file, which was created as follows:
`python download_dataset.py > museum_data.csv`
Since the dataset may be too large (with 112,000+ images), we wrote a script to download images from the dataset.
The script reads the CSV file and then uses a threadpool (using multiprocessing) to parallelize the image downloads. 

## CSV file
The columns in the museum_data.csv are
id, image_url, principalOrFirstMaker, title, longTitle, width, height

## To download images
In the python script download_images.py, make sure the `image_dir` variable in line 12 is updated to your own path. 
It will exist in /data/images, so make sure the `images` directory exists (if not just mkdir it). 
Then run `python download_images.py`
