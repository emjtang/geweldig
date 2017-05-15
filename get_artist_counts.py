# usage: cat museum_data.csv |  python get_artist_counts.py  > museum_data_top10.csv

from collections import Counter
import sys

top_ten = ["George Hendrik Breitner", "Jan Luyken", "Reinier Vinkeles", "Marius Bauer", "Isaac Israels", "Bernard Picart", "Rembrandt Harmensz. van Rijn", "Johannes Tavenraat", "Willem Witsen", "Simon Fokke"]
c = Counter()

max_width = -1
max_height = -1

for line in sys.stdin:
	if len(line.split(",")) < 6:
		continue
	#print line
	artist = line.split(",")[2]
	if artist in ["principalOrFirstMaker", "anonymous", ""]:
		continue
	width = int(line.split(",")[5]) 
	height = int(line.split(",")[6]) 
	if artist in top_ten:
		if c[artist] < 100:
			print line
			c[artist] += 1




#print c.most_common(30)
#print max_width
#print max_height