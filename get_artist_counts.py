# usage: cat museum_data.csv |  python get_artist_counts.py

from collections import Counter
import sys


c = Counter()
max_width = -1
max_height = -1

for line in sys.stdin:
	if len(line.split(",")) < 6:
		continue
	#print line
	artist = line.split(",")[2]
	if artist == "principalOrFirstMaker":
		continue
	c[artist] += 1
	width = int(line.split(",")[5]) 
	height = int(line.split(",")[6]) 
	if width > max_width:
		max_width = width
	if height > max_width:
		max_height = height


print c.most_common(30)
print max_width
print max_height