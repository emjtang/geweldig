import csv 
import math
from collections import Counter
import numpy as np  
import matplotlib.pyplot as plt


def importAccuracyCounts():
	accMap = np.zeros(shape=(225, 225))
	numCovered = np.zeros(shape=(225, 225))
	line = 0
	with open('cropped.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if int(row[2]) == 7:
				acc = 1
			else:
				acc = 0

			xcoord = int(row[0])
			ycoord = int(row[1])
			for xpos in np.arange(xcoord, xcoord + 45):
				for ypos in np.arange(ycoord, ycoord + 45):
					accMap[ypos][xpos] += float(acc)
					numCovered[ypos][xpos] += 1

	heatMap = accMap / numCovered

	# print M[0]
	return heatMap, accMap, numCovered

def main():
	heatMap, accMap, numCovered = importAccuracyCounts()
	# print heatMap, accMap, numCovered
	# print accMap[0]
	# print numCovered[0]
	# print heatMap[0]

	plt.imshow(heatMap, cmap='jet', interpolation='nearest')
	plt.show()



if __name__ == '__main__':
	main()