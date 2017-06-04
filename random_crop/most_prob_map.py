import csv 
import math
from collections import Counter
import numpy as np  
import matplotlib.pyplot as plt


def importAccuracyCounts():
	accMap = np.zeros(shape=(225, 225, 10))
	line = 0
	with open('cropped.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			pred = int(row[2])

			xcoord = int(row[0])
			ycoord = int(row[1])
			for xpos in np.arange(xcoord, xcoord + 45):
				for ypos in np.arange(ycoord, ycoord + 45):
					accMap[ypos][xpos][pred] += 1

	heatMap = np.argmax(accMap, axis=2)

	# print M[0]
	return heatMap, accMap

def main():
	heatMap, accMap = importAccuracyCounts()
	# print heatMap, accMap
	print accMap[0]
	print heatMap[0]

	plt.imshow(heatMap, cmap='jet', interpolation='nearest')
	plt.show()


if __name__ == '__main__':
	main()