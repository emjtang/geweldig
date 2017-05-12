
# principalOrFirstMaker
# long title
# image id (id)
# webImage: url
# width
# height
# year from long title

import urllib3
import json
import sys

http = urllib3.PoolManager()
cur_page = 0
print "id,image_url,principalOrFirstMaker,title,longTitle,width,height"
while True:
	sys.stderr.write(str(cur_page) + "\n")
	url = "https://www.rijksmuseum.nl/api/en/collection?key=zKroAovK&format=json&ps=100&hasImage=True&permitDownload=True&p=" + str(cur_page)
	r = http.request('GET', url)
	json_data = json.loads(r.data.decode('utf-8'))
	if len(json_data["artObjects"]) == 0:
		break
	for artObject in json_data["artObjects"]:
		if artObject["webImage"] is None:
			continue
		new_line = ""
		new_line += artObject["id"].replace(",", ";;") + ","
		new_line += artObject["webImage"]["url"].replace(",", ";;") + ","
		new_line += artObject["principalOrFirstMaker"].replace(",", ";;") + ","
		new_line += artObject["title"].replace(",", ";;") + ","
		new_line += artObject["longTitle"].replace(",", ";;") + ","
		new_line += str(artObject["webImage"]["width"]) + ","
		new_line += str(artObject["webImage"]["height"]) + ","
		print new_line.encode('ascii', 'ignore')
	cur_page += 1



