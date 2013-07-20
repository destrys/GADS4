import urllib2
import json
import csv
import time

# Initialize
lastTime = ' '
filename = 'test.csv'

while True:
    # Get Station Data and convert it to a dict
    response = urllib2.urlopen('http://citibikenyc.com/stations/json')
    citijson = response.read()
    response.close()

    data = json.loads(citijson)

    if data['executionTime'] != lastTime:
        csvfile = open(filename, 'a')
        csvwriter = csv.writer(csvfile)

        for station in data['stationBeanList']:
            stationid = station['id']
            availDocks = station['availableDocks']
            totalDocks = station['totalDocks']
            status = station['statusKey']
            availBikes = station['availableBikes']
            csvwriter.writerow([data['executionTime'],stationid,availBikes,availDocks,totalDocks,status])

        csvfile.close()
        lastTime = data['executionTime']

    time.sleep(0.5)

