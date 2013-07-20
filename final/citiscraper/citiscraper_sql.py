import urllib2
import json
import csv
import time
import sqlite3

# Initialize
lastTime = ' '
filename = 'citibike.db3'
insertString = 'INSERT INTO station_usage VALUES (?,?,?,?,?,?)'

## Connect to sqlite3 database
conn = sqlite3.connect(filename)
c = conn.cursor()

while True:
    # Get Station Data and convert it to a dict
    response = urllib2.urlopen('http://citibikenyc.com/stations/json')
    citijson = response.read()
    response.close()

    data = json.loads(citijson)

    if data['executionTime'] != lastTime:
        for station in data['stationBeanList']:
            stationid = station['id']
            availDocks = station['availableDocks']
            totalDocks = station['totalDocks']
            status = station['statusKey']
            availBikes = station['availableBikes']
            c.execute(insertString,[data['executionTime'],stationid,availBikes,availDocks,totalDocks,status])

        conn.commit()
        lastTime = data['executionTime']

    time.sleep(0.5)

