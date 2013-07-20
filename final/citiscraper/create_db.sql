CREATE TABLE station_usage(
    execTime   TEXT,
    stationID  INTEGER,
    availBikes INTEGER,
    availDocks INTEGER,
    totalDocks INTEGER,
    status     INTEGER,
    PRIMARY KEY(execTime,stationID));
