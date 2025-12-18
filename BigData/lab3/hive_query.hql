CREATE DATABASE IF NOT EXISTS steam;
USE steam;

CREATE EXTERNAL TABLE reviews (
  written_during_early_access BOOLEAN,
  playtime_forever INT
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  "separatorChar" = ",",
  "skip.header.line.count" = "1"
)
STORED AS TEXTFILE
LOCATION '/steam';

SELECT
  COUNT(*) AS users_count,
  AVG(playtime_forever)/60 AS avg_playtime_hours
FROM reviews
WHERE written_during_early_access = true;
