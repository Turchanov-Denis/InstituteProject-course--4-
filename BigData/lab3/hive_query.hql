CREATE DATABASE IF NOT EXISTS steam;
USE steam;

DROP TABLE IF EXISTS reviews;

CREATE EXTERNAL TABLE reviews (
  anonymous STRING,
  app_id STRING,
  app_name STRING,
  review_id STRING,
  language STRING,
  review STRING,
  timestamp_created STRING,
  timestamp_updated STRING,
  recommended STRING,
  votes_helpful STRING,
  votes_funny STRING,
  weighted_vote_score STRING,
  comment_count STRING,
  steam_purchase STRING,
  received_for_free STRING,
  written_during_early_access STRING,
  author_steamid STRING,
  author_num_games_owned STRING,
  author_num_reviews STRING,
  author_playtime_forever STRING,
  author_playtime_last_two_weeks STRING,
  author_playtime_at_review STRING,
  author_last_played STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar'     = '\"',
  'escapeChar'    = '\\',
  'skip.header.line.count' = '1'
)
STORED AS TEXTFILE
LOCATION '/steam';

SELECT anonymous, app_id, author_playtime_forever, written_during_early_access
FROM reviews
LIMIT 10;

SELECT
  COUNT(*) AS users_count,
  AVG(CAST(author_playtime_forever AS DOUBLE) / 60) AS avg_playtime_hours
FROM reviews
WHERE written_during_early_access = 'True';
