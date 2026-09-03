-- Drops the table if it already exists to allow clean resets
DROP TABLE IF EXISTS marketing_campaign;

-- Creates the core table structure with strict data types
CREATE TABLE marketing_campaign (
    user_id INTEGER PRIMARY KEY,
    test_group TEXT NOT NULL,
    converted INTEGER NOT NULL CHECK (converted IN (0, 1)),
    total_ads INTEGER NOT NULL,
    most_ads_day TEXT NOT NULL,
    most_ads_hour INTEGER NOT NULL CHECK (most_ads_hour BETWEEN 0 AND 23)
);