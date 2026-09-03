-- Create Staging Table
CREATE TABLE ab_campaign_logs (
    user_id INT PRIMARY KEY,
    test_group VARCHAR(10),
    converted INT,
    total_ads INT,
    most_ads_day VARCHAR(15),
    most_ads_hour INT
);

-- Advanced Query: Window Functions & Conversion Rates per Variant Group
WITH VariantMetrics AS (
    SELECT 
        test_group,
        COUNT(user_id) AS total_users,
        SUM(converted) AS total_conversions,
        ROUND(AVG(total_ads), 2) AS avg_ads_per_user
    FROM ab_campaign_logs
    GROUP BY test_group
)
SELECT 
    test_group,
    total_users,
    total_conversions,
    ROUND((total_conversions::DECIMAL / total_users) * 100, 2) AS conversion_rate_pct,
    avg_ads_per_user,
    -- Rank variant groups by conversion performance
    DENSE_RANK() OVER (ORDER BY (total_conversions::DECIMAL / total_users) DESC) AS rank_by_cr
FROM VariantMetrics;