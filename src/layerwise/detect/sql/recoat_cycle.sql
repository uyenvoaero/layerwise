-- name: gaps
-- Median time between consecutive PB layers (layer index difference of 1) per build.
-- The timestamp parse is the only DuckDB-specific call: ingest timestamps look like
-- 20240318T133538.870000, some without fractional seconds, so the fraction is padded to
-- microseconds before strptime.
WITH pb AS (
    SELECT
        build_id,
        layer_index,
        strptime(
            CASE
                WHEN position('.' IN timestamp) > 0
                    THEN rpad(timestamp, CAST(position('.' IN timestamp) + 6 AS INTEGER), '0')
                ELSE timestamp || '.000000'
            END,
            '%Y%m%dT%H%M%S.%f'
        ) AS ts
    FROM images
    WHERE modality = 'PB' AND timestamp IS NOT NULL
),
first_per_layer AS (
    SELECT
        build_id,
        layer_index,
        ts,
        ROW_NUMBER() OVER (PARTITION BY build_id, layer_index ORDER BY ts) AS rn
    FROM pb
),
ordered AS (
    SELECT
        build_id,
        layer_index,
        ts,
        LAG(layer_index) OVER (PARTITION BY build_id ORDER BY layer_index) AS previous_layer,
        LAG(ts) OVER (PARTITION BY build_id ORDER BY layer_index) AS previous_ts
    FROM first_per_layer
    WHERE rn = 1
),
gaps AS (
    SELECT
        build_id,
        EXTRACT(EPOCH FROM ts) - EXTRACT(EPOCH FROM previous_ts) AS gap_s
    FROM ordered
    WHERE layer_index - previous_layer = 1
)
SELECT
    build_id,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY gap_s) AS median_gap_s,
    COUNT(*) AS gaps_used
FROM gaps
GROUP BY build_id
ORDER BY build_id;
