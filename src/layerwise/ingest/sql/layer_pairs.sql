-- name: pairs
WITH ranked AS (
    SELECT
        build_id,
        layer_index,
        modality,
        image_id,
        ROW_NUMBER() OVER (
            PARTITION BY build_id, layer_index, modality ORDER BY timestamp, image_id
        ) AS rn
    FROM images
),
pb AS (
    SELECT build_id, layer_index, image_id AS pb_image_id
    FROM ranked
    WHERE modality = 'PB' AND rn = 1
),
ot AS (
    SELECT build_id, layer_index, image_id AS ot_image_id
    FROM ranked
    WHERE modality = 'OT' AND rn = 1
)
SELECT
    COALESCE(pb.build_id, ot.build_id) AS build_id,
    COALESCE(pb.layer_index, ot.layer_index) AS layer_index,
    pb.pb_image_id,
    ot.ot_image_id,
    CASE
        WHEN pb.pb_image_id IS NOT NULL AND ot.ot_image_id IS NOT NULL THEN 'paired'
        WHEN pb.pb_image_id IS NOT NULL THEN 'pb_only'
        ELSE 'ot_only'
    END AS pair_status
FROM pb
FULL OUTER JOIN ot
    ON pb.build_id = ot.build_id AND pb.layer_index = ot.layer_index
ORDER BY 1, 2

-- name: duplicates
WITH ranked AS (
    SELECT
        build_id,
        layer_index,
        modality,
        image_id,
        ROW_NUMBER() OVER (
            PARTITION BY build_id, layer_index, modality ORDER BY timestamp, image_id
        ) AS rn,
        FIRST_VALUE(image_id) OVER (
            PARTITION BY build_id, layer_index, modality ORDER BY timestamp, image_id
        ) AS kept_image_id
    FROM images
)
SELECT build_id, layer_index, modality, image_id, kept_image_id
FROM ranked
WHERE rn > 1
ORDER BY build_id, layer_index, modality, image_id
