WITH image_counts AS (
    SELECT
        build_id,
        COUNT(*) AS images,
        SUM(CASE WHEN use_for_detection THEN 1 ELSE 0 END) AS labeled_images
    FROM images
    GROUP BY build_id
),
box_counts AS (
    SELECT i.build_id, COUNT(*) AS boxes
    FROM boxes b
    INNER JOIN images i ON b.image_id = i.image_id
    GROUP BY i.build_id
)
SELECT
    f.level,
    f.outer_fold,
    f.inner_fold,
    f.role,
    COUNT(*) AS builds,
    COALESCE(SUM(ic.images), 0) AS images,
    COALESCE(SUM(ic.labeled_images), 0) AS labeled_images,
    COALESCE(SUM(bc.boxes), 0) AS boxes
FROM folds f
LEFT JOIN image_counts ic ON f.build_id = ic.build_id
LEFT JOIN box_counts bc ON f.build_id = bc.build_id
GROUP BY f.level, f.outer_fold, f.inner_fold, f.role
ORDER BY f.level, f.outer_fold, COALESCE(f.inner_fold, -1), f.role
