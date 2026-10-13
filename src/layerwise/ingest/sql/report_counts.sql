-- name: images_by_build
SELECT
    build_id,
    modality,
    COUNT(*) AS images,
    SUM(CASE WHEN label_status = 'labeled' THEN 1 ELSE 0 END) AS labeled,
    SUM(CASE WHEN label_status = 'labeled_no_defects' THEN 1 ELSE 0 END) AS labeled_no_defects,
    SUM(CASE WHEN label_status = 'unlabeled' THEN 1 ELSE 0 END) AS unlabeled,
    SUM(CASE WHEN size_flag THEN 1 ELSE 0 END) AS size_flagged
FROM images
GROUP BY build_id, modality
ORDER BY build_id, modality

-- name: boxes_by_class
SELECT
    source_class,
    COUNT(*) AS ingested,
    SUM(CASE WHEN clipped THEN 1 ELSE 0 END) AS clipped
FROM boxes
GROUP BY source_class
ORDER BY source_class

-- name: boxes_by_phase
SELECT
    phase_category,
    COUNT(*) AS boxes
FROM boxes
GROUP BY phase_category
ORDER BY phase_category

-- name: unpaired_by_build
SELECT build_id, 'PB' AS modality, COUNT(*) AS unpaired
FROM layer_pairs
WHERE pair_status = 'pb_only'
GROUP BY build_id
UNION ALL
SELECT build_id, 'OT' AS modality, COUNT(*) AS unpaired
FROM layer_pairs
WHERE pair_status = 'ot_only'
GROUP BY build_id
ORDER BY 1, 2
