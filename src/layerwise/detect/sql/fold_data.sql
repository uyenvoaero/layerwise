-- name: labeled_images
-- PB images that carry a label file (labeled or labeled_no_defects); unlabeled images are excluded.
SELECT image_id, build_id, layer_index, path, width, height, size_flag
FROM images
WHERE modality = 'PB' AND label_status IN ('labeled', 'labeled_no_defects')
ORDER BY build_id, layer_index, image_id;

-- name: labeled_boxes
SELECT b.image_id, b.xmin, b.ymin, b.xmax, b.ymax
FROM boxes b
INNER JOIN images i ON b.image_id = i.image_id
WHERE i.modality = 'PB' AND i.label_status IN ('labeled', 'labeled_no_defects')
ORDER BY b.image_id, b.box_id;

-- name: labeled_counts
-- Labeled PB images and boxes per build; the image count drives the 150-image flag.
WITH image_counts AS (
    SELECT build_id, COUNT(*) AS images
    FROM images
    WHERE modality = 'PB' AND label_status IN ('labeled', 'labeled_no_defects')
    GROUP BY build_id
),
box_counts AS (
    SELECT i.build_id, COUNT(*) AS boxes
    FROM boxes b
    INNER JOIN images i ON b.image_id = i.image_id
    WHERE i.modality = 'PB' AND i.label_status IN ('labeled', 'labeled_no_defects')
    GROUP BY i.build_id
)
SELECT ic.build_id, ic.images, COALESCE(bc.boxes, 0) AS boxes
FROM image_counts ic
LEFT JOIN box_counts bc ON ic.build_id = bc.build_id
ORDER BY ic.build_id;

-- name: pb_images
-- Every PB layer image of the data set, labeled or not (prediction covers unlabeled layers too).
SELECT image_id, build_id, layer_index, path, width, height, size_flag
FROM images
WHERE modality = 'PB'
ORDER BY build_id, layer_index, image_id;
