-- name: pooled
-- Pooled counts and metrics per model over the finished outer folds. `fold_results` holds one row
-- per (model, outer fold) with the held-out counts; each fold was evaluated at its own threshold,
-- so precision, recall and F1 come from the summed counts.
WITH agg AS (
    SELECT
        model,
        COUNT(*) AS folds,
        SUM(images) AS images,
        SUM(boxes) AS boxes,
        SUM(predictions) AS predictions,
        SUM(matches) AS matches,
        SUM(boxes_small) AS boxes_small,
        SUM(hits_small) AS hits_small,
        SUM(boxes_medium) AS boxes_medium,
        SUM(hits_medium) AS hits_medium,
        SUM(boxes_large) AS boxes_large,
        SUM(hits_large) AS hits_large
    FROM fold_results
    GROUP BY model
),
rates AS (
    SELECT
        *,
        CASE WHEN predictions = 0 THEN 0.0 ELSE CAST(matches AS DOUBLE) / predictions END AS precision,
        CASE WHEN boxes = 0 THEN NULL ELSE CAST(matches AS DOUBLE) / boxes END AS recall
    FROM agg
)
SELECT
    model,
    folds,
    images,
    boxes,
    precision,
    recall,
    CASE
        WHEN recall IS NULL THEN NULL
        WHEN precision + recall = 0 THEN 0.0
        ELSE 2 * precision * recall / (precision + recall)
    END AS f1,
    CASE WHEN boxes_small = 0 THEN NULL ELSE CAST(hits_small AS DOUBLE) / boxes_small END AS recall_small,
    CASE WHEN boxes_medium = 0 THEN NULL ELSE CAST(hits_medium AS DOUBLE) / boxes_medium END AS recall_medium,
    CASE WHEN boxes_large = 0 THEN NULL ELSE CAST(hits_large AS DOUBLE) / boxes_large END AS recall_large
FROM rates
ORDER BY model;

-- name: size_flags
-- PB images per build whose size differs from the expected size (the ingest `size_flag`).
SELECT build_id, COUNT(*) FILTER (WHERE size_flag) AS flagged
FROM images
WHERE modality = 'PB'
GROUP BY build_id
ORDER BY build_id;
