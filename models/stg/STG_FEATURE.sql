WITH feature_cte AS (
    SELECT
        * EXCLUDE (date),
        to_date(date, 'DD/MM/YYYY')::timestamp week_date
    FROM {{ref('ds2_feature_dataset')}}
)

SELECT * FROM feature_cte
