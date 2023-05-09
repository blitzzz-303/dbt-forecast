
-- depends_on: {{ ref('INT_PROPHET__UDF') }}
{{
    config(
        materialized='view'
    )
}}
WITH sale_and_predict AS (
    SELECT
        store_dept_pk,
        store,
        dept,
        week_date,
        -- calculate p75 and p25
        percentile_cont(
            0.75
        ) WITHIN GROUP (ORDER BY weekly_sales) OVER (PARTITION BY store_dept_pk) p75,
        percentile_cont(
            0.25
        ) WITHIN GROUP (ORDER BY weekly_sales) OVER (PARTITION BY store_dept_pk) p25,
        -- perform ML inference
        weekly_sales,
        iff(true,
            parse_json(
            predict_sales(
                object_construct(
                    'STORE_DEPT_PK', store_dept_pk,
                    'ds', week_date,
                    'cap', p75,
                    'floor', p25
                ))), null) pred,
        pred:yhat::float sales_predict,
        pred:yhat_lower::float sales_lower,
        pred:yhat_upper::float sales_upper
    FROM {{ ref('INT_SALES_ENHANCE') }}
    WHERE
        store IN ({{ var('stores') }})
        AND dept IN ({{ var('depts') }})
)

SELECT * EXCLUDE (pred)
FROM sale_and_predict
