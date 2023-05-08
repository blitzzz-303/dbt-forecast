
-- depends_on: {{ ref('INT_PROPHET_UDF') }}
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
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY weekly_Sales) OVER (PARTITION BY store_dept_pk) p75,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY weekly_Sales) OVER (PARTITION BY store_dept_pk) p25,
        -- perform ML inference
        weekly_Sales,
        iff(True,
        PARSE_JSON(
            PREDICT_SALES(
                object_construct(
                    'STORE_DEPT_PK', store_dept_pk,
                    'ds', week_date,
                    'cap', p75,
                    'floor', p25
                ))), NULL) pred,
        pred:yhat::float sales_predict,
        pred:yhat_lower::float sales_lower,
        pred:yhat_upper::float sales_upper
    FROM {{ ref('INT_SALES_ENHANCE') }} W
    WHERE 
        store in ({{ var('stores')}})
        AND dept in ({{ var('depts')}})
)
SELECT
    * EXCLUDE (pred)
FROM sale_and_predict