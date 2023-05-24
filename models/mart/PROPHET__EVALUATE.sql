
-- depends_on: {{ ref('INT_PROPHET__UDF') }}
{{
    config(
        materialized='view'
    )
}}

WITH
    sale_and_predict AS (
        SELECT
            store_dept_pk,
            store,
            dept,
            week_date,
            -- perform ML inference
            weekly_sales,
            parse_json(
                    predict_sales(
                        object_construct(
                            'STORE_DEPT_PK', store_dept_pk,
                            'ds', week_date
                        )
                    )
                ) predictions,
            predictions:yhat::float sales_predict,
            predictions:yhat_lower::float sales_lower,
            predictions:yhat_upper::float sales_upper
        FROM {{ ref('INT_SALES_ENHANCE') }}
        WHERE
            store IN ({{ var('stores') }})
            AND dept IN ({{ var('depts') }})
    ),

sale_and_predict_monthly as (
    SELECT
        store_dept_pk,
        store,
        dept,
        date_trunc('month', week_date) month_date,
        sum(weekly_sales) monthly_sales,
        sum(sales_predict) monthly_sales_predict,
        sum(sales_lower) monthly_sales_lower,
        sum(sales_upper) monthly_sales_upper,
        -- calculate MAPE
        abs(sum(weekly_sales) - sum(sales_predict)) / sum(weekly_sales) monthly_sales_error
    FROM sale_and_predict
    GROUP BY 1, 2, 3, 4
)
SELECT *
FROM sale_and_predict_monthly