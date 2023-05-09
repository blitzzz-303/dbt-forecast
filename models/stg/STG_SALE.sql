WITH sale_cte AS (
    SELECT
        *,
        to_date(date, 'DD/MM/YYYY')::timestamp week_date,
        concat(store, '#', dept) store_dept_pk
    FROM {{ref('ds2_sales_dataset')}}
)

SELECT * FROM sale_cte
