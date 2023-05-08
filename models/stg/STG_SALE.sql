with sale_cte as (
    select
        *,
        to_date(date, 'DD/MM/YYYY')::timestamp week_date,
        concat(store, '#', dept) store_dept_pk
    from {{ref('ds2_sales_dataset')}} sale
)
select * from sale_cte