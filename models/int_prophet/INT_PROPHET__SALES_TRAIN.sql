SELECT
    store_dept_pk,
    week_date,
    weekly_sales,
    isholiday,
    store,
    dept
FROM {{ ref('INT_SALES_ENHANCE') }}
WHERE
    week_date < '{{ var('test_date') }}'::date
