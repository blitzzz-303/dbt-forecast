SELECT
    store_dept_pk,
    week_date,
    weekly_sales,
    temperature,
    fuel_price,
    cpi,
    unemployment,
    isholiday
FROM {{ref('INT_SALES_ENHANCE')}}
WHERE
    store IN ({{ var('stores') }})
    AND dept IN ({{ var('depts') }})
    AND week_date < '{{ var('test_date') }}'::date
