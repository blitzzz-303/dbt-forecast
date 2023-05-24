SELECT
    store_dept_pk,
    week_date,
    weekly_sales,
    Temperature,
    Fuel_Price,
    CPI,
    Unemployment,
    IsHoliday
FROM {{ref('INT_SALES_ENHANCE')}}
WHERE 
    store in ({{ var('stores')}})
    AND dept in ({{ var('depts')}})
    AND week_date < '{{ var('test_date') }}'::date