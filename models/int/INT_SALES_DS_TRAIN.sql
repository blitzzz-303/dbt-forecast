SELECT
    store_dept_pk,
    week_date,
    iff(week_date < '{{ var('test_date') }}'::date, weekly_Sales, null) weekly_Sales
FROM {{ref('INT_SALES_DS_WEEKLY')}}
WHERE 
    ARRAY_CONTAINS(store::variant, SPLIT('{{ var('stores') }}', ','))
    AND ARRAY_CONTAINS(dept::variant, SPLIT('{{ var('depts') }}', ','))