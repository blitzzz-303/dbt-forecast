select
    store_dept_pk,
    week_date,
    iff(week_date < '2012-01-01', weekly_Sales, null) weekly_Sales
from {{ref('INT_SALES_DS_WEEKLY')}}
where Store in (1, 2) and Dept = 1