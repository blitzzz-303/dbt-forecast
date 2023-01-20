select
    store,
    week_date,
    'Bulgaria' country,
    iff(week_date < '2012-01-01', weekly_Sales, null) weekly_Sales
from {{ref('int_sds_weekly')}}
where Store in (1, 2) and Dept = 1