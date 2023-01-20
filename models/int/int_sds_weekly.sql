select
    store,
    Dept,
    to_date(date, 'DD/MM/YYYY')::timestamp week_date,
    weekly_Sales
from {{ref('stg_sds_enhanced')}} sde