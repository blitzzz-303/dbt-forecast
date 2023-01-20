select
    w.*,
    p.WEEKLY_SALES__PREDICT,
    p."yhat_upper" WEEKLY_SALES__UPPER,
    p."yhat_lower" WEEKLY_SALES__LOWER
from {{ref('INT_SALES_DS_WEEKLY')}} w
join {{ref('INT_SALES_DS_PREDICT')}} p
    on w.store = p.store
    and w.WEEK_DATE = p.WEEK_DATE::varchar::date