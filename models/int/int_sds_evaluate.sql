select
    w.*,
    p.WEEKLY_SALES__PREDICT,
    p."yhat_upper" WEEKLY_SALES__UPPER,
    p."yhat_lower" WEEKLY_SALES__LOWER
from {{ref('int_sds_weekly')}} w
join {{ref('int_sds_predict')}} p
    on w.store = p.store
    and w.WEEK_DATE = p.WEEK_DATE::varchar::date