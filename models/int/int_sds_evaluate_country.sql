select
    w.*,
    p.country,
    p.WEEKLY_SALES__PREDICT,
    p."yhat_upper" WEEKLY_SALES__UPPER,
    p."yhat_lower" WEEKLY_SALES__LOWER
from {{ref('INT_SDS_WEEKLY')}} w
join {{ref('INT_SDS_PREDICT_COUNTRY')}} p
    on w.store = p.store
    and w.WEEK_DATE = p.WEEK_DATE::varchar::date