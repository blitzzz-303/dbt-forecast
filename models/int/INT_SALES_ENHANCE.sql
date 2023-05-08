select
    concat(store, '#', dept) store_dept_pk,
    store,
    dept,
    to_date(date, 'DD/MM/YYYY')::timestamp week_date,
    weekly_Sales,
    Temperature,
    Fuel_Price,
    CPI,
    Unemployment,
    IsHoliday
from {{ref('STG_SALE')}} sale
left join {{ref('STG_STORE')}} store using (store)
left join {{ref('STG_FEATURE')}} feature using (store, week_date)