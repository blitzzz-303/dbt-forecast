SELECT
    sale.store_dept_pk,
    sale.store,
    sale.dept,
    sale.week_date,
    sale.weekly_sales,
    feature.temperature,
    feature.fuel_price,
    feature.cpi,
    feature.unemployment,
    feature.isholiday,
    store.type,
    store.size
FROM {{ ref('STG_SALE') }} sale
    LEFT JOIN {{ ref('STG_STORE') }} store USING (store)
    LEFT JOIN {{ ref('STG_FEATURE') }} feature USING (store, week_date)
