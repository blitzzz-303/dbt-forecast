with feature_cte as (
    select
        * exclude (date),
        to_date(date, 'DD/MM/YYYY')::timestamp week_date
    from {{ref('ds2_feature_dataset')}} feature
)
select * from feature_cte