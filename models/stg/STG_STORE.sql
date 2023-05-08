with store_cte as (
    select
        *
    from {{ref('ds2_store_dataset')}} store
)
select * from store_cte