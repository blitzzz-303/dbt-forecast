{{ config(
    pre_hook=[
      run_predict("{{ref('int_sds_train')}}", 'WEEK_DATE', 'STORE', 'WEEKLY_SALES')
    ]
) }}

select
  *
from {{this}}
