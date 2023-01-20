{{ config(
    pre_hook=[
      run_predict("{{ref('INT_SDS_TRAIN_COUNTRY')}}", 'WEEK_DATE', 'STORE', 'WEEKLY_SALES')
    ]
) }}

select
  *
from {{this}}
