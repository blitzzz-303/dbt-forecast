{{ config(
    pre_hook=[
      run_predict("{{ref('INT_SALES_DS_TRAIN')}}", 'WEEK_DATE', 'STORE_DEPT_PK', 'WEEKLY_SALES', 'True')
    ]
) }}

select
  *
from {{this}}
