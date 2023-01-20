select
    *
from {{ref('ds2_sales_dataset')}} sale
left join {{ref('ds2_store_dataset')}} store using (store)