WITH
    store_cte AS (
        SELECT *
        FROM {{ ref('ds2_store_dataset') }}
    )

SELECT * FROM store_cte
