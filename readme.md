## Introduction
dbt-sf-propeht is an experimental project to enable better, more user friendly forecast for Snowflake.

The forecast solution is defined as a Stored Procedure, run via Snowpark (Snowflake integrated python environment).

This demo support 1 category for the forecast, which enable to forecast multiple product at one run. Thus, this enables user to have more granularity for analysis and improve overall accuracy.

For some reasons, this package is not compatiple with generate_schema_name from dbt. This should be fixed in later release.

Main data set we should use to test https://www.kaggle.com/code/aremoto/retail-sales-forecast/data?select=stores+data-set.csv (already included in seed)





## Usage

1. Clone this repo
2. Build this container, by using below command
    
    ``docker-compose build``
3. Start this container, by using below command
    
    ``docker-compose run --rm sf``
    
4. (optional) to generate the Snowpark function, after login to your container, use below command
    ``python scripts/sf_func_gen.py``

5. start trying the function by running ``dbt build`` statement
