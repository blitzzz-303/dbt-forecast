{% macro generate_schema_name(custom_schema_name, node) -%}

    {%- set default_schema = target.schema -%}
    {%- if target.name[-3:] | upper == 'DEV' -%}

        {%- if custom_schema_name is none -%}

            {{ target.schema | trim }}_{{ default_schema }}

        {%- else -%}
            {{ target.schema | trim }}_{{ custom_schema_name }}

        {% endif %}    

    {%- else -%}

        {%- if custom_schema_name is none -%}

            {{ default_schema }}

        {%- else -%}
            {{ custom_schema_name }}

        {% endif %}    

    {%- endif -%}

{%- endmacro %}
